"""
==============================================================================
run_qpso_qmg_mpi.py  —  CUDA-Q 0.7.1 + AE-SOQPSO + MPI 並行主入口
==============================================================================

整合三篇論文的優勢：
  [Chen et al. 2025, JCTC]     QMG 動態電路 + chemistry constraint
  [Xiao et al. 2026, SQMG]     tensornet GPU 後端（記憶體高效，速度最快）
  [Tseng et al. 2024, AE-QTS]  調和加權 mbest + best-worst 配對更新

架構設計（解決 v9.6 的兩個根本問題）：

  問題 1：CUDA pinned memory 洩漏（每次評估 +2.5 GB）
    v9.6 解法：subprocess 隔離，每次評估啟動新 process，~3s 開銷/eval
    MPI 解法 ：每個 MPI rank = 獨立 OS process，CUDA context 隔離，
              process 初始化只做一次，0 subprocess 開銷
    根本原因 ：cudaMallocHost（pinned memory）綁定到 CUDA context，
              context 隨 process 結束自動釋放，MPI rank 生命週期等於 job

  問題 2：串行評估速度瓶頸（105s/eval × 20 粒子 = 35min/iter）
    v9.6：單 GPU 串行評估，所有粒子排隊等候
    MPI ：M 個粒子分配到 N_RANK 個 GPU 並行評估
          理論加速比 = min(N_RANK, M) = 8（8 卡 V100）
          實際加速（50 粒子 / 8 GPU）：
            每 rank 評估 ceil(50/8)=7 粒子 × 105s = 735s/iter
            vs 串行 50 × 105s = 5250s/iter → 加速 7.1 倍

通訊模式：
  1. Rank 0 計算新位置矩陣（M×D），broadcast 到所有 rank
  2. 各 rank 評估 positions[rank::N_RANK]（round-robin 負載均衡）
  3. allgather 回收所有結果，rank 0 更新 pbest/gbest

Backend 選擇：
  cudaq_tensornet（推薦）：SQMG 論文實測 N=8 比 cuStateVec 快 ~270 倍
  cudaq_nvidia   （穩定）：V100 sm_70 最穩定，已完整驗證
  cudaq_qpp      （備用）：純 CPU，功能測試用

放置位置：run_qpso_qmg_mpi.py（專案根目錄）

使用方式：
  mpirun -n 8 python run_qpso_qmg_mpi.py \
      --backend cudaq_tensornet \
      --particles 50 --iterations 40 \
      --task_name unconditional_9_ae_mpi

SLURM：sbatch cutn-qmg_mpi_8g.slurm
==============================================================================
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

# ── MPI 初始化（必須在 cudaq import 前完成）──────────────────────────────────
try:
    from mpi4py import MPI
    _COMM   = MPI.COMM_WORLD
    _RANK   = _COMM.Get_rank()
    _NRANK  = _COMM.Get_size()
    _HAS_MPI = True
except ImportError:
    _COMM    = None
    _RANK    = 0
    _NRANK   = 1
    _HAS_MPI = False
    if _RANK == 0:
        print("[WARN] mpi4py 未安裝，以單 rank 模式執行。", flush=True)

# ── GPU 綁定：每個 rank 綁定自己的 GPU（必須在 import cudaq 之前）────────────
_ORIGINAL_CUDA_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _ORIGINAL_CUDA_DEVICES:
    # 支援逗號分隔清單，例如 "0,1,2,3,4,5,6,7"
    _dev_list = [d.strip() for d in _ORIGINAL_CUDA_DEVICES.split(",") if d.strip()]
    _my_gpu   = _dev_list[_RANK % len(_dev_list)]
else:
    # 未設定時 rank k 對應 GPU k
    _my_gpu = str(_RANK)
os.environ["CUDA_VISIBLE_DEVICES"] = _my_gpu

# ── 現在才 import cudaq ──────────────────────────────────────────────────────
try:
    import cudaq
except ImportError:
    if _RANK == 0:
        print("[ERROR] 無法 import cudaq。請安裝：pip install cuda-quantum-cu12==0.7.1")
    sys.exit(1)

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

try:
    from qmg.generator_cudaq import MoleculeGeneratorCUDAQ
    from qmg.utils.weight_generator import ConditionalWeightsGenerator
except ImportError as e:
    if _RANK == 0:
        print(f"[ERROR] 無法 import qmg：{e}")
    sys.exit(1)

try:
    from qpso_optimizer_ae import AESOQPSOOptimizer
except ImportError as e:
    if _RANK == 0:
        print(f"[ERROR] 無法 import qpso_optimizer_ae：{e}")
    sys.exit(1)


# ===========================================================================
# 工具函式
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QMG CUDA-Q + AE-SOQPSO MPI 並行版（融合 SQMG + AE-QTS）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 分子生成參數
    p.add_argument("--num_heavy_atom",   type=int,   default=9)
    p.add_argument("--num_sample",       type=int,   default=10000)
    # QPSO 參數
    p.add_argument("--particles",        type=int,   default=50,
                   help="粒子總數 M（建議為 N_RANK 整數倍）")
    p.add_argument("--iterations",       type=int,   default=40,
                   help="最大迭代次數 T")
    p.add_argument("--alpha_max",        type=float, default=1.2)
    p.add_argument("--alpha_min",        type=float, default=0.4)
    p.add_argument("--mutation_prob",    type=float, default=0.15)
    p.add_argument("--stagnation_limit", type=int,   default=8)
    p.add_argument("--reinit_fraction",  type=float, default=0.20)
    # AE-QTS 參數
    p.add_argument("--ae_weighting",     action="store_true", default=True,
                   help="AE-QTS 調和加權 mbest（預設開啟）")
    p.add_argument("--no_ae_weighting",  action="store_false", dest="ae_weighting",
                   help="關閉 AE 調和加權（退化為標準 SOQPSO）")
    p.add_argument("--pair_interval",    type=int,   default=5,
                   help="AE 配對更新間隔（迭代數），0=關閉")
    p.add_argument("--rotate_factor",    type=float, default=0.01,
                   help="AE 配對更新幅度（對應 AE-QTS 的 Δθ）")
    # Backend
    p.add_argument(
        "--backend", type=str, default="cudaq_tensornet",
        choices=["cudaq_tensornet", "cudaq_nvidia", "cudaq_qpp",
                 "cudaq_nvidia_fp64", "cudaq_tensornet_mps"],
        help="cudaq_tensornet：SQMG 推薦，速度最快；cudaq_nvidia：最穩定",
    )
    # 其他
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--task_name",        type=str,   default="unconditional_9_ae_mpi")
    p.add_argument("--data_dir",         type=str,   default="results_mpi")
    return p.parse_args()


def setup_logger(log_path: str) -> logging.Logger:
    """只有 rank 0 建立有效 logger；其他 rank 回傳 NullLogger。"""
    if _RANK != 0:
        null = logging.getLogger(f"null_rank{_RANK}")
        null.addHandler(logging.NullHandler())
        return null

    logger = logging.getLogger("AEMPILogger")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s,%(msecs)03d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for h in [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def get_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024
    except Exception:
        pass
    return -1.0


def log_gpu_info(logger: logging.Logger) -> None:
    if _RANK != 0:
        return
    try:
        import subprocess as _sp
        out = _sp.check_output(
            "nvidia-smi --query-gpu=index,name,memory.total,driver_version "
            "--format=csv,noheader",
            shell=True, stderr=_sp.DEVNULL,
        ).decode().strip()
        for line in out.splitlines():
            logger.info(f"  GPU: {line.strip()}")
    except Exception:
        logger.info("  GPU info: 無法取得")

    try:
        import re
        ver_str = cudaq.__version__
        m = re.search(r'(\d+\.\d+\.\d+)', ver_str)
        logger.info(f"  CUDA-Q: {m.group(1) if m else ver_str}")
    except Exception:
        pass
    logger.info(f"  CUDA_VISIBLE_DEVICES（原始）: {_ORIGINAL_CUDA_DEVICES}")
    logger.info(f"  MPI N_RANK: {_NRANK}  各 rank GPU: "
                f"rank k → GPU {_dev_list[0] if _ORIGINAL_CUDA_DEVICES else 'k'}")


# ===========================================================================
# MPI 批次評估函式
# ===========================================================================

def make_mpi_batch_evaluate_fn(
    gen:    "MoleculeGeneratorCUDAQ",
    cwg:    "ConditionalWeightsGenerator",
    args:   argparse.Namespace,
    logger: logging.Logger,
) -> callable:
    """
    建立 MPI 並行批次評估函式。

    這個函式被傳入 AESOQPSOOptimizer 作為 batch_evaluate_fn。
    每次被 optimizer 呼叫時（只在 rank 0 呼叫）：
      1. Rank 0 觸發進入點，執行 bcast 通知其他 rank 進入評估階段
      2. 所有 rank 接收位置矩陣
      3. 各 rank 按 round-robin 評估 positions[rank::nrank]
      4. allgather 回收所有結果，rank 0 組裝並回傳

    注意：批次評估函式在 MPI 環境下必須由所有 rank「同時」進入，
    因此在主迴圈外需要包裹一個 allreduce 同步訊號。
    這裡採用簡單的 bcast 設計：rank 0 bcast positions 矩陣，
    其他 rank 在同步迴圈中等待 bcast，positions=None 表示結束信號。
    """
    def batch_evaluate_fn(positions: np.ndarray) -> list:
        # positions shape: (M, D)，只在 rank 0 有效的輸入
        return _mpi_evaluate_all(gen, cwg, args, positions)

    return batch_evaluate_fn


def _mpi_evaluate_all(
    gen:       "MoleculeGeneratorCUDAQ",
    cwg:       "ConditionalWeightsGenerator",
    args:      argparse.Namespace,
    positions: np.ndarray,   # (M, D)，由 rank 0 傳入
) -> list:
    """
    所有 rank 執行的批次評估核心。
    positions 在 rank 0 是有效的 (M,D) 矩陣；其他 rank 會透過 bcast 取得。
    """
    # Broadcast 位置矩陣（rank 0 傳出，其他 rank 接收）
    positions = _COMM.bcast(positions if _RANK == 0 else None, root=0)
    M = positions.shape[0]

    # Round-robin 分配：rank k 評估 positions[k, k+n, k+2n, ...]
    my_indices = list(range(_RANK, M, _NRANK))
    my_results = []

    for idx in my_indices:
        w    = positions[idx]
        w_c  = cwg.apply_chemistry_constraint(w.copy())
        gen.update_weight_vector(w_c)
        try:
            _, v, u = gen.sample_molecule(args.num_sample)
        except Exception as e:
            print(f"[Rank {_RANK}] sample_molecule 失敗（idx={idx}）：{e}", flush=True)
            v, u = 0.0, 0.0
        my_results.append((float(v), float(u), idx))

    # Allgather：每個 rank 的結果列表
    all_scattered = _COMM.allgather(my_results)

    # 重組回原始粒子順序
    ordered: list = [None] * M
    for rank_results in all_scattered:
        for v, u, idx in rank_results:
            ordered[idx] = (v, u)

    # 確保沒有 None（應不會發生，但防禦性處理）
    for i, r in enumerate(ordered):
        if r is None:
            ordered[i] = (0.0, 0.0)

    return ordered


# ===========================================================================
# 主程式
# ===========================================================================

def main():
    args = parse_args()

    # ── rank 0 建立輸出目錄與 logger ─────────────────────────────────────────
    if _RANK == 0:
        os.makedirs(args.data_dir, exist_ok=True)
    if _HAS_MPI:
        _COMM.Barrier()   # 確保目錄建立完成後其他 rank 才繼續

    log_path = os.path.join(args.data_dir, f"{args.task_name}.log")
    logger   = setup_logger(log_path)

    if _RANK == 0:
        logger.info(f"Task name: {args.task_name}")
        logger.info(f"Task: ['validity', 'uniqueness']")
        logger.info(f"Condition: ['None', 'None']")
        logger.info(f"objective: ['maximize', 'maximize']")
        logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
        logger.info(f"# of samples: {args.num_sample}")
        logger.info(f"smarts: None")
        logger.info(f"CUDA-Q backend: {args.backend}")
        logger.info(f"MPI ranks: {_NRANK}  (此 rank: {_RANK}，GPU: {_my_gpu})")
        logger.info(f"AE-QTS 調和加權 mbest: {'開啟' if args.ae_weighting else '關閉'}")
        logger.info(f"AE 配對更新間隔: {args.pair_interval} 迭代")
        log_gpu_info(logger)
        logger.info(f"[MEM] 啟動時 RSS={get_rss_mb():.0f} MB (rank 0)")

    # ── 所有 rank 初始化 ConditionalWeightsGenerator ─────────────────────────
    cwg = ConditionalWeightsGenerator(
        args.num_heavy_atom,
        smarts=None,
        disable_connectivity_position=[],
    )
    D = cwg.length_all_weight_vector   # 134

    if _RANK == 0:
        logger.info(f"Number of flexible parameters: {D}")

    # ── 所有 rank 初始化各自的 MoleculeGeneratorCUDAQ ─────────────────────────
    # 每個 rank 已透過 os.environ 綁定自己的 GPU
    # tensornet 後端：比 cuStateVec GPU 快約 270 倍（SQMG 論文 N=8 實測）
    gen = MoleculeGeneratorCUDAQ(
        num_heavy_atom            = args.num_heavy_atom,
        all_weight_vector         = np.zeros(D),   # 佔位
        backend_name              = args.backend,
        remove_bond_disconnection = True,
        chemistry_constraint      = True,
    )

    if _RANK == 0:
        logger.info(f"[Rank {_RANK}] Generator 初始化完成（backend={args.backend}）")

    # ── 驗證 MPI 環境（5 shots 快速測試）─────────────────────────────────────
    if _RANK == 0:
        logger.info("[MPI] 執行功能驗證（5 shots × N_RANK）...")

    w_test = cwg.generate_conditional_random_weights(random_seed=99)
    test_positions = np.tile(w_test, (_NRANK, 1))   # N_RANK 個相同粒子
    test_start = time.time()
    test_results = _mpi_evaluate_all(gen, cwg, args.__class__(
        num_heavy_atom=args.num_heavy_atom,
        num_sample=5,
    ) if False else type('Tmp', (), {'num_sample': 5, 'num_heavy_atom': args.num_heavy_atom})(),
        test_positions,
    )
    test_elapsed = time.time() - test_start

    if _RANK == 0:
        any_nonzero = any(v > 0 or u > 0 for v, u in test_results)
        logger.info(
            f"[MPI] 功能驗證完成（{test_elapsed:.1f}s）  "
            f"{'✓ 有效結果' if any_nonzero else '⚠ 全部 V=U=0，請確認 backend'}"
        )
        logger.info(f"[MEM] 功能驗證後 RSS={get_rss_mb():.0f} MB (rank 0)")

    if _HAS_MPI:
        _COMM.Barrier()

    # ── 建立 MPI 批次評估函式 ─────────────────────────────────────────────────
    batch_eval_fn = make_mpi_batch_evaluate_fn(gen, cwg, args, logger)

    # ── 建立 AE-SOQPSO 優化器（只有 rank 0 真正使用，但所有 rank 都建立）──────
    # 注意：optimize() 只能由 rank 0 呼叫，但 batch_eval_fn 內部會觸發 allgather
    # 因此 rank 0 呼叫 optimize() 時，其他 rank 需要「同步進入」批次評估函式
    # 解法：用 MPI 同步迴圈讓非 rank 0 的 rank 在評估函式等待

    if _RANK == 0:
        optimizer = AESOQPSOOptimizer(
            n_params           = D,
            n_particles        = args.particles,
            max_iterations     = args.iterations,
            logger             = logger,
            batch_evaluate_fn  = batch_eval_fn,
            seed               = args.seed,
            alpha_max          = args.alpha_max,
            alpha_min          = args.alpha_min,
            data_dir           = args.data_dir,
            task_name          = args.task_name,
            stagnation_limit   = args.stagnation_limit,
            reinit_fraction    = args.reinit_fraction,
            mutation_prob      = args.mutation_prob,
            ae_weighting       = args.ae_weighting,
            pair_interval      = args.pair_interval,
            rotate_factor      = args.rotate_factor,
        )
        total_evals = args.particles * (args.iterations + 1)
        logger.info(
            f"[QPSO config] M={args.particles}  T={args.iterations}  "
            f"total_evals={total_evals}  seed={args.seed}"
        )

    # ── 執行優化（rank 0 驅動，其他 rank 在 allgather 中等待）──────────────────
    # 由於 batch_eval_fn 在 rank 0 呼叫時會觸發 bcast + allgather，
    # 其他 rank 必須在同一時間進入相同的 _mpi_evaluate_all 函式。
    # 這由 MPI 通訊的阻塞語義保證：bcast 會等到所有 rank 都執行到 bcast 才繼續。
    #
    # 為讓其他 rank 持續在「等待評估」狀態，我們在 rank != 0 時執行一個
    # 同步接收迴圈，直到 rank 0 廣播結束信號（None）為止。

    if _RANK == 0:
        best_params, best_fitness = optimizer.optimize()
        # 發送結束信號給其他 rank
        if _HAS_MPI and _NRANK > 1:
            _COMM.bcast(None, root=0)   # None = 結束信號
    else:
        # 非 rank 0：持續等待評估請求，直到收到 None
        while True:
            positions = _COMM.bcast(None, root=0)
            if positions is None:
                break    # 收到結束信號，退出迴圈
            # 執行本 rank 的評估
            M = positions.shape[0]
            my_indices = list(range(_RANK, M, _NRANK))
            my_results = []
            for idx in my_indices:
                w   = positions[idx]
                w_c = cwg.apply_chemistry_constraint(w.copy())
                gen.update_weight_vector(w_c)
                try:
                    _, v, u = gen.sample_molecule(args.num_sample)
                except Exception:
                    v, u = 0.0, 0.0
                my_results.append((float(v), float(u), idx))
            _COMM.allgather(my_results)

    # ── 最終輸出（只有 rank 0）────────────────────────────────────────────────
    if _RANK == 0:
        best_npy = os.path.join(args.data_dir, f"{args.task_name}_best_params.npy")
        np.save(best_npy, best_params)
        logger.info(f"最佳參數已儲存: {best_npy}")
        logger.info(
            f"最終結果: V×U={best_fitness:.6f}  "
            + ("✓ 超越 BO 基線 0.8834!" if best_fitness > 0.8834
               else "✗ 未超越 — 建議增加 --particles 或 --iterations")
        )
        logger.info(f"[MEM] 程序結束前 RSS={get_rss_mb():.0f} MB (rank 0)")

    if _HAS_MPI:
        MPI.Finalize()


if __name__ == "__main__":
    main()
