"""
==============================================================================
run_qpso_qmg_mpi.py  —  CUDA-Q 0.7.1 + AE-SOQPSO + MPI 並行主入口（v1.1 修正版）
==============================================================================

v1.0 → v1.1 修正：

  ★ [BUG-FIX] MPI 結束信號與通訊對稱性問題：
      v1.0 的設計中，rank 0 透過 batch_evaluate_fn → _mpi_evaluate_all
      觸發 bcast，非 rank 0 在 while 迴圈中接收。但：
      (1) 結束信號的 bcast(None) 是 rank 0 在 main() 末尾「直接」呼叫，
          而非透過 _mpi_evaluate_all；
      (2) 非 rank 0 的 while 迴圈手動複製了 _mpi_evaluate_all 的邏輯，
          兩份程式碼必須保持完全同步，有維護風險。
      (3) 若 rank 0 在 optimize() 內部發生例外，不會發送結束信號，
          導致非 rank 0 永久阻塞在 bcast。

      v1.1 修正方案：
        - _mpi_evaluate_all 由「僅 rank 0 呼叫」改為「所有 rank 同時呼叫」
        - 加入 _mpi_signal_stop() 統一管理結束信號（包含 try/finally）
        - 非 rank 0 不再手動複製評估邏輯，改為直接呼叫 _mpi_evaluate_all
        - 在 rank 0 的 optimize() 外用 try/finally 確保結束信號必然發送
        - 「繼續」或「結束」由 bcast 傳遞的 flag 陣列（而非 None）決定，
          更明確且不依賴 Python 的 None bcast 行為

整合三篇論文的優勢：
  [Chen et al. 2025, JCTC]     QMG 動態電路 + chemistry constraint
  [Xiao et al. 2026, SQMG]     tensornet GPU 後端
  [Tseng et al. 2024, AE-QTS]  v1.1 修正版調和加權 + 配對更新

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

# ── MPI 初始化（必須在 cudaq import 前完成）────────────────────────────────
try:
    from mpi4py import MPI
    _COMM    = MPI.COMM_WORLD
    _RANK    = _COMM.Get_rank()
    _NRANK   = _COMM.Get_size()
    _HAS_MPI = True
except ImportError:
    _COMM    = None
    _RANK    = 0
    _NRANK   = 1
    _HAS_MPI = False
    if _RANK == 0:
        print("[WARN] mpi4py 未安裝，以單 rank 模式執行。", flush=True)

# ── GPU 綁定（必須在 import cudaq 之前）────────────────────────────────────
_ORIGINAL_CUDA_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _ORIGINAL_CUDA_DEVICES:
    _dev_list = [d.strip() for d in _ORIGINAL_CUDA_DEVICES.split(",") if d.strip()]
    _my_gpu   = _dev_list[_RANK % len(_dev_list)]
else:
    _dev_list = []
    _my_gpu   = str(_RANK)
os.environ["CUDA_VISIBLE_DEVICES"] = _my_gpu

# ── import cudaq ─────────────────────────────────────────────────────────────
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
# MPI 通訊常數
# ===========================================================================

# ★ v1.1：用整數 flag 取代 None，更明確且不依賴 Python bcast 對 None 的行為
_MPI_FLAG_CONTINUE = 1   # 繼續評估
_MPI_FLAG_STOP     = 0   # 結束信號


# ===========================================================================
# 工具函式
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QMG CUDA-Q + AE-SOQPSO MPI 並行版 v1.1（修正 MPI 通訊）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num_heavy_atom",   type=int,   default=9)
    p.add_argument("--num_sample",       type=int,   default=10000)
    p.add_argument("--particles",        type=int,   default=50)
    p.add_argument("--iterations",       type=int,   default=40)
    p.add_argument("--alpha_max",        type=float, default=1.2)
    p.add_argument("--alpha_min",        type=float, default=0.4)
    p.add_argument("--mutation_prob",    type=float, default=0.15)
    p.add_argument("--stagnation_limit", type=int,   default=8)
    p.add_argument("--reinit_fraction",  type=float, default=0.20)
    p.add_argument("--ae_weighting",     action="store_true", default=True)
    p.add_argument("--no_ae_weighting",  action="store_false", dest="ae_weighting")
    p.add_argument("--pair_interval",    type=int,   default=5)
    p.add_argument("--rotate_factor",    type=float, default=0.01)
    p.add_argument(
        "--backend", type=str, default="cudaq_tensornet",
        choices=["cudaq_tensornet", "cudaq_nvidia", "cudaq_qpp",
                 "cudaq_nvidia_fp64", "cudaq_tensornet_mps"],
    )
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--task_name",        type=str,   default="unconditional_9_ae_mpi")
    p.add_argument("--data_dir",         type=str,   default="results_mpi")
    return p.parse_args()


def setup_logger(log_path: str) -> logging.Logger:
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
        m = re.search(r'(\d+\.\d+\.\d+)', cudaq.__version__)
        logger.info(f"  CUDA-Q: {m.group(1) if m else cudaq.__version__}")
    except Exception:
        pass
    logger.info(f"  CUDA_VISIBLE_DEVICES（原始）: {_ORIGINAL_CUDA_DEVICES}")
    logger.info(f"  MPI N_RANK: {_NRANK}  rank {_RANK} → GPU {_my_gpu}")


# ===========================================================================
# ★ v1.1：統一 MPI 評估函式（所有 rank 共同呼叫）
# ===========================================================================

def _mpi_signal_stop() -> None:
    """
    向所有 rank 廣播停止信號。
    只有 rank 0 發起，其他 rank 在 _mpi_evaluate_all 的 flag bcast 中接收。
    """
    if not _HAS_MPI or _NRANK <= 1:
        return
    _COMM.bcast(_MPI_FLAG_STOP, root=0)


def _mpi_evaluate_all(
    gen:       "MoleculeGeneratorCUDAQ",
    cwg:       "ConditionalWeightsGenerator",
    args:      argparse.Namespace,
    positions: np.ndarray,   # (M, D)，只有 rank 0 傳入有效值，其他 rank 傳 None
) -> list:
    """
    ★ v1.1 修正：所有 MPI rank 同時呼叫此函式。

    通訊協定（v1.1）：
      1. Rank 0 bcast 整數 flag（_MPI_FLAG_CONTINUE 或 _MPI_FLAG_STOP）
      2. 所有 rank 接收 flag，若為 STOP 立即返回空列表
      3. Rank 0 bcast positions 矩陣（M×D）
      4. 各 rank 依 round-robin 評估 positions[rank::nrank]
      5. allgather 收集所有結果，每個 rank 得到完整的 ordered 列表
      6. 只有 rank 0 的回傳值被 optimizer 使用

    相比 v1.0 的改進：
      - 非 rank 0 直接呼叫此函式，不再手動複製評估邏輯
      - flag 機制比 None bcast 更明確，避免 Python bcast 的 None 歧義
      - rank 0 的 optimize() 外包 try/finally，確保 STOP 信號必然發送
    """
    if not _HAS_MPI or _NRANK <= 1:
        # 單 rank 模式：直接評估所有粒子
        M = positions.shape[0]
        results = []
        for idx in range(M):
            w   = positions[idx]
            w_c = cwg.apply_chemistry_constraint(w.copy())
            gen.update_weight_vector(w_c)
            try:
                _, v, u = gen.sample_molecule(args.num_sample)
            except Exception as e:
                print(f"[Rank 0] sample_molecule 失敗（idx={idx}）：{e}", flush=True)
                v, u = 0.0, 0.0
            results.append((float(v), float(u)))
        return results

    # ── Step 1：rank 0 廣播繼續信號 ─────────────────────────────────────
    flag = _COMM.bcast(_MPI_FLAG_CONTINUE if _RANK == 0 else None, root=0)
    if flag == _MPI_FLAG_STOP:
        return []

    # ── Step 2：rank 0 廣播位置矩陣 ─────────────────────────────────────
    positions = _COMM.bcast(positions if _RANK == 0 else None, root=0)
    M = positions.shape[0]

    # ── Step 3：round-robin 分配，各 rank 評估自己負責的粒子 ─────────────
    my_indices = list(range(_RANK, M, _NRANK))
    my_results = []

    for idx in my_indices:
        w   = positions[idx]
        w_c = cwg.apply_chemistry_constraint(w.copy())
        gen.update_weight_vector(w_c)
        try:
            _, v, u = gen.sample_molecule(args.num_sample)
        except Exception as e:
            print(f"[Rank {_RANK}] sample_molecule 失敗（idx={idx}）：{e}", flush=True)
            v, u = 0.0, 0.0
        my_results.append((float(v), float(u), idx))

    # ── Step 4：allgather 回收全部結果 ──────────────────────────────────
    all_scattered = _COMM.allgather(my_results)

    # ── Step 5：重組回原始粒子順序（僅 rank 0 實際使用）────────────────
    ordered: list = [(0.0, 0.0)] * M
    for rank_results in all_scattered:
        for v, u, idx in rank_results:
            ordered[idx] = (v, u)

    return ordered


# ===========================================================================
# batch_evaluate_fn 工廠（傳給 optimizer）
# ===========================================================================

def make_mpi_batch_evaluate_fn(
    gen:    "MoleculeGeneratorCUDAQ",
    cwg:    "ConditionalWeightsGenerator",
    args:   argparse.Namespace,
) -> callable:
    """
    建立 MPI 批次評估函式。

    ★ v1.1 重要說明：
      此函式只由 rank 0 的 optimizer 呼叫。
      非 rank 0 的 while 迴圈也呼叫 _mpi_evaluate_all（傳入 positions=None），
      由 flag bcast 同步兩端的進出。
    """
    def batch_evaluate_fn(positions: np.ndarray) -> list:
        # rank 0 傳入有效的 positions；_mpi_evaluate_all 會廣播給其他 rank
        return _mpi_evaluate_all(gen, cwg, args, positions)

    return batch_evaluate_fn


# ===========================================================================
# 主程式
# ===========================================================================

def main():
    args = parse_args()

    if _RANK == 0:
        os.makedirs(args.data_dir, exist_ok=True)
    if _HAS_MPI:
        _COMM.Barrier()

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
        logger.info(f"AE-SOQPSO v1.1  調和加權: {'開啟' if args.ae_weighting else '關閉'}")
        logger.info(f"AE 配對更新間隔: {args.pair_interval} 迭代")
        log_gpu_info(logger)
        logger.info(f"[MEM] 啟動時 RSS={get_rss_mb():.0f} MB (rank 0)")

    # ── 所有 rank 初始化 ConditionalWeightsGenerator ──────────────────────
    cwg = ConditionalWeightsGenerator(
        args.num_heavy_atom,
        smarts=None,
        disable_connectivity_position=[],
    )
    D = cwg.length_all_weight_vector   # 134

    if _RANK == 0:
        logger.info(f"Number of flexible parameters: {D}")

    # ── 所有 rank 各自初始化 MoleculeGeneratorCUDAQ ────────────────────────
    gen = MoleculeGeneratorCUDAQ(
        num_heavy_atom            = args.num_heavy_atom,
        all_weight_vector         = np.zeros(D),
        backend_name              = args.backend,
        remove_bond_disconnection = True,
        chemistry_constraint      = True,
    )

    if _RANK == 0:
        logger.info(f"[Rank {_RANK}] Generator 初始化完成（backend={args.backend}）")

    # ── ★ v1.1：功能驗證（所有 rank 同時呼叫 _mpi_evaluate_all）──────────
    if _RANK == 0:
        logger.info("[MPI v1.1] 執行功能驗證（5 shots × N_RANK）...")

    # 建立用於測試的臨時 args（只改 num_sample）
    class _TestArgs:
        num_sample      = 5
        num_heavy_atom  = args.num_heavy_atom

    w_test          = cwg.generate_conditional_random_weights(random_seed=99)
    test_positions  = np.tile(w_test, (_NRANK, 1))   # N_RANK 個相同粒子
    test_start      = time.time()

    # 所有 rank 同時呼叫（v1.1 設計）
    test_results = _mpi_evaluate_all(gen, cwg, _TestArgs(), test_positions)

    test_elapsed = time.time() - test_start

    if _RANK == 0:
        any_nonzero = any(v > 0 or u > 0 for v, u in test_results)
        logger.info(
            f"[MPI v1.1] 功能驗證完成（{test_elapsed:.1f}s）  "
            f"{'✓ 有效結果' if any_nonzero else '⚠ 全部 V=U=0，請確認 backend'}"
        )
        logger.info(f"[MEM] 功能驗證後 RSS={get_rss_mb():.0f} MB (rank 0)")

    if _HAS_MPI:
        _COMM.Barrier()

    # ── 建立 batch 評估函式 ────────────────────────────────────────────────
    batch_eval_fn = make_mpi_batch_evaluate_fn(gen, cwg, args)

    # ── 執行優化 ──────────────────────────────────────────────────────────
    # ★ v1.1 設計原則：
    #   rank 0：在 try/finally 中呼叫 optimizer.optimize()，
    #           finally 保證發送 STOP 信號。
    #
    #   非 rank 0：在 while 迴圈中持續呼叫 _mpi_evaluate_all(positions=None)，
    #              函式內的 flag bcast 決定繼續或退出。
    #              不再手動複製評估邏輯（v1.0 的維護陷阱）。
    #
    # 通訊對齊保證：
    #   每次 rank 0 的 batch_eval_fn → _mpi_evaluate_all 廣播兩次（flag + positions）
    #   非 rank 0 的 while 迴圈也在同一個 _mpi_evaluate_all 呼叫裡處理這兩次廣播
    #   => 兩端廣播計數永遠對齊

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

        best_params  = None
        best_fitness = -np.inf
        try:
            best_params, best_fitness = optimizer.optimize()
        except Exception as e:
            logger.error(f"[ERROR] optimizer.optimize() 發生例外：{e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # ★ v1.1 關鍵：無論 optimize() 是否成功，都必須發送 STOP 信號
            # 確保非 rank 0 能從 while 迴圈退出，不會永久阻塞
            if _HAS_MPI and _NRANK > 1:
                _mpi_signal_stop()
                logger.info("[MPI v1.1] STOP 信號已發送，等待所有 rank 完成...")

    else:
        # ★ v1.1：非 rank 0 直接在迴圈中呼叫 _mpi_evaluate_all
        # 函式內部的 flag bcast 決定繼續（CONTINUE）或退出（STOP）
        while True:
            results = _mpi_evaluate_all(gen, cwg, args, None)
            # results 為空列表代表收到 STOP 信號，退出迴圈
            if len(results) == 0:
                break

    # ── 所有 rank 同步，確保通訊完成 ──────────────────────────────────────
    if _HAS_MPI:
        _COMM.Barrier()

    # ── 最終輸出（只有 rank 0）────────────────────────────────────────────
    if _RANK == 0:
        if best_params is not None:
            best_npy = os.path.join(args.data_dir, f"{args.task_name}_best_params.npy")
            np.save(best_npy, best_params)
            logger.info(f"最佳參數已儲存: {best_npy}")
            logger.info(
                f"最終結果: V×U={best_fitness:.6f}  "
                + ("✓ 超越 BO 基線 0.8834!" if best_fitness > 0.8834
                   else "✗ 未超越 — 建議增加 --particles 或 --iterations")
            )
        else:
            logger.error("optimize() 未正常完成，無法儲存最佳參數。")
        logger.info(f"[MEM] 程序結束前 RSS={get_rss_mb():.0f} MB (rank 0)")

    if _HAS_MPI:
        MPI.Finalize()


if __name__ == "__main__":
    main()