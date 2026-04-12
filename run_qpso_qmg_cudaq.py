"""
==============================================================================
run_qpso_qmg_cudaq.py  ─ CUDA-Q 0.7.1 + SOQPSO 主入口（完整修正版）
==============================================================================
修正清單：
  [FIX-1] import 路徑改為 qmg.utils（配合 qmg/utils/ 套件目錄）
  [FIX-2] evaluate_fn 加入維度 assert，防止 QPSO 傳入錯誤 shape
  [FIX-3] data_dir 使用 os.makedirs(exist_ok=True) 確保目錄存在後才開 log
  [FIX-4] smarts=None 時 disable_connectivity_position 傳 [] 而非 None
  [FIX-5] --data_dir 預設值更名，避免與舊結果混淆
  [FIX-6] log_gpu_info 新增 CUDA-Q 版本字串的 regex 解析保護，
          避免 cudaq 0.7.1 回傳完整版本字串時印出異常
==============================================================================
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time

import numpy as np

# ── RDKit 靜音 ────────────────────────────────────────────────────────────────
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

# ── CUDA-Q ───────────────────────────────────────────────────────────────────
try:
    import cudaq
except ImportError:
    print("[ERROR] 無法 import cudaq。請執行：")
    print("  pip install cuda-quantum==0.7.1")
    sys.exit(1)

# ── QMG 套件 ─────────────────────────────────────────────────────────────────
try:
    from qmg.utils import ConditionalWeightsGenerator
except ImportError as e:
    print(f"[ERROR] 無法 import qmg.utils: {e}")
    print("  請確認 qmg/utils/__init__.py 存在且包含正確 import。")
    sys.exit(1)

try:
    from qmg.generator_cudaq import MoleculeGeneratorCUDAQ
except ImportError as e:
    print(f"[ERROR] 無法 import qmg.generator_cudaq: {e}")
    sys.exit(1)

# ── SOQPSO 優化器 ─────────────────────────────────────────────────────────────
try:
    from qpso_optimizer_qmg import QMGSOQPSOOptimizer
except ImportError as e:
    print(f"[ERROR] 無法 import qpso_optimizer_qmg: {e}")
    sys.exit(1)


# ===========================================================================
# 命令列參數
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QMG CUDA-Q 0.7.1 + SOQPSO 分子生成優化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 電路參數
    p.add_argument("--num_heavy_atom",   type=int,   default=9,
                   help="重原子數（目前支援 N=9）")
    p.add_argument("--num_sample",       type=int,   default=10000,
                   help="每次電路評估的 shots 數")
    # QPSO 參數
    p.add_argument("--particles",        type=int,   default=50,
                   help="QPSO 粒子數 M")
    p.add_argument("--iterations",       type=int,   default=200,
                   help="QPSO 迭代數 T（總 evals = M×(T+1)）")
    p.add_argument("--alpha_max",        type=float, default=1.5,
                   help="收斂係數上界")
    p.add_argument("--alpha_min",        type=float, default=0.5,
                   help="收斂係數下界")
    p.add_argument("--mutation_prob",    type=float, default=0.12,
                   help="Cauchy 變異機率")
    p.add_argument("--stagnation_limit", type=int,   default=8,
                   help="停滯偵測門檻（QPSO 迭代次數）")
    p.add_argument("--seed",             type=int,   default=42,
                   help="隨機種子")
    # Backend
    p.add_argument(
        "--backend", type=str, default="cudaq_nvidia",
        choices=["cudaq_qpp", "cudaq_nvidia", "cudaq_nvidia_fp64", "tensornet"],
        help="CUDA-Q 模擬後端（V100 單 GPU 推薦 cudaq_nvidia）",
    )
    # 輸出
    p.add_argument("--task_name", type=str,
                   default="unconditional_9_qpso",
                   help="實驗名稱（用於 log/csv/npy 檔名）")
    p.add_argument("--data_dir",  type=str,
                   default="results_dgx1_gpu_final",
                   help="結果輸出目錄")
    return p.parse_args()


# ===========================================================================
# Logger（格式對齊 unconditional_9.log）
# ===========================================================================

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("CUDAQQPSOLogger")
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


# ===========================================================================
# GPU 診斷資訊
# ===========================================================================

def log_gpu_info(logger: logging.Logger) -> None:
    try:
        import subprocess
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=index,name,memory.total,driver_version "
            "--format=csv,noheader",
            shell=True, stderr=subprocess.DEVNULL,
        ).decode().strip()
        for line in out.splitlines():
            logger.info(f"  GPU: {line.strip()}")
    except Exception:
        logger.info("  GPU info: 無法取得（nvidia-smi 不可用）")

    # [FIX-6] 0.7.1 的 __version__ 是完整字串，用 regex 提取純版號
    try:
        ver_str = cudaq.__version__
        match = re.search(r'(\d+\.\d+\.\d+)', ver_str)
        short_ver = match.group(1) if match else ver_str
        logger.info(f"  CUDA-Q version: {short_ver} (full: {ver_str[:60]})")
    except AttributeError:
        logger.info("  CUDA-Q version: unknown")

    try:
        avail = [str(t) for t in cudaq.get_targets()]
        logger.info(f"  Available targets: {avail}")
    except Exception:
        pass


# ===========================================================================
# 主流程
# ===========================================================================

def main() -> None:
    args = parse_args()

    # [FIX-3] 先建立目錄，才能開 log file
    os.makedirs(args.data_dir, exist_ok=True)

    log_path = os.path.join(args.data_dir, f"{args.task_name}.log")
    logger   = setup_logger(log_path)

    # ── 啟動資訊（格式對齊 unconditional_9.log）──────────────────────────────
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Task: ['validity', 'uniqueness']")
    logger.info(f"Condition: ['None', 'None']")
    logger.info(f"objective: ['maximize', 'maximize']")
    logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
    logger.info(f"# of samples: {args.num_sample}")
    logger.info(f"smarts: None")
    logger.info(f"disable_connectivity_position: []")
    logger.info(f"CUDA-Q backend: {args.backend}")
    log_gpu_info(logger)

    # ── ConditionalWeightsGenerator（N=9, smarts=None → 全部 134 參數可自由優化）
    # [FIX-4] smarts=None 時傳 []（空 list），不傳 None
    cwg = ConditionalWeightsGenerator(
        args.num_heavy_atom,
        smarts=None,
        disable_connectivity_position=[],
    )
    n_flexible = int((cwg.parameters_indicator == 0.0).sum())
    logger.info(f"Number of flexible parameters: {n_flexible}")

    # 健全性檢查：smarts=None 時 n_flexible 應等於 length_all_weight_vector(134)
    assert n_flexible == cwg.length_all_weight_vector, (
        f"[BUG] n_flexible={n_flexible} != "
        f"length_all_weight_vector={cwg.length_all_weight_vector}"
    )

    total_evals = args.particles * (args.iterations + 1)
    logger.info(
        f"[CUDAQ-QPSO config] "
        f"M={args.particles}  T={args.iterations}  "
        f"total_evals={total_evals}  seed={args.seed}  "
        f"backend={args.backend}"
    )

    # ── CUDA-Q 生成器（一次性建立，避免重複 JIT 編譯）────────────────────────
    logger.info(
        "[CUDAQ] 初始化 MoleculeGeneratorCUDAQ"
        "（首次 JIT 編譯可能需 10~60s，請耐心等待）..."
    )
    t_init = time.time()
    generator = MoleculeGeneratorCUDAQ(
        num_heavy_atom            = args.num_heavy_atom,
        backend_name              = args.backend,
        remove_bond_disconnection = True,
        chemistry_constraint      = True,
    )
    logger.info(f"[CUDAQ] 初始化完成，耗時 {time.time() - t_init:.1f}s")

    # ── Evaluate function（QPSO → 量子電路評估）───────────────────────────────
    def evaluate_fn(pos: np.ndarray) -> tuple:
        """
        pos : shape=(n_flexible=134,)，值域 [0,1]
        apply_chemistry_constraint 套用 bond_type softmax 後
        作為完整 134-dim weight vector 送入電路。
        """
        # [FIX-2] 維度防呆
        if len(pos) != n_flexible:
            raise ValueError(
                f"[evaluate_fn] pos 維度錯誤：got {len(pos)}, expected {n_flexible}"
            )
        w = cwg.apply_chemistry_constraint(pos.copy())
        generator.update_weight_vector(w)
        _, validity, uniqueness = generator.sample_molecule(args.num_sample)
        return float(validity), float(uniqueness)

    # ── SOQPSO 優化（與 Qiskit 版完全共用，僅 evaluate_fn 不同）─────────────
    optimizer = QMGSOQPSOOptimizer(
        n_params          = n_flexible,
        n_particles       = args.particles,
        max_iterations    = args.iterations,
        evaluate_fn       = evaluate_fn,
        logger            = logger,
        seed              = args.seed,
        alpha_max         = args.alpha_max,
        alpha_min         = args.alpha_min,
        data_dir          = args.data_dir,
        task_name         = args.task_name,
        stagnation_limit  = args.stagnation_limit,
        mutation_prob     = args.mutation_prob,
    )

    best_params, best_fitness = optimizer.optimize()

    # ── 儲存最佳參數 ─────────────────────────────────────────────────────────
    best_npy = os.path.join(args.data_dir, f"{args.task_name}_best_params.npy")
    np.save(best_npy, best_params)
    logger.info(f"最佳參數已儲存: {best_npy}")
    logger.info(
        f"最終結果: V×U={best_fitness:.6f}  "
        + ("✓ 超越 BO 基線 0.8834!" if best_fitness > 0.8834
           else "✗ 未超越 — 建議增加 --particles 或 --iterations")
    )


if __name__ == "__main__":
    main()