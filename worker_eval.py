"""
==============================================================================
worker_eval.py — 子行程評估工作者（v10.0 更新）
==============================================================================

v9.6 → v10.0 更新：
  - 新增 tensornet / tensornet-mps / nvidia-mqpu / nvidia-mgpu 後端支援
  - 改善錯誤訊息輸出（exit code 1 時印出完整 traceback）

說明：
  此檔案供 run_qpso_qmg_cudaq.py（v9.6 subprocess 版）使用。
  若使用 run_qpso_qmg_mpi.py（MPI 版），不需要此檔案。

  MPI 版優勢：
    - 不需要 subprocess，零 process 啟動開銷
    - CUDA context 隔離由 MPI rank 保證
    - tensornet 後端速度優勢完全發揮

放置位置：worker_eval.py（專案根目錄）
==============================================================================
"""
import argparse
import sys
import os

import numpy as np

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass


SUPPORTED_BACKENDS = [
    "cudaq_nvidia",
    "cudaq_qpp",
    "cudaq_nvidia_fp64",
    "cudaq_tensornet",        # v10.0 新增：SQMG 推薦，最快
    "cudaq_tensornet_mps",    # v10.0 新增：MPS TN
    "cudaq_nvidia_mqpu",      # v10.0 新增：multi-GPU shots
    "cudaq_nvidia_mgpu",      # v10.0 新增：multi-GPU statevec
]


def main():
    p = argparse.ArgumentParser(description="QMG worker_eval (v10.0)")
    p.add_argument("--weight_path",    type=str, required=True)
    p.add_argument("--result_path",    type=str, required=True)
    p.add_argument("--num_heavy_atom", type=int, default=9)
    p.add_argument("--num_sample",     type=int, default=10000)
    p.add_argument("--backend",        type=str, default="cudaq_tensornet",
                   choices=SUPPORTED_BACKENDS)
    args = p.parse_args()

    # 預設失敗輸出
    np.save(args.result_path, np.array([0.0, 0.0], dtype=np.float64))

    try:
        from qmg.generator_cudaq import MoleculeGeneratorCUDAQ
        from qmg.utils.weight_generator import ConditionalWeightsGenerator

        w = np.load(args.weight_path)
        assert len(w) == 134, f"weight 長度錯誤：{len(w)}，期待 134"

        cwg = ConditionalWeightsGenerator(args.num_heavy_atom, smarts=None)
        w_constrained = cwg.apply_chemistry_constraint(w.copy())

        gen = MoleculeGeneratorCUDAQ(
            num_heavy_atom            = args.num_heavy_atom,
            all_weight_vector         = w_constrained,
            backend_name              = args.backend,
            remove_bond_disconnection = True,
            chemistry_constraint      = True,
        )

        _, validity, uniqueness = gen.sample_molecule(args.num_sample)

        np.save(args.result_path, np.array([validity, uniqueness], dtype=np.float64))
        sys.exit(0)

    except Exception as e:
        print(f"[worker_eval] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()