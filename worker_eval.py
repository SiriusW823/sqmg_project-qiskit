"""
==============================================================================
worker_eval.py — 子行程評估工作者（v10.1 修正版）
==============================================================================

v10.0 → v10.1 修正：

  ★ [BUG-FIX] 雙重 chemistry constraint 問題：
      v10.0 中，run_qpso_qmg_cudaq.py 的 evaluate_fn 在儲存 weight 前
      已呼叫過 cwg.apply_chemistry_constraint(pos.copy())，
      但 worker_eval.py 載入後又再呼叫一次 apply_chemistry_constraint。
      
      apply_chemistry_constraint 不是冪等函式（non-idempotent）：
        第一次：對 softmax 前的原始隨機值做 softmax_temperature
        第二次：對已 softmax 過的值再做一次 softmax_temperature
        結果：機率分佈進一步向最大值集中，破壞 QPSO 的參數空間探索能力
        
      修正：worker 直接使用主行程已處理過的 constrained weights，
      不再重複套用 chemistry constraint。

說明：
  此檔案供 run_qpso_qmg_cudaq.py（v9.6/v10.x subprocess 版）使用。
  若使用 run_qpso_qmg_mpi.py（MPI 版），不需要此檔案。

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
    p = argparse.ArgumentParser(description="QMG worker_eval (v10.1)")
    p.add_argument("--weight_path",    type=str, required=True)
    p.add_argument("--result_path",    type=str, required=True)
    p.add_argument("--num_heavy_atom", type=int, default=9)
    p.add_argument("--num_sample",     type=int, default=10000)
    p.add_argument("--backend",        type=str, default="cudaq_tensornet",
                   choices=SUPPORTED_BACKENDS)
    args = p.parse_args()

    # 預設失敗輸出（確保 result_path 在任何錯誤情況下都存在）
    np.save(args.result_path, np.array([0.0, 0.0], dtype=np.float64))

    try:
        from qmg.generator_cudaq import MoleculeGeneratorCUDAQ

        # ★ v10.1 修正：
        #   主行程（run_qpso_qmg_cudaq.py）在儲存前已呼叫：
        #     w_constrained = cwg.apply_chemistry_constraint(pos.copy())
        #     np.save(weight_path, w_constrained)
        #   因此此處直接載入，不再重複套用 chemistry constraint。
        #
        #   舊版（v10.0）錯誤流程：
        #     w = np.load(...)                               # 已 constrained
        #     cwg = ConditionalWeightsGenerator(...)
        #     w_constrained = cwg.apply_chemistry_constraint(w.copy())  # 第二次！
        #     gen = MoleculeGeneratorCUDAQ(..., chemistry_constraint=True)
        #
        #   新版（v10.1）正確流程：
        #     w = np.load(...)                               # 已 constrained
        #     gen = MoleculeGeneratorCUDAQ(..., chemistry_constraint=False)
        w = np.load(args.weight_path)
        assert len(w) == 134, f"weight 長度錯誤：{len(w)}，期待 134"

        gen = MoleculeGeneratorCUDAQ(
            num_heavy_atom            = args.num_heavy_atom,
            all_weight_vector         = w,
            backend_name              = args.backend,
            remove_bond_disconnection = True,
            # ★ v10.1 關鍵修正：設為 False，避免 generator 內部再套用一次
            #   主行程的 evaluate_fn 已套用，此處不需要重複
            chemistry_constraint      = False,
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