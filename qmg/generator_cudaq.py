"""
==============================================================================
generator_cudaq.py  (CUDA-Q 0.7.1 / V100 sm_70 完整修正版 v2)
==============================================================================

環境確認（cudaq-v071 conda 環境）：
  cudaq       : 0.7.1
  Python      : 3.10.20
  numpy       : 1.26.4
  rdkit       : 2026.03.1
  custatevec  : 1.5.0 (pip custatevec-cu11)

修正清單：
  [FIX-1] cudaq.__version__ 字串解析：
          0.7.1 回傳 "CUDA-Q Version 0.7.1 (https://...)"，
          直接 split('.') + int() 會崩潰。改用 re.search 提取純版號。

  [FIX-2] _smoke_kernel 移至模組層級：
          @cudaq.kernel 裝飾器在 JIT 時用 inspect.getsource() 讀取原始碼，
          巢狀函式在某些 Python frame 下會失敗。模組層級定義最穩定。

  [FIX-3] ★ 最關鍵修正：cudaq.sample() 加入 explicit_measurements=True
          cudaq 0.7.1 SampleResult 行為（依 NVIDIA 官方文件）：
            - 預設（explicit_measurements=False）：
              __global__ 含所有 mz() qubit 在電路 END 的重測結果（最終狀態）
              對 20-qubit N=9 kernel：result.items() 回傳 20-bit 字串
              → decode 時 bitstring 長度不符 90，導致 SMILES 解碼完全錯誤
            - explicit_measurements=True：
              __global__ 含所有顯式 mz() 呼叫依序拼接
              對 N=9 kernel（90 個 mz()）：result.items() 回傳 90-bit 字串
              → decode 正確 ✓

  [FIX-4] _reconstruct_bitstrings_n9 簡化：
          加入 explicit_measurements=True 後，result.items() 直接回傳
          90-bit 字串，不再需要逐暫存器重建的複雜邏輯。
          保留版本感知框架，同時簡化 0.7.x 路徑。

  [FIX-5] sample_molecule 空 raw_counts 防呆。
==============================================================================
"""
from __future__ import annotations

import re
import warnings
import numpy as np
from typing import List, Union, Tuple

import cudaq

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from qmg.utils.chemistry_data_processing import MoleculeQuantumStateGenerator
from qmg.utils.weight_generator import ConditionalWeightsGenerator
from qmg.utils.build_dynamic_circuit_cudaq import DynamicCircuitBuilderCUDAQ


# ===========================================================================
# [FIX-2] 模組層級 smoke test kernel
# 必須在模組層級定義，避免 inspect.getsource() 在巢狀函式中失敗
# ===========================================================================

@cudaq.kernel
def _smoke_kernel():
    q = cudaq.qvector(1)
    h(q[0])
    mz(q[0])


# ===========================================================================
# CUDA-Q 版本與 V100 架構相容性
# ===========================================================================

def _check_cudaq_version_volta_compat() -> tuple[str, bool]:
    """
    解析 cudaq.__version__ 並判斷是否支援 V100 (sm_70)。

    cudaq 0.7.1 版本字串格式：
        "CUDA-Q Version 0.7.1 (https://github.com/NVIDIA/cuda-quantum ...)"
    直接 split('.') + int() 在 parts[0] = "CUDA-Q Version 0" 時會拋 ValueError。
    改用 re.search 提取純數字版號。

    回傳 (version_string, is_volta_compatible)
    is_volta_compatible=True 代表版本 <= 0.7.x，支援 sm_70。
    """
    try:
        ver_str = cudaq.__version__
        match = re.search(r'(\d+)\.(\d+)\.(\d+)', ver_str)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            return ver_str, (major, minor) <= (0, 7)
        # 無法解析版號 → 保守假設為相容
        return ver_str, True
    except Exception:
        return "unknown", True


def _verify_gpu_actually_used(target_name: str) -> bool:
    """
    用模組層級 _smoke_kernel 驗證 GPU target 確實有效。
    """
    if target_name not in ("nvidia", "nvidia-fp64", "tensornet"):
        return False
    try:
        result = cudaq.sample(_smoke_kernel, shots_count=16,
                              explicit_measurements=True)
        return len(dict(result.items())) > 0
    except Exception as e:
        warnings.warn(f"[CUDAQ] GPU smoke test 失敗：{e}")
        return False


# ===========================================================================
# Backend 映射
# ===========================================================================

_CUDAQ_TARGET_MAP = {
    "cudaq_qpp":         "qpp-cpu",
    "qpp-cpu":           "qpp-cpu",
    "cudaq_nvidia":      "nvidia",
    "nvidia":            "nvidia",
    "cudaq_nvidia_fp64": "nvidia-fp64",
    "nvidia-fp64":       "nvidia-fp64",
    "tensornet":         "tensornet",
    "qiskit_aer":        "qpp-cpu",
}
_GPU_TARGETS = {"nvidia", "nvidia-fp64", "tensornet"}


def _set_target_safe(target_name: str) -> str:
    ver_str, is_compat = _check_cudaq_version_volta_compat()
    if target_name in _GPU_TARGETS and not is_compat:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"[CUDAQ] CUDA-Q {ver_str} 不支援 V100 (sm_70)。\n"
            f"  請安裝：pip install cuda-quantum==0.7.1\n"
            f"{'='*60}"
        )
    try:
        cudaq.set_target(target_name)
    except Exception as e:
        raise RuntimeError(f"[CUDAQ] set_target('{target_name}') 失敗：{e}") from e

    if target_name in _GPU_TARGETS:
        if _verify_gpu_actually_used(target_name):
            print(f"[CUDAQ] GPU target '{target_name}' 驗證通過 ✓")
        else:
            warnings.warn(f"[CUDAQ] GPU smoke test 異常，可能在 CPU 執行。")
    return target_name


# ===========================================================================
# [FIX-3+4] bitstring 重建
# ===========================================================================

def _reconstruct_bitstrings_n9(result) -> dict[str, int]:
    """
    從 SampleResult 重建 {90-bit-bitstring: count} dict。

    前提：cudaq.sample() 已使用 explicit_measurements=True。
    在此模式下，result.items() 回傳 __global__ 暫存器的內容，
    即所有顯式 mz() 依序拼接的 90-bit 字串，可直接使用。

    容錯：若 items() 的字串長度不是 90（例如未來版本行為變化），
    則嘗試 get_sequential_data() 逐暫存器重建。
    """
    # ── 主路徑：explicit_measurements=True 後 items() 應直接是 90 bits ──────
    try:
        sample_items = list(result.items())
        if sample_items:
            first_bs = sample_items[0][0]
            if len(first_bs) == 90:
                return {bs: cnt for bs, cnt in sample_items}
            # 長度不符，記錄警告並繼續嘗試備用路徑
            warnings.warn(
                f"[CUDAQ] result.items() bitstring 長度為 {len(first_bs)}，"
                f"預期 90。嘗試 get_sequential_data() 備用路徑。"
            )
    except Exception as e:
        warnings.warn(f"[CUDAQ] result.items() 失敗：{e}，嘗試備用路徑。")

    # ── 備用路徑：逐暫存器重建（0.9.x+ 無 explicit_measurements 時的行為）──
    _N9_NAMED_REG_ORDER = [
        'a1_0', 'a1_1', 'a2_0', 'a2_1',
        'b21_0', 'b21_1',
        'a3_0', 'a3_1',
        'b31_0', 'b31_1', 'b32_0', 'b32_1',
        'a4_0', 'a4_1',
        'b41_0', 'b41_1', 'b42_0', 'b42_1', 'b43_0', 'b43_1',
        'a5_0', 'a5_1',
        'b51_0', 'b51_1', 'b52_0', 'b52_1', 'b53_0', 'b53_1', 'b54_0', 'b54_1',
        'a6_0', 'a6_1',
        'b61_0', 'b61_1', 'b62_0', 'b62_1', 'b63_0', 'b63_1',
        'b64_0', 'b64_1', 'b65_0', 'b65_1',
        'a7_0', 'a7_1',
        'b71_0', 'b71_1', 'b72_0', 'b72_1', 'b73_0', 'b73_1',
        'b74_0', 'b74_1', 'b75_0', 'b75_1', 'b76_0', 'b76_1',
        'a8_0', 'a8_1',
        'b81_0', 'b81_1', 'b82_0', 'b82_1', 'b83_0', 'b83_1',
        'b84_0', 'b84_1', 'b85_0', 'b85_1', 'b86_0', 'b86_1', 'b87_0', 'b87_1',
        'a9_0', 'a9_1',
        '__global__',
    ]
    available_regs = set(getattr(result, 'register_names', []))
    missing = [r for r in _N9_NAMED_REG_ORDER if r not in available_regs]
    if missing:
        raise RuntimeError(
            f"[CUDAQ] 無法重建 bitstring。\n"
            f"  items() 長度不符，且缺少暫存器：{missing[:5]}\n"
            f"  請確認 cudaq.sample() 使用了 explicit_measurements=True"
        )
    reg_data = {reg: result.get_sequential_data(reg)
                for reg in _N9_NAMED_REG_ORDER}
    n_shots = len(reg_data[_N9_NAMED_REG_ORDER[0]])
    counts: dict[str, int] = {}
    for i in range(n_shots):
        bs = ''.join(reg_data[reg][i] for reg in _N9_NAMED_REG_ORDER)
        if len(bs) != 90:
            raise RuntimeError(
                f"Shot {i}: 重建 bitstring 長度 {len(bs)} != 90"
            )
        counts[bs] = counts.get(bs, 0) + 1
    return counts


# ===========================================================================
# MoleculeGeneratorCUDAQ
# ===========================================================================

class MoleculeGeneratorCUDAQ:
    """
    CUDA-Q 版分子生成器（CUDA-Q 0.7.1 / V100 sm_70 相容）。

    公開介面：
        update_weight_vector(w)
        sample_molecule(num_sample) → (smiles_dict, validity, uniqueness)
    """

    def __init__(
        self,
        num_heavy_atom:            int,
        all_weight_vector:         Union[List[float], np.ndarray, None] = None,
        backend_name:              str   = "cudaq_nvidia",
        temperature:               float = 0.2,
        dynamic_circuit:           bool  = True,
        remove_bond_disconnection: bool  = True,
        chemistry_constraint:      bool  = True,
    ):
        if not dynamic_circuit:
            raise NotImplementedError("CUDA-Q 版目前僅支援 dynamic_circuit=True。")
        if num_heavy_atom != 9:
            raise NotImplementedError(
                f"目前僅支援 num_heavy_atom=9（N={num_heavy_atom} 尚未實作）。"
            )

        self.num_heavy_atom            = num_heavy_atom
        self.all_weight_vector         = (
            np.array(all_weight_vector, dtype=np.float64)
            if all_weight_vector is not None else None
        )
        self.backend_name              = backend_name
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint      = chemistry_constraint
        self.expected_bits             = num_heavy_atom * (num_heavy_atom + 1)  # 90

        self.circuit_builder = DynamicCircuitBuilderCUDAQ(
            num_heavy_atom            = num_heavy_atom,
            temperature               = temperature,
            remove_bond_disconnection = remove_bond_disconnection,
            chemistry_constraint      = chemistry_constraint,
        )
        self.kernel = self.circuit_builder.get_kernel()

        self.data_generator = MoleculeQuantumStateGenerator(
            heavy_atom_size = num_heavy_atom,
            ncpus           = 1,
            sanitize_method = "strict",
        )

        actual_target       = _CUDAQ_TARGET_MAP.get(backend_name, "qpp-cpu")
        self._active_target = _set_target_safe(actual_target)

        ver_str, _ = _check_cudaq_version_volta_compat()
        print(
            f"[CUDAQ] Generator initialized.\n"
            f"  cudaq version : {ver_str}\n"
            f"  active target : {self._active_target}\n"
            f"  N atoms       : {num_heavy_atom}\n"
            f"  weight dim    : {self.circuit_builder.length_all_weight_vector}\n"
            f"  expected_bits : {self.expected_bits}"
        )

    def update_weight_vector(
        self, all_weight_vector: Union[List[float], np.ndarray]
    ) -> None:
        self.all_weight_vector = np.array(all_weight_vector, dtype=np.float64)

    def sample_molecule(
        self,
        num_sample:  int,
        random_seed: int = 0,
    ) -> Tuple[dict, float, float]:
        assert self.all_weight_vector is not None, "請先呼叫 update_weight_vector()。"

        w = self.all_weight_vector
        assert len(w) == self.circuit_builder.length_all_weight_vector, (
            f"weight 長度不符：{len(w)} != "
            f"{self.circuit_builder.length_all_weight_vector}"
        )

        try:
            cudaq.set_random_seed(random_seed)
        except AttributeError:
            pass

        # [FIX-3] ★ 必須加 explicit_measurements=True
        # 否則 result.items() 回傳 20-bit（最終 qubit 狀態），非 90-bit（mz 序列）
        # 根據 NVIDIA 官方 0.7.1 文件：
        #   explicit_measurements=True → __global__ = 所有 mz() 依序拼接 → 90 bits ✓
        #   explicit_measurements=False（預設） → __global__ = 電路末端重測值 → 20 bits ✗
        result = cudaq.sample(
            self.kernel,
            w.tolist(),
            shots_count=num_sample,
            explicit_measurements=True,   # ← 關鍵
        )

        # [FIX-4] 重建 90-bit bitstring dict
        raw_counts = _reconstruct_bitstrings_n9(result)

        # [FIX-5] raw_counts 空值防呆
        if not raw_counts:
            warnings.warn("[CUDAQ] sample_molecule: raw_counts 為空，回傳 validity=0。")
            return {}, 0.0, 0.0

        # 解碼 bitstring → SMILES
        smiles_dict:    dict[str, int] = {}
        num_valid_shots = 0

        for bs, count in raw_counts.items():
            bs_fixed      = self.circuit_builder.apply_bond_disconnection_correction(bs)
            quantum_state = self.data_generator.post_process_quantum_state(
                bs_fixed, reverse=False
            )
            smiles = self.data_generator.QuantumStateToSmiles(quantum_state)
            smiles_dict[smiles] = smiles_dict.get(smiles, 0) + count
            if smiles and smiles != "None":
                num_valid_shots += count

        validity = num_valid_shots / num_sample

        num_unique_valid = len([k for k in smiles_dict if k and k != "None"])
        uniqueness = (
            num_unique_valid / num_valid_shots if num_valid_shots > 0 else 0.0
        )

        return smiles_dict, validity, uniqueness


MoleculeGenerator = MoleculeGeneratorCUDAQ


# ===========================================================================
# 快速功能驗證
# ===========================================================================
if __name__ == "__main__":
    import time

    print("=== MoleculeGeneratorCUDAQ 功能驗證 ===")
    ver_str, is_compat = _check_cudaq_version_volta_compat()
    print(f"CUDA-Q : {ver_str}  Volta compat: {'✓' if is_compat else '⚠ >=0.8'}")

    cwg = ConditionalWeightsGenerator(9, smarts=None)
    w   = cwg.generate_conditional_random_weights(random_seed=42)

    print("\n[Test CPU] qpp-cpu, 200 shots")
    gen = MoleculeGeneratorCUDAQ(9, all_weight_vector=w, backend_name="cudaq_qpp")
    t0  = time.time()
    smiles_dict, v, u = gen.sample_molecule(200)
    print(f"  V={v:.3f}  U={u:.3f}  V×U={v*u:.4f}  ({time.time()-t0:.1f}s)")
    valid_smiles = [k for k in smiles_dict if k and k != "None"]
    print(f"  有效分子數: {len(valid_smiles)}  範例: {valid_smiles[:3]}")

    print("\n[Test GPU] nvidia, 500 shots")
    try:
        gen_gpu = MoleculeGeneratorCUDAQ(9, all_weight_vector=w,
                                         backend_name="cudaq_nvidia")
        t0 = time.time()
        smiles_dict, v, u = gen_gpu.sample_molecule(500)
        print(f"  V={v:.3f}  U={u:.3f}  V×U={v*u:.4f}  ({time.time()-t0:.1f}s)  ✓")
        valid_smiles = [k for k in smiles_dict if k and k != "None"]
        print(f"  有效分子數: {len(valid_smiles)}  範例: {valid_smiles[:3]}")
    except Exception as e:
        print(f"  GPU 失敗：{e}")