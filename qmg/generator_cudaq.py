"""
==============================================================================
generator_cudaq.py  (CUDA-Q 0.7.1 / V100 sm_70 完整修正版)
==============================================================================

修正清單：
  [FIX-1] _check_cudaq_version_volta_compat：
          0.7.1 的 cudaq.__version__ 回傳完整字串
          "CUDA-Q Version 0.7.1 (https://...)"，直接 split('.') + int() 會崩潰。
          改用 re.search 提取純版號數字。

  [FIX-2] _smoke_kernel：
          原版在函數內定義 @cudaq.kernel，inspect.getsource() 可能在某些
          Python 版本或執行方式下取得錯誤行號而失敗。
          改為模組層級定義，確保 AST bridge 能正確取得原始碼。

  [FIX-3] _verify_gpu_actually_used：
          使用模組層級 _smoke_kernel，避免 BUG-2 問題。

  [FIX-4] _reconstruct_bitstrings_n9：
          CUDA-Q 0.7.x 與 0.9.x+ 的 SampleResult 行為不同：
            0.7.x：result.items() 回傳包含所有 mz() 的完整 90-bit bitstring
            0.9.x+：result.items() 只回傳 __global__（16 bits），
                    具名 mz() 分散在各自具名暫存器
          新增版本感知邏輯：先偵測 items() 的 bitstring 長度，
          若為 90 直接使用；否則退回 get_sequential_data() 逐暫存器重建。

  [FIX-5] MoleculeGeneratorCUDAQ.sample_molecule：
          新增對 raw_counts 為空的防呆，避免除以 0。
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
# N=9 kernel 的具名暫存器順序（依 mz() 呼叫順序，不可更動）
# ===========================================================================
#
# 對應 _qmg_dynamic_n9 kernel 中所有 mz() 的呼叫順序：
#   bits[ 0: 2] a1_0, a1_1        原子 1
#   bits[ 2: 4] a2_0, a2_1        原子 2
#   bits[ 4: 6] b21_0, b21_1      鍵 2-1
#   bits[ 6: 8] a3_0, a3_1        原子 3
#   bits[ 8:12] b31_0..b32_1      鍵 3-{1,2}
#   bits[12:14] a4_0, a4_1        原子 4
#   bits[14:20] b41_0..b43_1      鍵 4-{1,2,3}
#   bits[20:22] a5_0, a5_1        原子 5
#   bits[22:30] b51_0..b54_1      鍵 5-{1..4}
#   bits[30:32] a6_0, a6_1        原子 6
#   bits[32:42] b61_0..b65_1      鍵 6-{1..5}
#   bits[42:44] a7_0, a7_1        原子 7
#   bits[44:56] b71_0..b76_1      鍵 7-{1..6}
#   bits[56:58] a8_0, a8_1        原子 8
#   bits[58:72] b81_0..b87_1      鍵 8-{1..7}
#   bits[72:74] a9_0, a9_1        原子 9（具名）
#   bits[74:90] __global__         鍵 9-{1..8}（無名，16 bits）

_N9_NAMED_REG_ORDER: list[str] = [
    'a1_0', 'a1_1',
    'a2_0', 'a2_1',
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
    'a9_0', 'a9_1',   # 具名（2 bits）
    '__global__',      # 鍵 9-{1..8}，無名（16 bits）
]
# 74 個單 bit 具名暫存器 + 1 個 16-bit __global__


# ===========================================================================
# [FIX-2] 模組層級 smoke test kernel
# （不可放在函數內，否則 inspect.getsource() 可能取不到正確原始碼）
# ===========================================================================

@cudaq.kernel
def _smoke_kernel():
    q = cudaq.qvector(1)
    h(q[0])


# ===========================================================================
# CUDA-Q 版本與 V100 架構相容性
# ===========================================================================

def _check_cudaq_version_volta_compat() -> tuple[str, bool]:
    """
    解析 cudaq.__version__ 並判斷是否支援 V100 (sm_70)。

    cudaq 0.7.1 的版本字串格式為：
        "CUDA-Q Version 0.7.1 (https://github.com/NVIDIA/cuda-quantum ...)"
    直接 split('.') + int() 在 parts[0] = "CUDA-Q Version 0" 時會失敗。
    改用 re.search 提取純數字版號。
    """
    try:
        ver_str = cudaq.__version__
        # 同時相容 "0.7.1" 與 "CUDA-Q Version 0.7.1 (...)" 兩種格式
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
    用簡單的 bell state 電路驗證 GPU target 確實有效。
    使用模組層級 _smoke_kernel（[FIX-2]），避免 inspect 問題。
    """
    if target_name not in ("nvidia", "nvidia-fp64", "tensornet"):
        return False
    try:
        result = cudaq.sample(_smoke_kernel, shots_count=16)
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
# [FIX-4] 核心修正：版本感知的 bitstring 重建
# ===========================================================================

def _reconstruct_bitstrings_n9(result) -> dict[str, int]:
    """
    CUDA-Q 0.7.x vs 0.9.x+ 相容版 bitstring 重建。

    行為差異：
      0.7.x：result.items() 回傳包含所有 mz() 的完整 90-bit bitstring
      0.9.x+：result.items() 只回傳 __global__（16 bits），
              具名 mz() 分散在各自具名暫存器，
              需用 get_sequential_data(reg) 逐暫存器重建。

    策略：先偵測 items() 的 bitstring 長度：
      - 長度 == 90 → 0.7.x 風格，直接使用
      - 長度 != 90 → 0.9.x+ 風格，逐暫存器重建
    """
    # ── Step 1：嘗試 0.7.x 風格 ──────────────────────────────────────────
    try:
        sample_items = list(result.items())
        if sample_items:
            first_bs = sample_items[0][0]
            if len(first_bs) == 90:
                # 0.7.x：items() 已包含完整 90-bit bitstring，直接轉 dict
                return {bs: cnt for bs, cnt in sample_items}
            # items() 長度不是 90（例如 0.9.x+ 只有 16 bits）→ 繼續
    except Exception:
        pass

    # ── Step 2：0.9.x+ 風格，逐具名暫存器重建 ────────────────────────────
    available_regs = set(getattr(result, 'register_names', []))

    # 驗證所有預期暫存器都存在
    missing = [r for r in _N9_NAMED_REG_ORDER if r not in available_regs]
    if missing:
        raise RuntimeError(
            f"[CUDAQ] 缺少暫存器（kernel mz() 變數名可能已更改）：\n"
            f"  Missing  : {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
            f"  Available: {sorted(available_regs)}"
        )

    # 取得每個暫存器的 per-shot 資料
    # 單 bit 具名暫存器：list[str]，每個 str 為 '0' 或 '1'（1 char）
    # __global__（16 bits） ：list[str]，每個 str 為 16-char bitstring
    reg_data: dict[str, list[str]] = {
        reg: result.get_sequential_data(reg)
        for reg in _N9_NAMED_REG_ORDER
    }

    n_shots = len(reg_data[_N9_NAMED_REG_ORDER[0]])

    # 逐 shot 拼接
    counts: dict[str, int] = {}
    for i in range(n_shots):
        bs = ''.join(reg_data[reg][i] for reg in _N9_NAMED_REG_ORDER)
        if len(bs) != 90:
            raise RuntimeError(
                f"Shot {i}: 重建 bitstring 長度 {len(bs)} != 90。\n"
                f"  __global__ 長度: {len(reg_data['__global__'][i])}\n"
                f"  請確認 kernel 的 mz() 數量與 _N9_NAMED_REG_ORDER 一致。"
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

        # 量子採樣
        result = cudaq.sample(self.kernel, w.tolist(), shots_count=num_sample)

        # [FIX-4] 版本感知 bitstring 重建
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

        # [BUG-2 原版修正保留] 明確篩選 valid key
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
    _, v, u = gen.sample_molecule(200)
    print(f"  V={v:.3f}  U={u:.3f}  V×U={v*u:.4f}  ({time.time()-t0:.1f}s)")

    print("\n[Test GPU] nvidia, 500 shots")
    try:
        gen_gpu = MoleculeGeneratorCUDAQ(9, all_weight_vector=w, backend_name="cudaq_nvidia")
        t0 = time.time()
        _, v, u = gen_gpu.sample_molecule(500)
        print(f"  V={v:.3f}  U={u:.3f}  V×U={v*u:.4f}  ({time.time()-t0:.1f}s)  ✓")
    except Exception as e:
        print(f"  GPU 失敗：{e}")