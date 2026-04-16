"""
==============================================================================
generator_cudaq.py  (CUDA-Q 0.7.1 / V100 sm_70 完整修正版 v9.2)
==============================================================================

v9.1 → v9.2 根本修正：

  ★ CUDA-Q 0.7.1 cross-module list[float] broadcast bug（診斷確認）：
      cudaq.sample(imported_kernel, list_arg, shots_count=N) 的 broadcast 觸發機制：
      當 kernel 從其他模組 import 進來，CUDA-Q 0.7.1 對 list[float] 引數的
      型別識別與同檔案定義的 kernel 行為不同。
      
      傳入 w=[0.5]*134 時，CUDA-Q 0.7.1 將其視為「broadcast over 134 calls, 
      each with argument 0.5 (float)」，kernel 收到 float 而非 list[float]：
        RuntimeError: error: Invalid runtime argument type. 
        Argument of type <class 'float'> was provided, but list[float] was expected.
      
      已確認：diagnostic.py 中同檔案定義的 k_c1/k_c2/k_c3（list[float] 參數）
              可正常呼叫；但 import 的 _qmg_n9 失敗 → 根本原因是跨模組 dispatch。

      v9.2 解法：在本檔（generator_cudaq.py）直接定義 _qmg_n9_v9 kernel，
            確保 cudaq.sample 在「同檔案定義的 kernel」上呼叫 → broadcast 不觸發。
            MLIR 在 import 時只編譯一次，每次 sample 重用，解決 OOM。

速度對比（10000 shots，N=9）：
  qpp-cpu  : ~90s/eval
  nvidia   : ~1-5s/eval（V100 GPU + cuStateVec）
==============================================================================
"""
from __future__ import annotations

import math
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
# ★ v9.2 核心：_qmg_n9_v9 定義在本檔 module level
# 90 個 mz() 全部命名，不依賴 __global__（CUDA-Q 0.7.1 mid-circuit bug）
# ===========================================================================

@cudaq.kernel
def _qmg_n9_v9(w: list[float]):
    """N=9 QMG 動態電路，parametric kernel，w 長度 134，定義在本檔避免 broadcast bug。"""
    q = cudaq.qvector(20)

    # ── Phase 1: build_two_atoms  w[0:8] ──────────────────────────────────
    ry(math.pi * w[0], q[0])
    x(q[1])
    ry(math.pi * w[2], q[2])
    ry(math.pi * w[4], q[3])
    x.ctrl(q[0], q[1])
    ry.ctrl(math.pi * w[3], q[1], q[2])
    x.ctrl(q[2], q[3])
    ry.ctrl(math.pi * w[1], q[0], q[1])
    x.ctrl(q[1], q[2])
    ry.ctrl(math.pi * w[5], q[2], q[3])
    a1_0 = mz(q[0]); a1_1 = mz(q[1])
    a2_0 = mz(q[2]); a2_1 = mz(q[3])
    if a2_0 or a2_1:
        ry(math.pi * w[6], q[4]); x(q[5])
        x.ctrl(q[4], q[5])
        ry.ctrl(math.pi * w[7], q[4], q[5])
    b21_0 = mz(q[4]); b21_1 = mz(q[5])

    # ── Phase 2: atom 3  w[8:17] ──────────────────────────────────────────
    if a2_0: x(q[2])
    if a2_1: x(q[3])
    if b21_0: x(q[4])
    if b21_1: x(q[5])
    if a2_0 or a2_1:
        ry(math.pi * w[8], q[2]); ry(math.pi * w[9], q[3])
        ry.ctrl(math.pi * w[10], q[2], q[3])
    a3_0 = mz(q[2]); a3_1 = mz(q[3])
    if a3_0 or a3_1:
        ry(math.pi * w[11], q[5]); ry.ctrl(math.pi * w[13], q[5], q[4])
        ry.ctrl(math.pi * w[14], q[4], q[5])
        ry(math.pi * w[12], q[7]); ry.ctrl(math.pi * w[15], q[7], q[6])
        ry.ctrl(math.pi * w[16], q[6], q[7])
    b31_0 = mz(q[4]); b31_1 = mz(q[5])
    b32_0 = mz(q[6]); b32_1 = mz(q[7])

    # ── Phase 3: atom 4  w[17:29] ─────────────────────────────────────────
    if a3_0: x(q[2])
    if a3_1: x(q[3])
    if b31_0: x(q[4])
    if b31_1: x(q[5])
    if b32_0: x(q[6])
    if b32_1: x(q[7])
    if a3_0 or a3_1:
        ry(math.pi * w[17], q[2]); ry(math.pi * w[18], q[3])
        ry.ctrl(math.pi * w[19], q[2], q[3])
    a4_0 = mz(q[2]); a4_1 = mz(q[3])
    if a4_0 or a4_1:
        ry(math.pi * w[20], q[5]); ry.ctrl(math.pi * w[23], q[5], q[4])
        ry.ctrl(math.pi * w[24], q[4], q[5])
        ry(math.pi * w[21], q[7]); ry.ctrl(math.pi * w[25], q[7], q[6])
        ry.ctrl(math.pi * w[26], q[6], q[7])
        ry(math.pi * w[22], q[9]); ry.ctrl(math.pi * w[27], q[9], q[8])
        ry.ctrl(math.pi * w[28], q[8], q[9])
    b41_0 = mz(q[4]); b41_1 = mz(q[5])
    b42_0 = mz(q[6]); b42_1 = mz(q[7])
    b43_0 = mz(q[8]); b43_1 = mz(q[9])

    # ── Phase 4: atom 5  w[29:44] ─────────────────────────────────────────
    if a4_0: x(q[2])
    if a4_1: x(q[3])
    if b41_0: x(q[4])
    if b41_1: x(q[5])
    if b42_0: x(q[6])
    if b42_1: x(q[7])
    if b43_0: x(q[8])
    if b43_1: x(q[9])
    if a4_0 or a4_1:
        ry(math.pi * w[29], q[2]); ry(math.pi * w[30], q[3])
        ry.ctrl(math.pi * w[31], q[2], q[3])
    a5_0 = mz(q[2]); a5_1 = mz(q[3])
    if a5_0 or a5_1:
        ry(math.pi * w[32], q[5]); ry.ctrl(math.pi * w[36], q[5], q[4])
        ry.ctrl(math.pi * w[37], q[4], q[5])
        ry(math.pi * w[33], q[7]); ry.ctrl(math.pi * w[38], q[7], q[6])
        ry.ctrl(math.pi * w[39], q[6], q[7])
        ry(math.pi * w[34], q[9]); ry.ctrl(math.pi * w[40], q[9], q[8])
        ry.ctrl(math.pi * w[41], q[8], q[9])
        ry(math.pi * w[35], q[11]); ry.ctrl(math.pi * w[42], q[11], q[10])
        ry.ctrl(math.pi * w[43], q[10], q[11])
    b51_0 = mz(q[4]); b51_1 = mz(q[5])
    b52_0 = mz(q[6]); b52_1 = mz(q[7])
    b53_0 = mz(q[8]); b53_1 = mz(q[9])
    b54_0 = mz(q[10]); b54_1 = mz(q[11])

    # ── Phase 5: atom 6  w[44:62] ─────────────────────────────────────────
    if a5_0: x(q[2])
    if a5_1: x(q[3])
    if b51_0: x(q[4])
    if b51_1: x(q[5])
    if b52_0: x(q[6])
    if b52_1: x(q[7])
    if b53_0: x(q[8])
    if b53_1: x(q[9])
    if b54_0: x(q[10])
    if b54_1: x(q[11])
    if a5_0 or a5_1:
        ry(math.pi * w[44], q[2]); ry(math.pi * w[45], q[3])
        ry.ctrl(math.pi * w[46], q[2], q[3])
    a6_0 = mz(q[2]); a6_1 = mz(q[3])
    if a6_0 or a6_1:
        ry(math.pi * w[47], q[5]); ry.ctrl(math.pi * w[52], q[5], q[4])
        ry.ctrl(math.pi * w[53], q[4], q[5])
        ry(math.pi * w[48], q[7]); ry.ctrl(math.pi * w[54], q[7], q[6])
        ry.ctrl(math.pi * w[55], q[6], q[7])
        ry(math.pi * w[49], q[9]); ry.ctrl(math.pi * w[56], q[9], q[8])
        ry.ctrl(math.pi * w[57], q[8], q[9])
        ry(math.pi * w[50], q[11]); ry.ctrl(math.pi * w[58], q[11], q[10])
        ry.ctrl(math.pi * w[59], q[10], q[11])
        ry(math.pi * w[51], q[13]); ry.ctrl(math.pi * w[60], q[13], q[12])
        ry.ctrl(math.pi * w[61], q[12], q[13])
    b61_0 = mz(q[4]); b61_1 = mz(q[5])
    b62_0 = mz(q[6]); b62_1 = mz(q[7])
    b63_0 = mz(q[8]); b63_1 = mz(q[9])
    b64_0 = mz(q[10]); b64_1 = mz(q[11])
    b65_0 = mz(q[12]); b65_1 = mz(q[13])

    # ── Phase 6: atom 7  w[62:83] ─────────────────────────────────────────
    if a6_0: x(q[2])
    if a6_1: x(q[3])
    if b61_0: x(q[4])
    if b61_1: x(q[5])
    if b62_0: x(q[6])
    if b62_1: x(q[7])
    if b63_0: x(q[8])
    if b63_1: x(q[9])
    if b64_0: x(q[10])
    if b64_1: x(q[11])
    if b65_0: x(q[12])
    if b65_1: x(q[13])
    if a6_0 or a6_1:
        ry(math.pi * w[62], q[2]); ry(math.pi * w[63], q[3])
        ry.ctrl(math.pi * w[64], q[2], q[3])
    a7_0 = mz(q[2]); a7_1 = mz(q[3])
    if a7_0 or a7_1:
        ry(math.pi * w[65], q[5]); ry.ctrl(math.pi * w[71], q[5], q[4])
        ry.ctrl(math.pi * w[72], q[4], q[5])
        ry(math.pi * w[66], q[7]); ry.ctrl(math.pi * w[73], q[7], q[6])
        ry.ctrl(math.pi * w[74], q[6], q[7])
        ry(math.pi * w[67], q[9]); ry.ctrl(math.pi * w[75], q[9], q[8])
        ry.ctrl(math.pi * w[76], q[8], q[9])
        ry(math.pi * w[68], q[11]); ry.ctrl(math.pi * w[77], q[11], q[10])
        ry.ctrl(math.pi * w[78], q[10], q[11])
        ry(math.pi * w[69], q[13]); ry.ctrl(math.pi * w[79], q[13], q[12])
        ry.ctrl(math.pi * w[80], q[12], q[13])
        ry(math.pi * w[70], q[15]); ry.ctrl(math.pi * w[81], q[15], q[14])
        ry.ctrl(math.pi * w[82], q[14], q[15])
    b71_0 = mz(q[4]); b71_1 = mz(q[5])
    b72_0 = mz(q[6]); b72_1 = mz(q[7])
    b73_0 = mz(q[8]); b73_1 = mz(q[9])
    b74_0 = mz(q[10]); b74_1 = mz(q[11])
    b75_0 = mz(q[12]); b75_1 = mz(q[13])
    b76_0 = mz(q[14]); b76_1 = mz(q[15])

    # ── Phase 7: atom 8  w[83:107] ────────────────────────────────────────
    if a7_0: x(q[2])
    if a7_1: x(q[3])
    if b71_0: x(q[4])
    if b71_1: x(q[5])
    if b72_0: x(q[6])
    if b72_1: x(q[7])
    if b73_0: x(q[8])
    if b73_1: x(q[9])
    if b74_0: x(q[10])
    if b74_1: x(q[11])
    if b75_0: x(q[12])
    if b75_1: x(q[13])
    if b76_0: x(q[14])
    if b76_1: x(q[15])
    if a7_0 or a7_1:
        ry(math.pi * w[83], q[2]); ry(math.pi * w[84], q[3])
        ry.ctrl(math.pi * w[85], q[2], q[3])
    a8_0 = mz(q[2]); a8_1 = mz(q[3])
    if a8_0 or a8_1:
        ry(math.pi * w[86], q[5]); ry.ctrl(math.pi * w[93], q[5], q[4])
        ry.ctrl(math.pi * w[94], q[4], q[5])
        ry(math.pi * w[87], q[7]); ry.ctrl(math.pi * w[95], q[7], q[6])
        ry.ctrl(math.pi * w[96], q[6], q[7])
        ry(math.pi * w[88], q[9]); ry.ctrl(math.pi * w[97], q[9], q[8])
        ry.ctrl(math.pi * w[98], q[8], q[9])
        ry(math.pi * w[89], q[11]); ry.ctrl(math.pi * w[99], q[11], q[10])
        ry.ctrl(math.pi * w[100], q[10], q[11])
        ry(math.pi * w[90], q[13]); ry.ctrl(math.pi * w[101], q[13], q[12])
        ry.ctrl(math.pi * w[102], q[12], q[13])
        ry(math.pi * w[91], q[15]); ry.ctrl(math.pi * w[103], q[15], q[14])
        ry.ctrl(math.pi * w[104], q[14], q[15])
        ry(math.pi * w[92], q[17]); ry.ctrl(math.pi * w[105], q[17], q[16])
        ry.ctrl(math.pi * w[106], q[16], q[17])
    b81_0 = mz(q[4]); b81_1 = mz(q[5])
    b82_0 = mz(q[6]); b82_1 = mz(q[7])
    b83_0 = mz(q[8]); b83_1 = mz(q[9])
    b84_0 = mz(q[10]); b84_1 = mz(q[11])
    b85_0 = mz(q[12]); b85_1 = mz(q[13])
    b86_0 = mz(q[14]); b86_1 = mz(q[15])
    b87_0 = mz(q[16]); b87_1 = mz(q[17])

    # ── Phase 8: atom 9  w[107:134] ───────────────────────────────────────
    if a8_0: x(q[2])
    if a8_1: x(q[3])
    if b81_0: x(q[4])
    if b81_1: x(q[5])
    if b82_0: x(q[6])
    if b82_1: x(q[7])
    if b83_0: x(q[8])
    if b83_1: x(q[9])
    if b84_0: x(q[10])
    if b84_1: x(q[11])
    if b85_0: x(q[12])
    if b85_1: x(q[13])
    if b86_0: x(q[14])
    if b86_1: x(q[15])
    if b87_0: x(q[16])
    if b87_1: x(q[17])
    if a8_0 or a8_1:
        ry(math.pi * w[107], q[2]); ry(math.pi * w[108], q[3])
        ry.ctrl(math.pi * w[109], q[2], q[3])
    a9_0 = mz(q[2]); a9_1 = mz(q[3])
    if a9_0 or a9_1:
        ry(math.pi * w[110], q[5]); ry.ctrl(math.pi * w[118], q[5], q[4])
        ry.ctrl(math.pi * w[119], q[4], q[5])
        ry(math.pi * w[111], q[7]); ry.ctrl(math.pi * w[120], q[7], q[6])
        ry.ctrl(math.pi * w[121], q[6], q[7])
        ry(math.pi * w[112], q[9]); ry.ctrl(math.pi * w[122], q[9], q[8])
        ry.ctrl(math.pi * w[123], q[8], q[9])
        ry(math.pi * w[113], q[11]); ry.ctrl(math.pi * w[124], q[11], q[10])
        ry.ctrl(math.pi * w[125], q[10], q[11])
        ry(math.pi * w[114], q[13]); ry.ctrl(math.pi * w[126], q[13], q[12])
        ry.ctrl(math.pi * w[127], q[12], q[13])
        ry(math.pi * w[115], q[15]); ry.ctrl(math.pi * w[128], q[15], q[14])
        ry.ctrl(math.pi * w[129], q[14], q[15])
        ry(math.pi * w[116], q[17]); ry.ctrl(math.pi * w[130], q[17], q[16])
        ry.ctrl(math.pi * w[131], q[16], q[17])
        ry(math.pi * w[117], q[19]); ry.ctrl(math.pi * w[132], q[19], q[18])
        ry.ctrl(math.pi * w[133], q[18], q[19])
    # ★ bond-9 全部命名，不依賴 __global__
    b91_0 = mz(q[4]);  b91_1 = mz(q[5])
    b92_0 = mz(q[6]);  b92_1 = mz(q[7])
    b93_0 = mz(q[8]);  b93_1 = mz(q[9])
    b94_0 = mz(q[10]); b94_1 = mz(q[11])
    b95_0 = mz(q[12]); b95_1 = mz(q[13])
    b96_0 = mz(q[14]); b96_1 = mz(q[15])
    b97_0 = mz(q[16]); b97_1 = mz(q[17])
    b98_0 = mz(q[18]); b98_1 = mz(q[19])


# ===========================================================================
# 90 個命名暫存器（與 _qmg_n9_v9 mz() 命名完全對應）
# ===========================================================================

_N9_ALL_REGS: list[str] = [
    'a1_0', 'a1_1', 'a2_0', 'a2_1', 'b21_0', 'b21_1',
    'a3_0', 'a3_1', 'b31_0', 'b31_1', 'b32_0', 'b32_1',
    'a4_0', 'a4_1', 'b41_0', 'b41_1', 'b42_0', 'b42_1', 'b43_0', 'b43_1',
    'a5_0', 'a5_1', 'b51_0', 'b51_1', 'b52_0', 'b52_1',
    'b53_0', 'b53_1', 'b54_0', 'b54_1',
    'a6_0', 'a6_1', 'b61_0', 'b61_1', 'b62_0', 'b62_1', 'b63_0', 'b63_1',
    'b64_0', 'b64_1', 'b65_0', 'b65_1',
    'a7_0', 'a7_1', 'b71_0', 'b71_1', 'b72_0', 'b72_1', 'b73_0', 'b73_1',
    'b74_0', 'b74_1', 'b75_0', 'b75_1', 'b76_0', 'b76_1',
    'a8_0', 'a8_1', 'b81_0', 'b81_1', 'b82_0', 'b82_1', 'b83_0', 'b83_1',
    'b84_0', 'b84_1', 'b85_0', 'b85_1', 'b86_0', 'b86_1', 'b87_0', 'b87_1',
    'a9_0', 'a9_1',
    'b91_0', 'b91_1', 'b92_0', 'b92_1', 'b93_0', 'b93_1', 'b94_0', 'b94_1',
    'b95_0', 'b95_1', 'b96_0', 'b96_1', 'b97_0', 'b97_1', 'b98_0', 'b98_1',
]
assert len(_N9_ALL_REGS) == 90


# ===========================================================================
# smoke test kernel（file-level 定義，可被正確 inspect）
# ===========================================================================

@cudaq.kernel
def _smoke_kernel_v9():
    q = cudaq.qvector(1)
    h(q[0])
    mz(q[0])


# ===========================================================================
# 工具函式
# ===========================================================================

def _check_cudaq_version_volta_compat() -> tuple[str, bool]:
    try:
        ver_str = cudaq.__version__
        match = re.search(r'(\d+)\.(\d+)\.(\d+)', ver_str)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            return ver_str, (major, minor) <= (0, 7)
        return ver_str, True
    except Exception:
        return "unknown", True


def _gpu_target_available() -> bool:
    """正確檢查 nvidia target 是否可用（修正 0.7.1 target string 帶 newline 的 bug）。"""
    try:
        for t in cudaq.get_targets():
            name = str(t).split('\n')[0].replace('Target', '').strip()
            if name == 'nvidia':
                return True
        return False
    except Exception:
        return False


def _verify_gpu_smoke() -> bool:
    try:
        result = cudaq.sample(_smoke_kernel_v9, shots_count=16)
        return len(dict(result.items())) > 0
    except Exception as e:
        warnings.warn(f"[CUDAQ] GPU smoke test 失敗：{e}")
        return False


_CUDAQ_TARGET_MAP = {
    "cudaq_qpp":         "qpp-cpu",
    "qpp-cpu":           "qpp-cpu",
    "cudaq_nvidia":      "nvidia",
    "nvidia":            "nvidia",
    "cudaq_nvidia_fp64": "nvidia-fp64",
    "nvidia-fp64":       "nvidia-fp64",
    "qiskit_aer":        "qpp-cpu",
}
_GPU_TARGETS = {"nvidia", "nvidia-fp64"}


def _set_target_safe(target_name: str) -> str:
    ver_str, is_compat = _check_cudaq_version_volta_compat()
    if target_name in _GPU_TARGETS and not is_compat:
        raise RuntimeError(
            f"[CUDAQ] CUDA-Q {ver_str} 不支援 V100 (sm_70)。"
            f"請安裝：pip install cuda-quantum-cu12==0.7.1"
        )
    try:
        cudaq.set_target(target_name)
    except Exception as e:
        raise RuntimeError(f"[CUDAQ] set_target('{target_name}') 失敗：{e}") from e
    if target_name in _GPU_TARGETS:
        if _verify_gpu_smoke():
            print(f"[CUDAQ] GPU target '{target_name}' 驗證通過 ✓")
        else:
            warnings.warn("[CUDAQ] GPU smoke test 異常，可能仍在 CPU 執行。")
    return target_name


# ===========================================================================
# 90-bit bitstring 重建
# ===========================================================================

def _reconstruct_bitstrings_n9(result) -> dict[str, int]:
    """用 90 個命名暫存器重建 bitstring，不依賴 __global__。"""
    try:
        reg_data = {reg: result.get_sequential_data(reg) for reg in _N9_ALL_REGS}
    except AttributeError:
        warnings.warn("[CUDAQ] get_sequential_data() 不存在，使用 items() fallback。")
        counts: dict[str, int] = {}
        for bs_raw, cnt in result.items():
            if len(bs_raw) == 90:
                counts[bs_raw] = counts.get(bs_raw, 0) + cnt
        return counts
    except Exception as e:
        warnings.warn(f"[CUDAQ] get_sequential_data() 失敗：{e}")
        return {}

    n_shots = len(reg_data.get('a1_0', []))
    if n_shots == 0:
        warnings.warn("[CUDAQ] n_shots=0。")
        return {}

    counts: dict[str, int] = {}
    malformed = 0
    for i in range(n_shots):
        bs = ''.join(reg_data[reg][i] for reg in _N9_ALL_REGS)
        if len(bs) != 90:
            malformed += 1
            continue
        counts[bs] = counts.get(bs, 0) + 1
    if malformed:
        warnings.warn(f"[CUDAQ] {malformed}/{n_shots} shots bitstring 長度異常。")
    return counts


# ===========================================================================
# MoleculeGeneratorCUDAQ  (v9.2)
# ===========================================================================

class MoleculeGeneratorCUDAQ:
    """CUDA-Q 版分子生成器（CUDA-Q 0.7.1 / V100 sm_70，v9.2）。"""

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
            raise NotImplementedError(f"目前僅支援 num_heavy_atom=9。")

        self.num_heavy_atom           = num_heavy_atom
        self.all_weight_vector        = (
            np.array(all_weight_vector, dtype=np.float64)
            if all_weight_vector is not None else None
        )
        self.backend_name             = backend_name
        self.remove_bond_disconnection = remove_bond_disconnection
        self.length_all_weight_vector = int(
            8 + (num_heavy_atom - 2) * (num_heavy_atom + 3) * 3 / 2
        )  # 134

        # DynamicCircuitBuilderCUDAQ：僅用 prepare_weights / apply_bond_disconnection
        self.circuit_builder = DynamicCircuitBuilderCUDAQ(
            num_heavy_atom            = num_heavy_atom,
            temperature               = temperature,
            remove_bond_disconnection = remove_bond_disconnection,
            chemistry_constraint      = chemistry_constraint,
        )
        self.data_generator = MoleculeQuantumStateGenerator(
            heavy_atom_size = num_heavy_atom, ncpus=1, sanitize_method="strict",
        )

        actual_target       = _CUDAQ_TARGET_MAP.get(backend_name, "qpp-cpu")
        self._active_target = _set_target_safe(actual_target)

        # ★ v9.2：使用本檔 module-level kernel，不 import 跨模組
        self._kernel = _qmg_n9_v9

        ver_str, _ = _check_cudaq_version_volta_compat()
        print(
            f"[CUDAQ] Generator initialized (v9.2).\n"
            f"  cudaq version  : {ver_str}\n"
            f"  active target  : {self._active_target}\n"
            f"  GPU available  : {_gpu_target_available()}\n"
            f"  kernel         : _qmg_n9_v9 (local, MLIR compiled once at import)\n"
            f"  reconstruction : 90 named registers, no __global__"
        )

    def update_weight_vector(self, w: Union[List[float], np.ndarray]) -> None:
        self.all_weight_vector = np.array(w, dtype=np.float64)

    def sample_molecule(self, num_sample: int, random_seed: int = 0) -> Tuple[dict, float, float]:
        assert self.all_weight_vector is not None
        w = self.all_weight_vector
        assert len(w) == self.length_all_weight_vector

        try:
            cudaq.set_random_seed(random_seed)
        except AttributeError:
            pass

        w_list = self.circuit_builder.prepare_weights(w)

        # ★ v9.2：同檔案 kernel，broadcast 不觸發
        result = cudaq.sample(self._kernel, w_list, shots_count=num_sample)

        raw_counts = _reconstruct_bitstrings_n9(result)
        if not raw_counts:
            warnings.warn("[CUDAQ] raw_counts 為空，validity=0。")
            return {}, 0.0, 0.0

        smiles_dict: dict[str, int] = {}
        num_valid = 0
        for bs, cnt in raw_counts.items():
            bs_fixed = self.circuit_builder.apply_bond_disconnection_correction(bs)
            qs       = self.data_generator.post_process_quantum_state(bs_fixed, reverse=False)
            smi      = self.data_generator.QuantumStateToSmiles(qs)
            smiles_dict[smi] = smiles_dict.get(smi, 0) + cnt
            if smi and smi != "None":
                num_valid += cnt

        validity   = num_valid / num_sample
        n_unique   = len([k for k in smiles_dict if k and k != "None"])
        uniqueness = n_unique / num_valid if num_valid > 0 else 0.0
        return smiles_dict, validity, uniqueness


MoleculeGenerator = MoleculeGeneratorCUDAQ


# ===========================================================================
# 快速功能驗證
# ===========================================================================
if __name__ == "__main__":
    import time
    print("=== MoleculeGeneratorCUDAQ 功能驗證 (v9.2) ===")
    ver_str, is_compat = _check_cudaq_version_volta_compat()
    print(f"CUDA-Q: {ver_str}  V100 compat: {'✓' if is_compat else '⚠'}")
    print(f"GPU target 可用: {_gpu_target_available()}")

    cwg = ConditionalWeightsGenerator(9, smarts=None)
    w   = cwg.generate_conditional_random_weights(random_seed=42)

    for backend, shots, lbl in [("cudaq_nvidia", 500, "GPU"), ("cudaq_qpp", 200, "CPU")]:
        print(f"\n[{lbl}] {backend}, {shots} shots")
        try:
            gen = MoleculeGeneratorCUDAQ(9, all_weight_vector=w, backend_name=backend)
            t0  = time.time()
            sd, v, u = gen.sample_molecule(shots)
            print(f"  V={v:.3f}  U={u:.3f}  V×U={v*u:.4f}  ({time.time()-t0:.1f}s)")
            valid = [k for k in sd if k and k != "None"]
            print(f"  有效分子：{len(valid)} 種，範例：{valid[:3]}")
            print(f"  {'✓ 正常' if v>0.3 and u>0.3 else '⚠ 偏低' if v>0 else '✗ validity=0'}")
        except Exception as e:
            print(f"  ✗ 失敗：{e}")