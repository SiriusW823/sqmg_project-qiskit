"""
==============================================================================
generator_cudaq.py  (CUDA-Q 0.7.1 / V100 sm_70 完整修正版 v9.5)
==============================================================================

v9.4 → v9.5 關鍵 Bug 修正：

  ★ [BUG-FIX] _reconstruct_bitstrings_n9 型別判斷錯誤（V=0 根本原因）

    症狀：
      所有評估結果 V=0.000, U=0.000，所有 bitstring 為全 1（'111...1'）

    根本原因：
      CUDA-Q 0.7.1 的 get_sequential_data() 回傳 list[str]，
      每個元素是字串 '0' 或 '1'，而非 bool 或 int。

      原版程式碼：
        buf[i * 90] = 1 if bit else 0

      Python 中非空字串皆為 truthy：
        bool('0') == True   ← '0' 是非空字串，判斷為 True！
        bool('1') == True
      → 無論 bit 是 '0' 還是 '1'，都填入 1
      → 所有 shot 的 90-bit bitstring 全部變成 '111...1'
      → 全部轉換為無效分子 → validity = 0

    修正：
      buf[i * 90] = 1 if bit == '1' else 0
      或等價：
      buf[i * 90] = int(bit)   # '0'→0, '1'→1，直接轉型最安全

  ★ [附帶修正] result.items() fallback 路徑
    CUDA-Q nvidia target 的 result.items() 回傳 20-bit 全局 bitstring
    （對應 20 個量子位元），不是 90-bit 命名暫存器串接。
    舊版 fallback 判斷 len(bs_raw) == 90 永遠為 False，完全無效。
    修正：fallback 路徑加入更詳細的診斷日誌，
    主要重建路徑仍依賴 get_sequential_data（已知可回傳 90 個 register）。

v9.4 修正保留：
  - del result → gc.collect() → malloc_trim(0) 防止 OOM
  - bytearray buffer 低記憶體模式
  - 週期性 generator 重建支援

v9.3 修正保留（分號 AST bug）：
    使用 build_dynamic_circuit_cudaq._qmg_n9（v9.1 分號修正版）
==============================================================================
"""
from __future__ import annotations

import ctypes
import gc
import re
import warnings
import numpy as np
from typing import List, Union, Tuple

import cudaq

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from qmg.utils.chemistry_data_processing import MoleculeQuantumStateGenerator
from qmg.utils.weight_generator import ConditionalWeightsGenerator
from qmg.utils.build_dynamic_circuit_cudaq import DynamicCircuitBuilderCUDAQ, _qmg_n9


# ===========================================================================
# 記憶體工具
# ===========================================================================

def _free_cpp_heap() -> None:
    """
    強制 glibc 將 C++ heap 已釋放的記憶體歸還給 OS。
    CUDA-Q 0.7.1 的 pybind11 C++ binding 不會主動釋放，
    需呼叫 malloc_trim(0) 才能讓 RSS 下降。
    在非 glibc 環境（macOS / musl）靜默忽略。
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ===========================================================================
# 90 個命名暫存器（與 build_dynamic_circuit_cudaq.py 的 _qmg_n9 完全對應）
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
# smoke test kernel
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
    try:
        for t in cudaq.get_targets():
            # 相容不同版本的 target 字串格式
            t_str = str(t)
            if 'nvidia' in t_str.lower():
                return True
        return False
    except Exception:
        return False


def _verify_gpu_smoke() -> bool:
    try:
        result = cudaq.sample(_smoke_kernel_v9, shots_count=16)
        ok = len(dict(result.items())) > 0
        del result
        gc.collect()
        _free_cpp_heap()
        return ok
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
# 90-bit bitstring 重建（v9.5 型別修正版）
# ===========================================================================

def _reconstruct_bitstrings_n9(result) -> dict[str, int]:
    """
    用 90 個命名暫存器重建 bitstring。

    v9.5 關鍵修正：
      CUDA-Q 0.7.1 的 get_sequential_data() 回傳 list[str]，
      每個元素是字串 '0' 或 '1'，而非 bool 或 int。

      原版：1 if bit else 0
        → Python 非空字串皆為 truthy，'0' 也是 True
        → '0' 和 '1' 都被轉為 1 → 全部 bitstring 變成 '111...1'

      修正：int(bit)
        → '0' → 0，'1' → 1，型別轉換語意明確且安全

    記憶體優化（v9.4 保留）：
      逐 register 讀取後立即 del，不保留整個 reg_data dict，
      使用 bytearray buffer 避免大量中間字串物件。
    """
    try:
        # Step 1: 確認 n_shots
        first_data = result.get_sequential_data(_N9_ALL_REGS[0])
        n_shots = len(first_data)
        if n_shots == 0:
            warnings.warn("[CUDAQ] n_shots=0，get_sequential_data 回傳空 list。")
            return {}

        # Step 2: 預建 bytearray buffer
        # shape: n_shots × 90，每個 shot 佔 90 bytes
        buf = bytearray(n_shots * 90)

        # Step 3: 第 0 個 register 已讀入 first_data，直接填入
        # ★ v9.5 修正：int(bit) 取代 1 if bit else 0
        #   '0' → 0，'1' → 1
        #   同時相容 str / bool / int 三種可能的回傳型別
        for i, bit in enumerate(first_data):
            buf[i * 90] = int(bit)
        del first_data  # 立即釋放

        # Step 4: 逐 register 讀取填入 buffer，不保留整個 dict
        for reg_idx, reg in enumerate(_N9_ALL_REGS[1:], start=1):
            reg_data = result.get_sequential_data(reg)
            for i, bit in enumerate(reg_data):
                # ★ v9.5 修正：int(bit) 取代 1 if bit else 0
                buf[i * 90 + reg_idx] = int(bit)
            del reg_data  # 立即釋放，不累積

        # Step 5: 從 buffer 組裝 bitstring 並計數
        counts: dict[str, int] = {}
        malformed = 0
        for i in range(n_shots):
            row = buf[i * 90: i * 90 + 90]
            if len(row) != 90:
                malformed += 1
                continue
            bs = ''.join('1' if b else '0' for b in row)
            counts[bs] = counts.get(bs, 0) + 1

        del buf  # 釋放 bytearray

        if malformed:
            warnings.warn(f"[CUDAQ] {malformed}/{n_shots} shots bitstring 長度異常。")

        # 診斷日誌：確認 bitstring 分布合理（非全 0 也非全 1）
        if counts:
            sample_bs = next(iter(counts))
            ones_ratio = sample_bs.count('1') / 90
            if ones_ratio > 0.95:
                warnings.warn(
                    f"[CUDAQ] 警告：bitstring 中 '1' 比例 = {ones_ratio:.2f}（過高，可能仍有型別問題）"
                )
            elif ones_ratio < 0.01:
                warnings.warn(
                    f"[CUDAQ] 警告：bitstring 中 '1' 比例 = {ones_ratio:.2f}（過低，可能量子電路未執行）"
                )

        return counts

    except AttributeError:
        # get_sequential_data() 不存在 → 記錄詳細診斷並回傳空
        warnings.warn(
            "[CUDAQ] get_sequential_data() 不存在。\n"
            "  此 CUDA-Q 版本可能不支援命名暫存器讀取。\n"
            "  result.items() 回傳的是全局 bitstring（20-bit），無法重建 90-bit 暫存器資料。\n"
            "  請確認使用 CUDA-Q 0.7.1。"
        )
        return {}
    except Exception as e:
        warnings.warn(f"[CUDAQ] _reconstruct_bitstrings_n9 失敗：{e}")
        import traceback
        traceback.print_exc()
        return {}


# ===========================================================================
# MoleculeGeneratorCUDAQ  (v9.5)
# ===========================================================================

class MoleculeGeneratorCUDAQ:
    """CUDA-Q 版分子生成器（CUDA-Q 0.7.1 / V100 sm_70，v9.5）。

    v9.5：修正 _reconstruct_bitstrings_n9 的型別判斷 bug（V=0 根本原因）。
    v9.4：修正 OOM Kill（del result + gc.collect + malloc_trim）。
    v9.3：修正 list[float] broadcast dispatch 錯誤（分號 AST bug）。
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
            raise NotImplementedError(f"目前僅支援 num_heavy_atom=9。")

        self.num_heavy_atom            = num_heavy_atom
        self.all_weight_vector         = (
            np.array(all_weight_vector, dtype=np.float64)
            if all_weight_vector is not None else None
        )
        self.backend_name              = backend_name
        self.remove_bond_disconnection = remove_bond_disconnection
        self.length_all_weight_vector  = int(
            8 + (num_heavy_atom - 2) * (num_heavy_atom + 3) * 3 / 2
        )  # 134

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
        self._kernel        = _qmg_n9

        ver_str, _ = _check_cudaq_version_volta_compat()
        print(
            f"[CUDAQ] Generator initialized (v9.5).\n"
            f"  cudaq version  : {ver_str}\n"
            f"  active target  : {self._active_target}\n"
            f"  GPU available  : {_gpu_target_available()}\n"
            f"  kernel         : _qmg_n9 (v9.1 semicolon-free)\n"
            f"  reconstruction : 90 named registers (v9.5 int(bit) fix)\n"
            f"  memory mgmt    : malloc_trim enabled ✓"
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

        # ── 量子電路採樣 ─────────────────────────────────────────────────
        result = cudaq.sample(self._kernel, w_list, shots_count=num_sample)

        # ── bitstring 重建（低記憶體 + v9.5 型別修正）────────────────────
        raw_counts = _reconstruct_bitstrings_n9(result)

        # ★ v9.4 Fix-1：明確刪除 result，觸發 C++ SampleResult 析構
        del result
        del w_list
        gc.collect()
        _free_cpp_heap()   # 將 C++ heap 已釋放記憶體歸還 OS

        if not raw_counts:
            warnings.warn("[CUDAQ] raw_counts 為空，validity=0。")
            return {}, 0.0, 0.0

        # ── SMILES 轉換 ───────────────────────────────────────────────────
        smiles_dict: dict[str, int] = {}
        num_valid = 0
        for bs, cnt in raw_counts.items():
            bs_fixed = self.circuit_builder.apply_bond_disconnection_correction(bs)
            qs       = self.data_generator.post_process_quantum_state(bs_fixed, reverse=False)
            smi      = self.data_generator.QuantumStateToSmiles(qs)
            smiles_dict[smi] = smiles_dict.get(smi, 0) + cnt
            if smi and smi != "None":
                num_valid += cnt

        # ★ v9.4 Fix-2：釋放中間物件
        del raw_counts
        gc.collect()

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
    print("=== MoleculeGeneratorCUDAQ 功能驗證 (v9.5) ===")
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
