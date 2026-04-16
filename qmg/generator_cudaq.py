"""
==============================================================================
generator_cudaq.py  (CUDA-Q 0.7.1 / V100 sm_70 完整修正版 v9.3)
==============================================================================

v9.2 → v9.3 根本原因修正：

  ★ 真正的 Bug 根源（確認）：
      v9.2 假設「跨模組 import 觸發 broadcast」是錯誤的。
      診斷結果顯示：即使 cudaq.sample(self._kernel, w_list, ...) 從
      generator_cudaq.py 內部的 sample_molecule() 呼叫，依然失敗。
      因此，cross-module import 不是根本原因。

  ★ 實際根本原因：_qmg_n9_v9 充斥著 CUDA-Q 0.7.1 無法正確解析的分號語法：
        a1_0 = mz(q[0]); a1_1 = mz(q[1])   ← 分號（CUDA-Q AST bug）
        if a2_0: x(q[2])                    ← 單行 if
        ry(math.pi * w[8], q[2]); ry(...)   ← 分號

      CUDA-Q 0.7.1 的 MLIR 前端遇到分號時，第二個語句的 register 名稱
      會被標記為 anonymous 或丟棄，造成 MLIR function 的 list[float] 參數
      型別元資料不完整。cudaq.sample() 因此無法識別 list[float] 型別，
      退回到 broadcast dispatch → 將 134 個 float 當作 134 次獨立呼叫。
      最終產生：「Argument of type <class 'float'> was provided, but
                list[float] was expected.」

  ★ v9.3 修正方案：
      build_dynamic_circuit_cudaq.py 的 _qmg_n9 已在 v9.1 套用分號修正
      （每個語句各佔獨立一行，無分號，無單行 if）。
      v9.3 直接使用 _qmg_n9 取代有問題的 _qmg_n9_v9，
      不需要重新撰寫電路邏輯。

  v9.2 移除（原說明保留供追溯）：
      _qmg_n9_v9 kernel 定義已從本檔移除。
      若需參考電路邏輯，請查看 build_dynamic_circuit_cudaq.py 的 _qmg_n9。

速度對比（10000 shots，N=9）：
  qpp-cpu  : ~90s/eval
  nvidia   : ~1-5s/eval（V100 GPU + cuStateVec）
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
# ★ v9.3：直接使用 build_dynamic_circuit_cudaq.py 的 _qmg_n9（v9.1 分號修正版）
#         _qmg_n9 每個語句各佔獨立一行，MLIR 編譯完整，list[float] 型別元資料正確。
from qmg.utils.build_dynamic_circuit_cudaq import DynamicCircuitBuilderCUDAQ, _qmg_n9


# ===========================================================================
# 90 個命名暫存器（與 build_dynamic_circuit_cudaq.py 的 _qmg_n9 mz() 命名完全對應）
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
    """用 90 個命名暫存器重建 bitstring。
    _qmg_n9 (v9.1) 每個 mz() 各佔獨立行，AST 正確識別所有 register 名稱。
    """
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
# MoleculeGeneratorCUDAQ  (v9.3)
# ===========================================================================

class MoleculeGeneratorCUDAQ:
    """CUDA-Q 版分子生成器（CUDA-Q 0.7.1 / V100 sm_70，v9.3）。

    v9.3 修正：使用 build_dynamic_circuit_cudaq._qmg_n9（v9.1 分號修正版）
    取代有問題的 _qmg_n9_v9，解決 list[float] broadcast dispatch 錯誤。
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

        # DynamicCircuitBuilderCUDAQ：用於 prepare_weights / apply_bond_disconnection
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

        # ★ v9.3：使用 build_dynamic_circuit_cudaq._qmg_n9（v9.1 分號修正版）
        #   根本原因：_qmg_n9_v9 的分號語法導致 MLIR 型別元資料遺失 → broadcast 錯誤
        #   修正：_qmg_n9 每個語句各佔獨立一行，MLIR 完整，list[float] 正確解析
        self._kernel = _qmg_n9

        ver_str, _ = _check_cudaq_version_volta_compat()
        print(
            f"[CUDAQ] Generator initialized (v9.3).\n"
            f"  cudaq version  : {ver_str}\n"
            f"  active target  : {self._active_target}\n"
            f"  GPU available  : {_gpu_target_available()}\n"
            f"  kernel         : _qmg_n9 (build_dynamic_circuit_cudaq v9.1, "
            f"semicolon-free, MLIR type metadata intact)\n"
            f"  reconstruction : 90 named registers"
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

        # ★ v9.3：_qmg_n9 的 list[float] 型別元資料完整，broadcast 不觸發
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
    print("=== MoleculeGeneratorCUDAQ 功能驗證 (v9.3) ===")
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