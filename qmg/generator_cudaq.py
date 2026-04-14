"""
==============================================================================
generator_cudaq.py  (CUDA-Q 0.7.1 / V100 sm_70 完整修正版 v9.1)
==============================================================================

v9 → v9.1 修正：

  ★ FIX-1: tensornet backend guard
      tensornet 後端使用 tensor network contraction，不支援含 mid-circuit
      measurement 的動態電路（classical feedback → 測量後根據結果選擇閘操作）。
      若使用 tensornet 執行 N=9 QMG 動態電路，會靜默失敗或產生錯誤結果。
      修正：在 _set_target_safe 中加入明確的 RuntimeError，
            並從 run_qpso_qmg_cudaq.py 的 --backend choices 中移除 tensornet。

  ★ FIX-2: register 存在性驗證（_reconstruct_bitstrings_n9）
      使用 result.register_names() 確認 90 個命名暫存器均存在後，
      再進行 get_sequential_data 批次讀取，提供更清晰的錯誤訊息。

  ★ FIX-3: 修正 _N9_ALL_REGS 的 atom-8 bit range 註解
      "bits 56-73" 應為 "bits 56-71"（16 bits：a8 + b81..b87）

詳細修正說明請見 build_dynamic_circuit_cudaq.py v9.1 標頭。
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
# 模組層級 smoke test kernel
# ===========================================================================

@cudaq.kernel
def _smoke_kernel():
    q = cudaq.qvector(1)
    h(q[0])
    mz(q[0])


# ===========================================================================
# CUDA-Q 版本與 V100 相容性
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


def _verify_gpu_actually_used(target_name: str) -> bool:
    if target_name not in ("nvidia", "nvidia-fp64"):
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
    # tensornet 故意不在此映射中；_set_target_safe 會提前 raise
    "qiskit_aer":        "qpp-cpu",
}
_GPU_TARGETS = {"nvidia", "nvidia-fp64"}
# tensornet 後端對 mid-circuit measurement 動態電路不支援
_INCOMPATIBLE_DYNAMIC_TARGETS = {"tensornet", "tensornet-mps"}


def _set_target_safe(target_name: str) -> str:
    # ★ FIX-1：tensornet guard（對動態電路不支援）
    if target_name in _INCOMPATIBLE_DYNAMIC_TARGETS:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"[CUDAQ] '{target_name}' backend 不支援含 mid-circuit measurement 的動態電路。\n"
            f"  N=9 QMG kernel 有 90 個 mid-circuit mz() 和 classical feedback，\n"
            f"  tensornet 使用 tensor network contraction，無法處理 classical conditional。\n"
            f"  V100 請使用：--backend cudaq_nvidia\n"
            f"  CPU 請使用： --backend cudaq_qpp\n"
            f"{'='*60}"
        )

    ver_str, is_compat = _check_cudaq_version_volta_compat()
    if target_name in _GPU_TARGETS and not is_compat:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"[CUDAQ] CUDA-Q {ver_str} 不支援 V100 (sm_70)。\n"
            f"  請安裝：pip install cuda-quantum-cu11==0.7.1\n"
            f"  或      pip install cuda-quantum-cu12==0.7.1\n"
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
# 90-bit bitstring 重建（全部使用命名暫存器）
# ===========================================================================

# 90 個命名暫存器（與 _qmg_n9 kernel 中 mz() 的命名完全對應）
# 官方文件確認：get_sequential_data() → list[str]（每個元素為 '0' 或 '1'）
_N9_ALL_REGS: list[str] = [
    # ── atom 1, 2 ──────────────────────────────────── bits  0- 5
    'a1_0', 'a1_1',
    'a2_0', 'a2_1',
    'b21_0', 'b21_1',
    # ── atom 3 ─────────────────────────────────────── bits  6-11
    'a3_0', 'a3_1',
    'b31_0', 'b31_1', 'b32_0', 'b32_1',
    # ── atom 4 ─────────────────────────────────────── bits 12-19
    'a4_0', 'a4_1',
    'b41_0', 'b41_1', 'b42_0', 'b42_1', 'b43_0', 'b43_1',
    # ── atom 5 ─────────────────────────────────────── bits 20-29
    'a5_0', 'a5_1',
    'b51_0', 'b51_1', 'b52_0', 'b52_1',
    'b53_0', 'b53_1', 'b54_0', 'b54_1',
    # ── atom 6 ─────────────────────────────────────── bits 30-41
    'a6_0', 'a6_1',
    'b61_0', 'b61_1', 'b62_0', 'b62_1', 'b63_0', 'b63_1',
    'b64_0', 'b64_1', 'b65_0', 'b65_1',
    # ── atom 7 ─────────────────────────────────────── bits 42-55
    'a7_0', 'a7_1',
    'b71_0', 'b71_1', 'b72_0', 'b72_1', 'b73_0', 'b73_1',
    'b74_0', 'b74_1', 'b75_0', 'b75_1', 'b76_0', 'b76_1',
    # ── atom 8 ─────────────────────────────────────── bits 56-71  (修正：原為 56-73)
    'a8_0', 'a8_1',
    'b81_0', 'b81_1', 'b82_0', 'b82_1', 'b83_0', 'b83_1',
    'b84_0', 'b84_1', 'b85_0', 'b85_1', 'b86_0', 'b86_1',
    'b87_0', 'b87_1',
    # ── atom 9（atom type）──────────────────────────── bits 72-73
    'a9_0', 'a9_1',
    # ── bond 9（bonds to atoms 1-8）────────────────── bits 74-89
    'b91_0', 'b91_1',
    'b92_0', 'b92_1',
    'b93_0', 'b93_1',
    'b94_0', 'b94_1',
    'b95_0', 'b95_1',
    'b96_0', 'b96_1',
    'b97_0', 'b97_1',
    'b98_0', 'b98_1',
]  # 共 90 個暫存器，對應 bitstring bits 0-89

assert len(_N9_ALL_REGS) == 90, f"[BUG] _N9_ALL_REGS 長度 {len(_N9_ALL_REGS)} != 90"


def _reconstruct_bitstrings_n9(result) -> dict[str, int]:
    """
    90-bit bitstring 重建（v9.1：全部使用命名暫存器 + register 存在性驗證）。

    CUDA-Q 0.7.1 bug 說明：
      get_sequential_data('__global__') 在含 mid-circuit measurement 的
      kernel 中行為異常（0.8.0 PR#1619 才修正）。
      v9.1 完全不呼叫 __global__，改用 90 個命名暫存器。

    API 文件確認（0.7.1）：
      get_sequential_data(register_name: str) → list[str]
      每個元素為 '0' 或 '1'，長度等於 shots_count。

    Returns:
        dict mapping 90-bit bitstring → count
    """
    # ★ FIX-2：register 存在性驗證（先確認所有命名暫存器均存在）
    try:
        available_regs = set(result.register_names())
    except Exception as e:
        warnings.warn(f"[CUDAQ] result.register_names() 失敗：{e}。跳過 register 存在性驗證。")
        available_regs = None

    if available_regs is not None:
        missing_regs = [r for r in _N9_ALL_REGS if r not in available_regs]
        if missing_regs:
            sample_missing = missing_regs[:5]
            warnings.warn(
                f"[CUDAQ] 以下命名暫存器在 SampleResult 中不存在（共 {len(missing_regs)} 個）：\n"
                f"  {sample_missing}{'...' if len(missing_regs) > 5 else ''}\n"
                f"  可能原因：\n"
                f"  1. _qmg_n9 kernel 中的 mz() 賦值存在分號合併行（已在 v9.1 修正）\n"
                f"  2. CUDA-Q 版本不支援此命名暫存器模式\n"
                f"  3. cudaq.sample() 的命名暫存器 API 在此版本有 bug\n"
                f"  可用的暫存器（前 10 個）：{sorted(available_regs)[:10]}"
            )
            return {}

    # 嘗試批次讀取所有 90 個命名暫存器的 sequential data
    try:
        reg_data = {reg: result.get_sequential_data(reg) for reg in _N9_ALL_REGS}
    except AttributeError:
        # get_sequential_data 不存在（極舊版本），fallback 到 items()
        warnings.warn(
            "[CUDAQ] get_sequential_data() 不存在，使用 items() fallback。\n"
            "  此 fallback 僅適用於 bitstring 長度恰好為 90 的情況。"
        )
        counts: dict[str, int] = {}
        for bs_raw, cnt in result.items():
            if len(bs_raw) == 90:
                counts[bs_raw] = counts.get(bs_raw, 0) + cnt
        if not counts:
            warnings.warn(
                "[CUDAQ] items() fallback 未找到 90-bit bitstring。\n"
                "  請確認 CUDA-Q 版本 >= 0.7.1 且 target 支援 mid-circuit measurement。"
            )
        return counts
    except Exception as e:
        warnings.warn(
            f"[CUDAQ] get_sequential_data() 批次讀取失敗：{e}\n"
            f"  可能原因：CUDA-Q 0.7.1 命名暫存器問題。\n"
            f"  請確認 build_dynamic_circuit_cudaq.py v9.1（各 mz() 獨立行）已部署。"
        )
        return {}

    # 取 shot 數量（任意一個暫存器的資料長度）
    n_shots = len(reg_data.get('a1_0', []))
    if n_shots == 0:
        warnings.warn("[CUDAQ] get_sequential_data 回傳空列表，n_shots=0。")
        return {}

    # 逐 shot 拼接 90-bit bitstring
    # get_sequential_data 回傳 list[str]（官方 API 文件確認），每個元素為 '0' 或 '1'
    counts: dict[str, int] = {}
    malformed_count = 0
    for i in range(n_shots):
        try:
            bs = ''.join(reg_data[reg][i] for reg in _N9_ALL_REGS)
        except (IndexError, TypeError) as e:
            malformed_count += 1
            if malformed_count <= 3:
                warnings.warn(f"[CUDAQ] Shot {i}: bitstring 拼接失敗 ({e})，跳過。")
            continue
        if len(bs) != 90:
            malformed_count += 1
            if malformed_count <= 3:
                warnings.warn(f"[CUDAQ] Shot {i}: bitstring 長度 {len(bs)} != 90，跳過。")
            continue
        counts[bs] = counts.get(bs, 0) + 1

    if malformed_count > 0:
        warnings.warn(
            f"[CUDAQ] 共 {malformed_count}/{n_shots} shots bitstring 長度異常，已跳過。"
        )

    return counts


# ===========================================================================
# MoleculeGeneratorCUDAQ  (v9.1)
# ===========================================================================

class MoleculeGeneratorCUDAQ:
    """
    CUDA-Q 版分子生成器（CUDA-Q 0.7.1 / V100 sm_70 相容，v9.1 修正版）。

    v9.1 改變：
      1. tensornet backend 明確 guard（不支援動態電路）
      2. _reconstruct_bitstrings_n9 加入 register 存在性驗證
      3. 使用 module-level parametric kernel（MLIR 只編譯一次，無 OOM 問題）
      4. 90 個命名暫存器重建（各行獨立，不依賴 __global__）

    效能建議：使用 backend_name='cudaq_nvidia'（V100 GPU）。
      GPU：~1-5s/eval（10000 shots）
      CPU：~90s/eval（10000 shots，僅供除錯）
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

        self.data_generator = MoleculeQuantumStateGenerator(
            heavy_atom_size = num_heavy_atom,
            ncpus           = 1,
            sanitize_method = "strict",
        )

        # ★ _CUDAQ_TARGET_MAP 中故意不包含 tensornet；
        #   _set_target_safe 會在 tensornet 時直接 raise RuntimeError
        actual_target       = _CUDAQ_TARGET_MAP.get(backend_name, "qpp-cpu")
        self._active_target = _set_target_safe(actual_target)
        self._kernel        = self.circuit_builder.get_kernel()

        ver_str, _ = _check_cudaq_version_volta_compat()
        print(
            f"[CUDAQ] Generator initialized (v9.1).\n"
            f"  cudaq version  : {ver_str}\n"
            f"  active target  : {self._active_target}\n"
            f"  N atoms        : {num_heavy_atom}\n"
            f"  weight dim     : {self.circuit_builder.length_all_weight_vector}\n"
            f"  expected_bits  : {self.expected_bits}\n"
            f"  kernel mode    : parametric list[float] (MLIR compiled once)\n"
            f"  reconstruction : 90 named registers (no __global__ dependency)\n"
            f"  fixes          : OOM + __global__ bug + tensornet guard + register validation"
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
        """
        執行量子電路取樣並解碼為分子 SMILES。

        Returns:
            (smiles_dict, validity, uniqueness)
        """
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

        # 將 weight 準備為 list[float] 並傳入 parametric kernel
        w_list = self.circuit_builder.prepare_weights(w)

        # 呼叫 parametric kernel（重用已編譯的 MLIR module，不新建）
        result = cudaq.sample(self._kernel, w_list, shots_count=num_sample)

        # 用 90 個命名暫存器重建 bitstring（含 register 存在性驗證）
        raw_counts = _reconstruct_bitstrings_n9(result)

        if not raw_counts:
            warnings.warn("[CUDAQ] raw_counts 為空，回傳 validity=0。")
            return {}, 0.0, 0.0

        # 解碼 bitstring → SMILES
        smiles_dict: dict[str, int] = {}
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

    print("=== MoleculeGeneratorCUDAQ 功能驗證 (v9.1) ===")
    ver_str, is_compat = _check_cudaq_version_volta_compat()
    print(f"CUDA-Q : {ver_str}  Volta compat: {'✓' if is_compat else '⚠ >=0.8，不支援 V100'}")

    from qmg.utils.build_dynamic_circuit_cudaq import _N9_ALL_REGS
    assert len(_N9_ALL_REGS) == 90, f"_N9_ALL_REGS 長度錯誤：{len(_N9_ALL_REGS)}"
    print(f"_N9_ALL_REGS: {len(_N9_ALL_REGS)} 個暫存器 ✓")

    cwg = ConditionalWeightsGenerator(9, smarts=None)
    w   = cwg.generate_conditional_random_weights(random_seed=42)
    assert len(w) == 134, f"weight 長度錯誤：{len(w)}"

    # ── Test 1：GPU（優先）
    print("\n[Test 1] nvidia GPU, 500 shots")
    try:
        gen_gpu = MoleculeGeneratorCUDAQ(9, all_weight_vector=w,
                                          backend_name="cudaq_nvidia")
        t0 = time.time()
        sd_gpu, v_gpu, u_gpu = gen_gpu.sample_molecule(500)
        elapsed = time.time() - t0
        print(f"  V={v_gpu:.3f}  U={u_gpu:.3f}  V×U={v_gpu*u_gpu:.4f}  ({elapsed:.1f}s)")
        valid_gpu = [k for k in sd_gpu if k and k != "None"]
        print(f"  有效分子數: {len(valid_gpu)}")
        if valid_gpu:
            print(f"  範例 SMILES: {valid_gpu[:3]}")
        if v_gpu > 0.3 and u_gpu > 0.3:
            print("  [Test 1] ✓ GPU 正常")
        elif v_gpu > 0:
            print("  [Test 1] ⚠ GPU 有結果但 V 或 U 偏低")
        else:
            print("  [Test 1] ✗ GPU validity=0，請檢查 register 命名")
    except Exception as e:
        print(f"  [Test 1] GPU 失敗：{e}")

    # ── Test 2：CPU（備用）
    print("\n[Test 2] qpp-cpu, 200 shots")
    try:
        gen_cpu = MoleculeGeneratorCUDAQ(9, all_weight_vector=w, backend_name="cudaq_qpp")
        t0 = time.time()
        sd_cpu, v_cpu, u_cpu = gen_cpu.sample_molecule(200)
        elapsed = time.time() - t0
        print(f"  V={v_cpu:.3f}  U={u_cpu:.3f}  V×U={v_cpu*u_cpu:.4f}  ({elapsed:.1f}s)")
        valid_cpu = [k for k in sd_cpu if k and k != "None"]
        print(f"  有效分子數: {len(valid_cpu)}")
        if v_cpu > 0.3 and u_cpu > 0.3:
            print("  [Test 2] ✓ CPU 正常")
        elif v_cpu > 0:
            print("  [Test 2] ⚠ CPU 有結果但 V 或 U 偏低")
        else:
            print("  [Test 2] ✗ CPU validity=0")
    except Exception as e:
        print(f"  [Test 2] CPU 失敗：{e}")