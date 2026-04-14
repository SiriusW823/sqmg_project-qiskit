"""
==============================================================================
generator_cudaq.py  (CUDA-Q 0.7.1 / V100 sm_70 完整修正版 v8.1)
==============================================================================

v8.1 修正（相對於 v8）：
  ★ BUG FIX：_reconstruct_bitstrings_n9() 中
      bond9_bits = global_data[i][4:20]   ← 錯誤：跳過前4位，導致atom-9所有
                                            bond type 資訊系統性錯位
      →
      bond9_bits = global_data[i][0:16]   ← 正確：16個無名mz()從index 0開始

    影響：修正前 atom-9 的 bond type 資訊全部偏移，造成 uniqueness 長期壓在
    0.05~0.35，validity 亦受影響。修正後應接近論文基線 V×U ≈ 0.8834。

v8 設計：
  1. 記憶體管理：sample_molecule() 完成後 del kernel + gc.collect()
  2. 效能：GPU (nvidia) supportsConditionalFeedback=True
             → cuStateVec 一次完成所有 shots，預計 1~5s/eval

速度對比（10000 shots，N=9）：
  qpp-cpu  : ~90s/eval，Python shot-by-shot loop
  nvidia   : ~1-5s/eval，cuStateVec 原生處理（GPU 記憶體內完成）
==============================================================================
"""
from __future__ import annotations

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
    "tensornet":         "tensornet",
    "qiskit_aer":        "qpp-cpu",
}
_GPU_TARGETS = {"nvidia", "nvidia-fp64"}


def _set_target_safe(target_name: str) -> str:
    ver_str, is_compat = _check_cudaq_version_volta_compat()
    if target_name in _GPU_TARGETS and not is_compat:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"[CUDAQ] CUDA-Q {ver_str} 不支援 V100 (sm_70)。\n"
            f"  請安裝：pip install cuda-quantum-cu11==0.7.1\n"
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
# 90-bit bitstring 重建（具名暫存器策略）
# ===========================================================================

_N9_NAMED_REGS: list[str] = [
    'a1_0', 'a1_1',
    'a2_0', 'a2_1',
    'b21_0', 'b21_1',
    'a3_0', 'a3_1',
    'b31_0', 'b31_1', 'b32_0', 'b32_1',
    'a4_0', 'a4_1',
    'b41_0', 'b41_1', 'b42_0', 'b42_1', 'b43_0', 'b43_1',
    'a5_0', 'a5_1',
    'b51_0', 'b51_1', 'b52_0', 'b52_1',
    'b53_0', 'b53_1', 'b54_0', 'b54_1',
    'a6_0', 'a6_1',
    'b61_0', 'b61_1', 'b62_0', 'b62_1', 'b63_0', 'b63_1',
    'b64_0', 'b64_1', 'b65_0', 'b65_1',
    'a7_0', 'a7_1',
    'b71_0', 'b71_1', 'b72_0', 'b72_1', 'b73_0', 'b73_1',
    'b74_0', 'b74_1', 'b75_0', 'b75_1', 'b76_0', 'b76_1',
    'a8_0', 'a8_1',
    'b81_0', 'b81_1', 'b82_0', 'b82_1', 'b83_0', 'b83_1',
    'b84_0', 'b84_1', 'b85_0', 'b85_1', 'b86_0', 'b86_1',
    'b87_0', 'b87_1',
    'a9_0', 'a9_1',
]  # 74 個具名暫存器


def _reconstruct_bitstrings_n9(result) -> dict[str, int]:
    """
    90-bit bitstring 重建：
      bits[ 0:74] — 74 個具名 mz() → get_sequential_data(reg)
      bits[74:90] — 16 個無名 mz()（鍵 9-{1..8}）→ __global__[0:16]

    ★ v8.1 修正：
      __global__ 暫存器中 atom-9 的 bond bits 從 index 0 開始，
      原本的 [4:20] 是錯誤的（跳過了 b91_0, b91_1, b92_0, b92_1 四個位元），
      導致整個 atom-9 bond type 資訊系統性錯位，uniqueness 長期偏低。
      正確索引：global_data[i][0:16]
    """
    try:
        reg_data = {reg: result.get_sequential_data(reg) for reg in _N9_NAMED_REGS}
    except AttributeError:
        warnings.warn("[CUDAQ] get_sequential_data() 不存在，使用 items() fallback。")
        counts: dict[str, int] = {}
        for bs_raw, cnt in result.items():
            if len(bs_raw) == 90:
                counts[bs_raw] = counts.get(bs_raw, 0) + cnt
        return counts

    n_shots = len(reg_data['a1_0'])
    if n_shots == 0:
        warnings.warn("[CUDAQ] get_sequential_data 回傳空列表，n_shots=0。")
        return {}

    try:
        global_data = result.get_sequential_data('__global__')
    except Exception:
        global_data = None

    counts: dict[str, int] = {}
    warned_global = False
    for i in range(n_shots):
        named_bits = ''.join(reg_data[reg][i] for reg in _N9_NAMED_REGS)

        # ★ v8.1 BUG FIX：改為 [0:16]，atom-9 的 16 個 bond bits 從 index 0 開始
        # 原始錯誤：global_data[i][4:20] 會跳過前 4 位，導致 b91、b92 位元錯位
        if global_data and len(global_data) > i and len(global_data[i]) >= 16:
            bond9_bits = global_data[i][0:16]
        else:
            bond9_bits = '0' * 16
            if not warned_global:
                warnings.warn(
                    "[CUDAQ] __global__ 不可用，鍵 9-{1..8} 補零。"
                    "bond_disconnection_correction 確保原子9至少有一鍵。"
                )
                warned_global = True

        bs = named_bits + bond9_bits
        if len(bs) != 90:
            raise RuntimeError(f"Shot {i}: bitstring 長度 {len(bs)} != 90")
        counts[bs] = counts.get(bs, 0) + 1

    return counts


# ===========================================================================
# MoleculeGeneratorCUDAQ
# ===========================================================================

class MoleculeGeneratorCUDAQ:
    """
    CUDA-Q 版分子生成器（CUDA-Q 0.7.1 / V100 sm_70 相容）。

    效能建議：使用 backend_name='cudaq_nvidia'（GPU）。
      GPU：supportsConditionalFeedback=True → 1~5s/eval（10000 shots）
      CPU：supportsConditionalFeedback=False → ~95s/eval（10000 shots）
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

        actual_target       = _CUDAQ_TARGET_MAP.get(backend_name, "qpp-cpu")
        self._active_target = _set_target_safe(actual_target)

        ver_str, _ = _check_cudaq_version_volta_compat()
        print(
            f"[CUDAQ] Generator initialized.\n"
            f"  cudaq version : {ver_str}\n"
            f"  active target : {self._active_target}\n"
            f"  N atoms       : {num_heavy_atom}\n"
            f"  weight dim    : {self.circuit_builder.length_all_weight_vector}\n"
            f"  expected_bits : {self.expected_bits}\n"
            f"  design        : closure/capture + gc.collect() memory management\n"
            f"  bond9 fix     : global_data[i][0:16] (v8.1 bugfix)"
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

        # ★ 建立 closure kernel（weights bake 進去，無參數）
        kernel = self.circuit_builder.build_kernel_from_weights(w)

        try:
            # cudaq.sample(kernel) 無額外參數 → __isBroadcast=False
            # → 正確的 conditionalOnMeasure 路徑
            result = cudaq.sample(kernel, shots_count=num_sample)
        finally:
            # ★ 立即釋放 MLIR module，避免 OOM
            del kernel
            gc.collect()

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

    print("=== MoleculeGeneratorCUDAQ 功能驗證 (v8.1) ===")
    ver_str, is_compat = _check_cudaq_version_volta_compat()
    print(f"CUDA-Q : {ver_str}  Volta compat: {'✓' if is_compat else '⚠ >=0.8'}")

    cwg = ConditionalWeightsGenerator(9, smarts=None)
    w   = cwg.generate_conditional_random_weights(random_seed=42)

    # ── Test 1：GPU（優先）──────────────────────────────────────────────────
    print("\n[Test 1] nvidia GPU, 500 shots（建議使用 GPU 跑正式實驗）")
    try:
        gen_gpu = MoleculeGeneratorCUDAQ(9, all_weight_vector=w,
                                         backend_name="cudaq_nvidia")
        t0 = time.time()
        sd_gpu, v_gpu, u_gpu = gen_gpu.sample_molecule(500)
        elapsed = time.time() - t0
        print(f"  V={v_gpu:.3f}  U={u_gpu:.3f}  V×U={v_gpu*u_gpu:.4f}  ({elapsed:.1f}s)")
        valid_gpu = [k for k in sd_gpu if k and k != "None"]
        print(f"  有效分子數: {len(valid_gpu)}")
        print(f"  範例 SMILES: {valid_gpu[:5]}")
        if v_gpu > 0:
            print("  [Test 1] ✓ GPU 正常，預計正式實驗 ~1-5s/eval")
        else:
            print("  [Test 1] ✗ GPU validity=0，改用 CPU 繼續")
    except Exception as e:
        print(f"  [Test 1] GPU 失敗：{e}")

    # ── Test 2：CPU（備用）──────────────────────────────────────────────────
    print("\n[Test 2] qpp-cpu, 200 shots（備用，~95s/eval 較慢）")
    gen_cpu = MoleculeGeneratorCUDAQ(9, all_weight_vector=w, backend_name="cudaq_qpp")
    t0 = time.time()
    sd_cpu, v_cpu, u_cpu = gen_cpu.sample_molecule(200)
    elapsed = time.time() - t0
    print(f"  V={v_cpu:.3f}  U={u_cpu:.3f}  V×U={v_cpu*u_cpu:.4f}  ({elapsed:.1f}s)")
    valid_cpu = [k for k in sd_cpu if k and k != "None"]
    print(f"  有效分子數: {len(valid_cpu)}")
    print(f"  範例 SMILES: {valid_cpu[:5]}")
    if v_cpu > 0:
        print("  [Test 2] ✓ CPU 正常")
        proj_time = elapsed / 200 * 10000 * 520 / 3600
        print(f"  [Test 2] 預計完整實驗耗時：~{proj_time:.0f}h（建議改用 GPU）")
    else:
        print("  [Test 2] ✗ CPU validity=0")

    print("\n=== 部署後正式實驗指令 ===")
    print("tmux new-session -s qmg_paper")
    print("# 在 tmux 內：")
    print("python run_qpso_qmg_cudaq.py \\")
    print("    --backend cudaq_nvidia --num_heavy_atom 9 \\")
    print("    --num_sample 10000 --particles 20 --iterations 25 \\")
    print("    --alpha_max 1.2 --alpha_min 0.4 --mutation_prob 0.10 \\")
    print("    --stagnation_limit 12 --seed 42 \\")
    print("    --task_name unconditional_9_qpso_paper \\")
    print("    --data_dir results_paper_comparison")