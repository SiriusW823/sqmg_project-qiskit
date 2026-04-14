"""
==============================================================================
generator_cudaq.py  (CUDA-Q 0.7.1 / V100 sm_70 完整修正版 v9)
==============================================================================

v8 → v9 修正（根本原因分析）：

  ★ CUDA-Q 0.7.1 已知 Bug：
      CUDA-Q 0.7.1 對含 mid-circuit measurement 的 kernel，
      get_sequential_data('__global__') 行為異常。
      （CUDA-Q GitHub release 0.8.0 明確標注修正：
       "Fixes an issue with sampling when kernel contains mid-circuit measurements"）

      v8 的 _reconstruct_bitstrings_n9() 依賴 __global__ 取 bond-9 的 16 bits：
        global_data = result.get_sequential_data('__global__')
      實際執行時此呼叫拋出異常，被 `except Exception: global_data = None` 吃掉，
      導致 bond9_bits = '0' * 16 永遠為零
      → 所有 atom-9 鍵型資訊完全遺失
      → uniqueness 長期壓在 0.05~0.35（與 log 觀察完全吻合）

  ★ OOM Killed at eval ~155：
      v8 的 make_qmg_n9_kernel() 每次呼叫建立新 @cudaq.kernel，
      MLIR module 累積在 C++ registry 中，
      del kernel + gc.collect() 無法釋放 C++ 層記憶體
      → 約 155 次評估後 OOM Killed

  v9 解法：
    (1) build_dynamic_circuit_cudaq.py 改用 module-level parametric kernel
        _qmg_n9(w: list[float])，@cudaq.kernel 裝飾器在 import 時編譯一次
        MLIR，每次 cudaq.sample(_qmg_n9, w_list, ...) 重用 → 無記憶體累積
    (2) 所有 90 個 mz() 均有命名（b91_0...b98_1 新增）
    (3) _reconstruct_bitstrings_n9() 使用 _N9_ALL_REGS（90 個命名暫存器）
        完全不依賴 __global__ → 規避 CUDA-Q 0.7.1 bug

速度對比（10000 shots，N=9）：
  qpp-cpu  : ~90s/eval，Python shot-by-shot loop
  nvidia   : ~1-5s/eval，cuStateVec 原生處理（GPU 記憶體內完成）
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
# 90-bit bitstring 重建（全部使用命名暫存器）
# ===========================================================================

# ★ v9：90 個命名暫存器（74 原有 + 16 bond-9 新增）
# 不再依賴 __global__，徹底規避 CUDA-Q 0.7.1 bug
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
    # ── atom 8 ─────────────────────────────────────── bits 56-73
    'a8_0', 'a8_1',
    'b81_0', 'b81_1', 'b82_0', 'b82_1', 'b83_0', 'b83_1',
    'b84_0', 'b84_1', 'b85_0', 'b85_1', 'b86_0', 'b86_1',
    'b87_0', 'b87_1',
    # ── atom 9（atom type）──────────────────────────── bits 72-73
    'a9_0', 'a9_1',
    # ★ v9 新增：bond-9（bonds to atoms 1-8）──────── bits 74-89
    'b91_0', 'b91_1',
    'b92_0', 'b92_1',
    'b93_0', 'b93_1',
    'b94_0', 'b94_1',
    'b95_0', 'b95_1',
    'b96_0', 'b96_1',
    'b97_0', 'b97_1',
    'b98_0', 'b98_1',
]  # 90 個暫存器，對應 bitstring bits 0-89


def _reconstruct_bitstrings_n9(result) -> dict[str, int]:
    """
    90-bit bitstring 重建（v9：全部使用命名暫存器，不依賴 __global__）。

    CUDA-Q 0.7.1 bug 說明：
      get_sequential_data('__global__') 在含 mid-circuit measurement 的
      kernel 中行為異常（0.8.0 才修正）。v9 完全不呼叫 __global__，
      改用 get_sequential_data(reg_name) 存取 90 個命名暫存器。

    Returns:
        dict mapping 90-bit bitstring → count
    """
    # 嘗試用 get_sequential_data 取得所有 90 個命名暫存器
    try:
        reg_data = {reg: result.get_sequential_data(reg) for reg in _N9_ALL_REGS}
    except AttributeError:
        # API 不存在（極舊版本），fallback 到 items()
        warnings.warn(
            "[CUDAQ] get_sequential_data() 不存在，使用 items() fallback。"
            "此 fallback 僅適用於 bitstring 長度恰好為 90 的情況。"
        )
        counts: dict[str, int] = {}
        for bs_raw, cnt in result.items():
            if len(bs_raw) == 90:
                counts[bs_raw] = counts.get(bs_raw, 0) + cnt
        if not counts:
            warnings.warn(
                "[CUDAQ] items() fallback 未找到 90-bit bitstring。"
                "請確認 CUDA-Q 版本 >= 0.7.1 且 target 支援 mid-circuit measurement。"
            )
        return counts
    except Exception as e:
        # 其他例外（如命名暫存器不存在）
        warnings.warn(
            f"[CUDAQ] get_sequential_data() 失敗：{e}\n"
            f"  可能原因：CUDA-Q 0.7.1 kernel 命名暫存器問題。"
            f"  請確認 kernel 中 mz() 的變數名稱與 _N9_ALL_REGS 一致。"
        )
        return {}

    # 取 shot 數量（任意一個暫存器的資料長度）
    n_shots = len(reg_data.get('a1_0', []))
    if n_shots == 0:
        warnings.warn("[CUDAQ] get_sequential_data 回傳空列表，n_shots=0。")
        return {}

    # 逐 shot 拼接 90-bit bitstring
    counts: dict[str, int] = {}
    malformed_count = 0
    for i in range(n_shots):
        bs = ''.join(reg_data[reg][i] for reg in _N9_ALL_REGS)
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
# MoleculeGeneratorCUDAQ  (v9)
# ===========================================================================

class MoleculeGeneratorCUDAQ:
    """
    CUDA-Q 版分子生成器（CUDA-Q 0.7.1 / V100 sm_70 相容，v9 根本修正版）。

    v9 主要改變：
      1. 使用 module-level parametric kernel（MLIR 只編譯一次，無 OOM 問題）
      2. 90 個命名暫存器重建（不依賴 __global__，規避 CUDA-Q 0.7.1 bug）
      3. sample_molecule() 無需 del kernel + gc.collect()

    效能建議：使用 backend_name='cudaq_nvidia'（GPU）。
      GPU：supportsConditionalFeedback=True → 1~5s/eval（10000 shots）
      CPU：supportsConditionalFeedback=False → ~90s/eval（10000 shots）
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
        # ★ v9：取得 module-level parametric kernel（已編譯好）
        self._kernel        = self.circuit_builder.get_kernel()

        ver_str, _ = _check_cudaq_version_volta_compat()
        print(
            f"[CUDAQ] Generator initialized (v9).\n"
            f"  cudaq version  : {ver_str}\n"
            f"  active target  : {self._active_target}\n"
            f"  N atoms        : {num_heavy_atom}\n"
            f"  weight dim     : {self.circuit_builder.length_all_weight_vector}\n"
            f"  expected_bits  : {self.expected_bits}\n"
            f"  kernel mode    : parametric list[float] (MLIR compiled once)\n"
            f"  reconstruction : 90 named registers (no __global__ dependency)\n"
            f"  fixes          : OOM + CUDA-Q 0.7.1 __global__ bug"
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

        v9 改變：
          - 使用 parametric kernel，不建立新 MLIR module
          - 不需要 del kernel + gc.collect()
          - 使用 90 個命名暫存器重建 bitstring

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

        # ★ v9：將 weight 準備為 list[float] 並傳入 parametric kernel
        w_list = self.circuit_builder.prepare_weights(w)

        # 呼叫 parametric kernel（重用已編譯的 MLIR module，不新建）
        result = cudaq.sample(self._kernel, w_list, shots_count=num_sample)

        # ★ v9：用 90 個命名暫存器重建 bitstring（不依賴 __global__）
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

    print("=== MoleculeGeneratorCUDAQ 功能驗證 (v9) ===")
    ver_str, is_compat = _check_cudaq_version_volta_compat()
    print(f"CUDA-Q : {ver_str}  Volta compat: {'✓' if is_compat else '⚠ >=0.8，不支援 V100'}")

    # 驗證 _N9_ALL_REGS 長度
    from qmg.utils.build_dynamic_circuit_cudaq import _N9_ALL_REGS
    assert len(_N9_ALL_REGS) == 90, f"_N9_ALL_REGS 長度錯誤：{len(_N9_ALL_REGS)}"
    print(f"_N9_ALL_REGS: {len(_N9_ALL_REGS)} 個暫存器 ✓")

    cwg = ConditionalWeightsGenerator(9, smarts=None)
    w   = cwg.generate_conditional_random_weights(random_seed=42)
    assert len(w) == 134, f"weight 長度錯誤：{len(w)}"

    # ── Test 1：GPU（優先）──────────────────────────────────────────────────
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
        print(f"  範例 SMILES: {valid_gpu[:5]}")
        if v_gpu > 0.3 and u_gpu > 0.3:
            print("  [Test 1] ✓ GPU 正常，V 和 U 均 > 0.3（v9 修正後預期明顯改善）")
        elif v_gpu > 0:
            print("  [Test 1] ⚠ GPU 有結果但 V 或 U 偏低，可能 list[float] kernel 需調整")
        else:
            print("  [Test 1] ✗ GPU validity=0，改用 CPU 繼續")
    except Exception as e:
        print(f"  [Test 1] GPU 失敗：{e}")
        print("  嘗試 CPU...")

    # ── Test 2：CPU（備用）──────────────────────────────────────────────────
    print("\n[Test 2] qpp-cpu, 200 shots")
    try:
        gen_cpu = MoleculeGeneratorCUDAQ(9, all_weight_vector=w, backend_name="cudaq_qpp")
        t0 = time.time()
        sd_cpu, v_cpu, u_cpu = gen_cpu.sample_molecule(200)
        elapsed = time.time() - t0
        print(f"  V={v_cpu:.3f}  U={u_cpu:.3f}  V×U={v_cpu*u_cpu:.4f}  ({elapsed:.1f}s)")
        valid_cpu = [k for k in sd_cpu if k and k != "None"]
        print(f"  有效分子數: {len(valid_cpu)}")
        print(f"  範例 SMILES: {valid_cpu[:5]}")
        if v_cpu > 0.3 and u_cpu > 0.3:
            print("  [Test 2] ✓ CPU 正常")
            proj_time = elapsed / 200 * 10000 * 520 / 3600
            print(f"  預計完整實驗耗時：~{proj_time:.0f}h（強烈建議改用 GPU）")
        elif v_cpu > 0:
            print("  [Test 2] ⚠ CPU 有結果但 V 或 U 偏低")
        else:
            print("  [Test 2] ✗ CPU validity=0")
    except Exception as e:
        print(f"  [Test 2] CPU 失敗：{e}")

    print("\n=== v9 部署後正式實驗指令 ===")
    print("mkdir -p results_paper_comparison")
    print("tmux new-session -s qmg_paper")
    print("python run_qpso_qmg_cudaq.py \\")
    print("    --backend cudaq_nvidia --num_heavy_atom 9 \\")
    print("    --num_sample 10000 --particles 20 --iterations 25 \\")
    print("    --alpha_max 1.2 --alpha_min 0.4 --mutation_prob 0.10 \\")
    print("    --stagnation_limit 12 --seed 42 \\")
    print("    --task_name unconditional_9_qpso_paper \\")
    print("    --data_dir results_paper_comparison")