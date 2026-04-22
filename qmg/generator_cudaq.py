"""
==============================================================================
generator_cudaq.py  (CUDA-Q 0.7.1 / V100 sm_70 完整修正版 v10.0)
==============================================================================

v9.5 → v10.0 新增功能：

  ★ [NEW] tensornet 後端支援
      SQMG 論文（arXiv:2604.13877v1）實測：
        N=8 時 tensornet(GPU) 比 cuStateVec(CPU) 快 ~2.2×10³ 倍
        N=8 時 tensornet(GPU) 比 cuStateVec(GPU) 快 ~4.5×10⁴/0.167 ≈ 270 倍
      QMG N=9 動態電路（20 qubits，90 mid-circuit measurements）：
        tensornet 以 Tensor-Network Contraction 模擬，不需要完整 2^20 狀態向量
        對有限糾纏深度的動態電路有顯著記憶體和速度優勢

      注意：cudaq tensornet 0.7.1 對動態電路（if bit:）的支援狀態：
        已在 cudaq.sample() 模式下支援 mid-circuit measurement + classical if
        使用前建議先以小 shots 驗證結果正確性

  ★ [NEW] nvidia-mqpu / nvidia-mgpu 後端支援
      nvidia-mqpu：自動將 shots 分配到多張 GPU（需多 GPU 環境）
      nvidia-mgpu ：多 GPU state-vector 模擬（大狀態向量）

  ★ [NEW] _verify_backend_smoke() 強化
      對 tensornet 後端單獨做 smoke test，確保 mid-circuit 可用

  v9.5 修正保留：
    - _reconstruct_bitstrings_n9 型別修正（int(bit) 取代 1 if bit else 0）
    - del result + gc.collect + malloc_trim 防止 OOM
    - bytearray buffer 低記憶體模式

  v9.1 修正保留：
    - _qmg_n9 分號修正版（build_dynamic_circuit_cudaq.py）

放置位置：qmg/generator_cudaq.py
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
    """強制 glibc 將 C++ heap 釋放記憶體歸還 OS（非 glibc 環境靜默忽略）。"""
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
def _smoke_kernel_v10():
    q = cudaq.qvector(1)
    h(q[0])
    mz(q[0])


@cudaq.kernel
def _smoke_midcircuit_v10():
    """tensornet 用：驗證 mid-circuit measurement + classical if 可用。"""
    q = cudaq.qvector(4)
    h(q[0])
    b0 = mz(q[0])
    if b0:
        x(q[1])
    mz(q[1])
    mz(q[2])
    mz(q[3])


# ===========================================================================
# 後端對應表（v10.0 新增 tensornet / mqpu / mgpu）
# ===========================================================================

_CUDAQ_TARGET_MAP = {
    # CPU
    "cudaq_qpp":              "qpp-cpu",
    "qpp-cpu":                "qpp-cpu",
    # GPU state-vector（cuStateVec，V100 sm_70 最穩定）
    "cudaq_nvidia":           "nvidia",
    "nvidia":                 "nvidia",
    "cudaq_nvidia_fp64":      "nvidia-fp64",
    "nvidia-fp64":            "nvidia-fp64",
    # ★ v10.0 新增：TensorNet GPU（SQMG 論文推薦，記憶體高效，速度最快）
    "cudaq_tensornet":        "tensornet",
    "tensornet":              "tensornet",
    "cudaq_tensornet_mps":    "tensornet-mps",
    "tensornet-mps":          "tensornet-mps",
    # ★ v10.0 新增：Multi-GPU（MPI 架構下通常每 rank 使用單 GPU，此處備用）
    "cudaq_nvidia_mqpu":      "nvidia-mqpu",
    "nvidia-mqpu":            "nvidia-mqpu",
    "cudaq_nvidia_mgpu":      "nvidia-mgpu",
    "nvidia-mgpu":            "nvidia-mgpu",
    # 舊版相容
    "qiskit_aer":             "qpp-cpu",
}

# GPU 後端集合（用於 Volta 相容性檢查）
_GPU_TARGETS = {
    "nvidia", "nvidia-fp64",
    "tensornet", "tensornet-mps",          # v10.0 新增
    "nvidia-mqpu", "nvidia-mgpu",           # v10.0 新增
    "nvidia-mqpu-fp64", "nvidia-mqpu-mps",
}


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
            if 'nvidia' in str(t.name).lower():
                return True
        return False
    except Exception:
        return False


def _tensornet_midcircuit_ok() -> bool:
    """驗證 tensornet 後端是否支援 mid-circuit measurement。"""
    try:
        cudaq.set_target("tensornet")
        result = cudaq.sample(_smoke_midcircuit_v10, shots_count=16)
        ok = len(dict(result.items())) > 0
        del result
        gc.collect()
        return ok
    except Exception as e:
        warnings.warn(f"[CUDAQ] tensornet mid-circuit smoke test 失敗：{e}")
        return False


def _verify_gpu_smoke(target_name: str) -> bool:
    try:
        if target_name == "tensornet":
            return _tensornet_midcircuit_ok()
        result = cudaq.sample(_smoke_kernel_v10, shots_count=16)
        ok = len(dict(result.items())) > 0
        del result
        gc.collect()
        _free_cpp_heap()
        return ok
    except Exception as e:
        warnings.warn(f"[CUDAQ] {target_name} smoke test 失敗：{e}")
        return False


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
        if _verify_gpu_smoke(target_name):
            print(f"[CUDAQ] Target '{target_name}' 驗證通過 ✓")
        else:
            warnings.warn(f"[CUDAQ] {target_name} smoke test 異常，請確認環境。")
    return target_name


# ===========================================================================
# 90-bit bitstring 重建（v9.5 型別修正版，v10.0 無變動）
# ===========================================================================

def _reconstruct_bitstrings_n9(result) -> dict[str, int]:
    """
    用 90 個命名暫存器重建 bitstring。
    v9.5 關鍵修正：int(bit) 取代 1 if bit else 0，正確處理 str '0'/'1'。
    """
    try:
        first_data = result.get_sequential_data(_N9_ALL_REGS[0])
        n_shots = len(first_data)
        if n_shots == 0:
            warnings.warn("[CUDAQ] n_shots=0，get_sequential_data 回傳空 list。")
            return {}

        buf = bytearray(n_shots * 90)

        for i, bit in enumerate(first_data):
            buf[i * 90] = int(bit)
        del first_data

        for reg_idx, reg in enumerate(_N9_ALL_REGS[1:], start=1):
            reg_data = result.get_sequential_data(reg)
            for i, bit in enumerate(reg_data):
                buf[i * 90 + reg_idx] = int(bit)
            del reg_data

        counts: dict[str, int] = {}
        malformed = 0
        for i in range(n_shots):
            row = buf[i * 90: i * 90 + 90]
            if len(row) != 90:
                malformed += 1
                continue
            bs = ''.join('1' if b else '0' for b in row)
            counts[bs] = counts.get(bs, 0) + 1

        del buf

        if malformed:
            warnings.warn(f"[CUDAQ] {malformed}/{n_shots} shots bitstring 長度異常。")

        if counts:
            sample_bs = next(iter(counts))
            ones_ratio = sample_bs.count('1') / 90
            if ones_ratio > 0.95:
                warnings.warn(
                    f"[CUDAQ] 警告：bitstring 中 '1' 比例 = {ones_ratio:.2f}（過高）"
                )
            elif ones_ratio < 0.01:
                warnings.warn(
                    f"[CUDAQ] 警告：bitstring 中 '1' 比例 = {ones_ratio:.2f}（過低）"
                )

        return counts

    except AttributeError:
        warnings.warn(
            "[CUDAQ] get_sequential_data() 不存在。\n"
            "  確認使用 CUDA-Q 0.7.1。"
        )
        return {}
    except Exception as e:
        warnings.warn(f"[CUDAQ] _reconstruct_bitstrings_n9 失敗：{e}")
        import traceback
        traceback.print_exc()
        return {}


# ===========================================================================
# MoleculeGeneratorCUDAQ  (v10.0)
# ===========================================================================

class MoleculeGeneratorCUDAQ:
    """CUDA-Q 版分子生成器（v10.0）。

    v10.0：新增 tensornet / nvidia-mqpu / nvidia-mgpu 後端支援。
    v9.5：修正 _reconstruct_bitstrings_n9 型別判斷 bug。
    v9.4：修正 OOM Kill（del result + gc.collect + malloc_trim）。
    v9.3：修正 list[float] broadcast dispatch 錯誤（分號 AST bug）。

    tensornet 後端說明（SQMG 論文）：
      以 cuTensorNet 做 Tensor-Network Contraction 模擬，
      避免顯式儲存 2^N 狀態向量。N=8 時比 cuStateVec GPU 快 ~270 倍。
      適合大 N（N≥8）或記憶體受限情境。
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
            raise NotImplementedError(f"目前僅支援 num_heavy_atom=9（N={num_heavy_atom}）。")

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
            f"[CUDAQ] Generator initialized (v10.0).\n"
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

        # ★ 釋放 CUDA/TN 資源
        del result
        del w_list
        gc.collect()
        _free_cpp_heap()

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

        del raw_counts
        gc.collect()

        validity   = num_valid / num_sample
        n_unique   = len([k for k in smiles_dict if k and k != "None"])
        uniqueness = n_unique / num_valid if num_valid > 0 else 0.0
        return smiles_dict, validity, uniqueness


MoleculeGenerator = MoleculeGeneratorCUDAQ


# ===========================================================================
# 快速功能驗證（含 tensornet）
# ===========================================================================
if __name__ == "__main__":
    import time
    print("=== MoleculeGeneratorCUDAQ 功能驗證 (v10.0) ===")
    ver_str, is_compat = _check_cudaq_version_volta_compat()
    print(f"CUDA-Q: {ver_str}  V100 compat: {'✓' if is_compat else '⚠'}")
    print(f"GPU target 可用: {_gpu_target_available()}")

    cwg = ConditionalWeightsGenerator(9, smarts=None)
    w   = cwg.generate_conditional_random_weights(random_seed=42)

    test_cases = [
        ("cudaq_tensornet", 200, "TensorNet GPU"),
        ("cudaq_nvidia",    200, "cuStateVec GPU"),
        ("cudaq_qpp",       100, "CPU"),
    ]

    for backend, shots, lbl in test_cases:
        print(f"\n[{lbl}] {backend}, {shots} shots")
        try:
            gen = MoleculeGeneratorCUDAQ(9, all_weight_vector=w, backend_name=backend)
            t0  = time.time()
            sd, v, u = gen.sample_molecule(shots)
            elapsed  = time.time() - t0
            print(f"  V={v:.3f}  U={u:.3f}  V×U={v*u:.4f}  ({elapsed:.1f}s)")
            valid = [k for k in sd if k and k != "None"]
            print(f"  有效分子：{len(valid)} 種")
            if v > 0:
                print(f"  ✓ 正常")
            else:
                print(f"  ✗ validity=0，請檢查 backend")
        except Exception as e:
            print(f"  ✗ 失敗：{e}")