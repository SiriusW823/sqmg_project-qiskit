"""
==============================================================================
generator_cudaq.py
CUDA-Q 版本的 MoleculeGenerator（對應 Qiskit generator.py）
==============================================================================

Qiskit → CUDA-Q 對應關係
─────────────────────────────────────────────────────────────────────────────
  AerSimulator + SamplerV2          →  cudaq.set_target() + cudaq.sample()
  job = sampler.run([transpiled_qc]) →  result = cudaq.sample(kernel, weights,
  results[0].data.c.get_bitstrings()      shots_count=num_sample)
                                          result.get_bitstrings()
  DynamicCircuitBuilder             →  DynamicCircuitBuilderCUDAQ
  post_process_quantum_state(rev=True) → post_process_quantum_state(rev=False)

Backend 對應
─────────────────────────────────────────────────────────────────────────────
  "cudaq_qpp"        → CPU 模擬（對應 qiskit_aer AerSimulator）
  "cudaq_custatevec" → NVIDIA GPU（cuStateVec，需 CUDA）
  "cudaq_nvidia"     → NVIDIA GPU（別名）
  "cudaq_mqpu"       → 多 GPU 並行

使用方式（介面與 Qiskit 版完全相同）：
  from generator_cudaq import MoleculeGeneratorCUDAQ as MoleculeGenerator
  mg = MoleculeGenerator(9, all_weight_vector=w, backend_name='cudaq_qpp')
  smiles_dict, validity, uniqueness = mg.sample_molecule(10000)
==============================================================================
"""
from __future__ import annotations

import numpy as np
from collections import Counter
from typing import List, Union

import cudaq

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from chemistry_data_processing import MoleculeQuantumStateGenerator
from weight_generator import ConditionalWeightsGenerator
from build_dynamic_circuit_cudaq import DynamicCircuitBuilderCUDAQ


# ===========================================================================
# Backend 映射
# ===========================================================================
_CUDAQ_TARGET_MAP = {
    "cudaq_qpp":        "qpp-cpu",    # 本地 CPU（對應 AerSimulator）
    "cudaq_custatevec": "nvidia",     # NVIDIA GPU（cuStateVec）
    "cudaq_nvidia":     "nvidia",     # 同上（別名）
    "cudaq_mqpu":       "nvidia-mqpu",# 多 GPU
}


# ===========================================================================
# MoleculeGeneratorCUDAQ
# ===========================================================================

class MoleculeGeneratorCUDAQ:
    """
    CUDA-Q 版分子生成器。

    與 Qiskit MoleculeGenerator 完全相同的公開介面：
      __init__(num_heavy_atom, all_weight_vector, backend_name, ...)
      update_weight_vector(w)
      sample_molecule(num_sample) → (smiles_dict, validity, uniqueness)

    主要差異：
      - cudaq.sample() 取代 Qiskit SamplerV2（不需 transpile 步驟）
      - bitstring post-process 改 reverse=False（CUDA-Q 依 mz() 順序輸出）
    """

    def __init__(
        self,
        num_heavy_atom:            int,
        all_weight_vector:         Union[List[float], np.ndarray] = None,
        backend_name:              str   = "cudaq_qpp",
        temperature:               float = 0.2,
        dynamic_circuit:           bool  = True,
        remove_bond_disconnection: bool  = True,
        chemistry_constraint:      bool  = True,
    ):
        if not dynamic_circuit:
            raise NotImplementedError(
                "CUDA-Q 版本目前僅支援 dynamic_circuit=True。"
            )
        self.num_heavy_atom            = num_heavy_atom
        self.all_weight_vector         = (
            np.array(all_weight_vector, dtype=np.float64)
            if all_weight_vector is not None else None
        )
        self.backend_name              = backend_name
        self.temperature               = temperature
        self.dynamic_circuit           = dynamic_circuit
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint      = chemistry_constraint

        self.circuit_builder = DynamicCircuitBuilderCUDAQ(
            num_heavy_atom            = num_heavy_atom,
            temperature               = temperature,
            remove_bond_disconnection = remove_bond_disconnection,
            chemistry_constraint      = chemistry_constraint,
        )
        self.kernel = self.circuit_builder.get_kernel()

        self.data_generator = MoleculeQuantumStateGenerator(
            heavy_atom_size=num_heavy_atom, ncpus=1, sanitize_method="strict"
        )
        self._setup_target()

    def _setup_target(self):
        """設定 CUDA-Q 模擬 target（對應 Qiskit _build_backend）。"""
        target_name = _CUDAQ_TARGET_MAP.get(self.backend_name)
        if target_name is None:
            raise ValueError(
                f"未知 backend '{self.backend_name}'。"
                f"可用：{list(_CUDAQ_TARGET_MAP)}"
            )
        try:
            cudaq.set_target(target_name)
        except Exception as e:
            import warnings
            warnings.warn(f"無法設定 '{target_name}'（{e}），降級使用 qpp-cpu。")
            cudaq.set_target("qpp-cpu")

    def update_weight_vector(self, all_weight_vector: Union[List[float], np.ndarray]):
        """更新電路權重（對應 Qiskit update_weight_vector）。"""
        self.all_weight_vector = np.array(all_weight_vector, dtype=np.float64)

    def sample_molecule(
        self,
        num_sample:  int,
        random_seed: int = 0,
    ):
        """
        執行量子電路取樣，計算 validity 與 uniqueness。

        對應 Qiskit MoleculeGenerator.sample_molecule()。

        Returns:
            (smiles_dict, validity, uniqueness)
        """
        assert self.all_weight_vector is not None, \
            "all_weight_vector 尚未設定，請先傳入或呼叫 update_weight_vector()。"

        w = self.all_weight_vector
        assert len(w) == self.circuit_builder.length_all_weight_vector

        # 設定隨機種子（對應 Qiskit CircuitBuilder 的 random.seed(random_seed)）
        cudaq.set_random_seed(random_seed)

        # ── 取樣（對應 Qiskit Sampler.run() + job.result()）────────────
        result = cudaq.sample(
            self.kernel,
            w.tolist(),          # @cudaq.kernel 要求 list[float]
            shots_count=num_sample,
        )

        # ── Bitstring 處理 ─────────────────────────────────────────────
        # CUDA-Q get_bitstrings() 依 mz() 呼叫順序 → reverse=False
        # Qiskit  get_bitstrings() big-endian         → reverse=True
        raw_bitstrings    = result.get_bitstrings()
        smiles_dict       = {}
        num_valid_molecule = 0

        for bs in raw_bitstrings:
            bs_corrected  = self.circuit_builder.apply_bond_disconnection_correction(bs)
            quantum_state = self.data_generator.post_process_quantum_state(
                bs_corrected, reverse=False   # ← CUDA-Q 用 False
            )
            smiles = self.data_generator.QuantumStateToSmiles(quantum_state)
            smiles_dict[smiles] = smiles_dict.get(smiles, 0) + 1
            if smiles:
                num_valid_molecule += 1

        validity   = num_valid_molecule / num_sample
        uniqueness = (
            (len(smiles_dict) - 1) / num_valid_molecule
            if num_valid_molecule > 0 else 0.0
        )
        return smiles_dict, validity, uniqueness


# 相容性別名
MoleculeGenerator = MoleculeGeneratorCUDAQ


# ===========================================================================
# 快速測試
# ===========================================================================
if __name__ == "__main__":
    n   = 5
    cwg = ConditionalWeightsGenerator(n, smarts=None)
    w   = cwg.generate_conditional_random_weights(random_seed=3)
    mg  = MoleculeGeneratorCUDAQ(n, all_weight_vector=w, backend_name="cudaq_qpp")
    smiles_dict, validity, uniqueness = mg.sample_molecule(1000)
    print(f"Validity  : {validity*100:.2f}%")
    print(f"Uniqueness: {uniqueness*100:.2f}%")
    top5 = sorted(smiles_dict.items(), key=lambda x: -x[1])[:5]
    print("Top-5 SMILES:", top5)
