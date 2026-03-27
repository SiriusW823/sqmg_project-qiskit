"""
==============================================================================
build_dynamic_circuit_cudaq.py
CUDA-Q 版本的 QMG 動態量子電路（對應 Qiskit DynamicCircuitBuilder）
==============================================================================

Qiskit → CUDA-Q 核心對應關係
─────────────────────────────────────────────────────────────────────────────
  qc.ry(π·w, q[i])                       →  ry(math.pi * w, q[i])
  qc.x(q[i])                             →  x(q[i])
  qc.cx(q[i], q[j])                      →  x.ctrl(q[i], q[j])
  qc.cry(π·w, ctrl, tgt)                 →  ry.ctrl(π·w, q[ctrl], q[tgt])
  qc.if_test((CR, 0)) as else_: ... with else_: <body>
                                          →  if bit_from_mz:
                                                 <body>
  qc.measure(q[i], c[j])                 →  c_j = mz(q[i])

重要差異說明
─────────────────────────────────────────────────────────────────────────────
  1. Bit ordering
     - Qiskit：get_bitstrings() big-endian（bit[0]=最高位）
       → post_process_quantum_state 需 reverse=True
     - CUDA-Q：get_bitstrings() 依 mz() 呼叫順序，bit[0]=第一個 mz
       → post_process_quantum_state 需 reverse=False

  2. Bond-disconnection correction
     - Qiskit：電路內 if_test + X gate + re-measure
     - CUDA-Q：等效後處理 apply_bond_disconnection_correction()

  3. chemistry_constraint 由 weight_generator 前處理完成（兩版相同）

  4. Qubit layout（N=9，共 20 qubits = 4 + 2*(N-1)）
     q[0:2]  原子 1 類型（永久）
     q[2:4]  原子 k 類型工作區（重用）
     q[4:20] 鍵類型工作區（重用）

  5. Clbit layout（90 bits，依 mz() 呼叫順序）
     bits[ 0: 2] 原子 1   bits[ 2: 4] 原子 2   bits[ 4: 6] 鍵 2-1
     bits[ 6: 8] 原子 3   bits[ 8:12] 鍵 3-{1,2}
     bits[12:14] 原子 4   bits[14:20] 鍵 4-{1,2,3}
     bits[20:22] 原子 5   bits[22:30] 鍵 5-{1..4}
     bits[30:32] 原子 6   bits[32:42] 鍵 6-{1..5}
     bits[42:44] 原子 7   bits[44:56] 鍵 7-{1..6}
     bits[56:58] 原子 8   bits[58:72] 鍵 8-{1..7}
     bits[72:74] 原子 9   bits[74:90] 鍵 9-{1..8}

目前支援：N = 9（unconditional_9 實驗）
==============================================================================
"""
from __future__ import annotations

import math
import numpy as np
from typing import Union, List

import cudaq


# ===========================================================================
# CUDA-Q Kernel（N=9 完全展開版）
# ===========================================================================

@cudaq.kernel
def _qmg_dynamic_n9(weights: list[float]):
    """N=9 QMG 動態分子生成電路（CUDA-Q 實作）。"""
    q = cudaq.qvector(20)  # 20 qubits for N=9

    # ========================================================================
    # Phase 1: build_two_atoms   weights[0:8]
    # ========================================================================
    ry(math.pi * weights[0], q[0])
    x(q[1])
    ry(math.pi * weights[2], q[2])
    ry(math.pi * weights[4], q[3])
    x.ctrl(q[0], q[1])
    ry.ctrl(math.pi * weights[3], q[1], q[2])
    x.ctrl(q[2], q[3])
    ry.ctrl(math.pi * weights[1], q[0], q[1])
    x.ctrl(q[1], q[2])
    ry.ctrl(math.pi * weights[5], q[2], q[3])

    a1_0 = mz(q[0]);  a1_1 = mz(q[1])    # bits[ 0: 2] 原子 1
    a2_0 = mz(q[2]);  a2_1 = mz(q[3])    # bits[ 2: 4] 原子 2

    if a2_0 or a2_1:
        ry(math.pi * weights[6], q[4])
        x(q[5])
        x.ctrl(q[4], q[5])
        ry.ctrl(math.pi * weights[7], q[4], q[5])
    b21_0 = mz(q[4]);  b21_1 = mz(q[5])  # bits[ 4: 6] 鍵 2-1

    # ========================================================================
    # Phase 2: atom 3   weights[8:17]
    # ========================================================================
    if a2_0:  x(q[2])
    if a2_1:  x(q[3])
    if b21_0: x(q[4])
    if b21_1: x(q[5])

    if a2_0 or a2_1:
        ry(math.pi * weights[8],  q[2])
        ry(math.pi * weights[9],  q[3])
        ry.ctrl(math.pi * weights[10], q[2], q[3])
    a3_0 = mz(q[2]);  a3_1 = mz(q[3])    # bits[ 6: 8] 原子 3

    if a3_0 or a3_1:
        ry(math.pi * weights[11], q[5])
        ry.ctrl(math.pi * weights[13], q[5], q[4])
        ry.ctrl(math.pi * weights[14], q[4], q[5])
        ry(math.pi * weights[12], q[7])
        ry.ctrl(math.pi * weights[15], q[7], q[6])
        ry.ctrl(math.pi * weights[16], q[6], q[7])
    b31_0 = mz(q[4]);  b31_1 = mz(q[5])  # bits[ 8:10] 鍵 3-1
    b32_0 = mz(q[6]);  b32_1 = mz(q[7])  # bits[10:12] 鍵 3-2

    # ========================================================================
    # Phase 3: atom 4   weights[17:29]
    # ========================================================================
    if a3_0:  x(q[2])
    if a3_1:  x(q[3])
    if b31_0: x(q[4])
    if b31_1: x(q[5])
    if b32_0: x(q[6])
    if b32_1: x(q[7])

    if a3_0 or a3_1:
        ry(math.pi * weights[17], q[2])
        ry(math.pi * weights[18], q[3])
        ry.ctrl(math.pi * weights[19], q[2], q[3])
    a4_0 = mz(q[2]);  a4_1 = mz(q[3])    # bits[12:14] 原子 4

    if a4_0 or a4_1:
        ry(math.pi * weights[20], q[5])
        ry.ctrl(math.pi * weights[23], q[5], q[4])
        ry.ctrl(math.pi * weights[24], q[4], q[5])
        ry(math.pi * weights[21], q[7])
        ry.ctrl(math.pi * weights[25], q[7], q[6])
        ry.ctrl(math.pi * weights[26], q[6], q[7])
        ry(math.pi * weights[22], q[9])
        ry.ctrl(math.pi * weights[27], q[9], q[8])
        ry.ctrl(math.pi * weights[28], q[8], q[9])
    b41_0 = mz(q[4]);  b41_1 = mz(q[5])  # bits[14:16] 鍵 4-1
    b42_0 = mz(q[6]);  b42_1 = mz(q[7])  # bits[16:18] 鍵 4-2
    b43_0 = mz(q[8]);  b43_1 = mz(q[9])  # bits[18:20] 鍵 4-3

    # ========================================================================
    # Phase 4: atom 5   weights[29:44]
    # ========================================================================
    if a4_0:  x(q[2])
    if a4_1:  x(q[3])
    if b41_0: x(q[4])
    if b41_1: x(q[5])
    if b42_0: x(q[6])
    if b42_1: x(q[7])
    if b43_0: x(q[8])
    if b43_1: x(q[9])

    if a4_0 or a4_1:
        ry(math.pi * weights[29], q[2])
        ry(math.pi * weights[30], q[3])
        ry.ctrl(math.pi * weights[31], q[2], q[3])
    a5_0 = mz(q[2]);  a5_1 = mz(q[3])    # bits[20:22] 原子 5

    if a5_0 or a5_1:
        ry(math.pi * weights[32], q[5])
        ry.ctrl(math.pi * weights[36], q[5], q[4])
        ry.ctrl(math.pi * weights[37], q[4], q[5])
        ry(math.pi * weights[33], q[7])
        ry.ctrl(math.pi * weights[38], q[7], q[6])
        ry.ctrl(math.pi * weights[39], q[6], q[7])
        ry(math.pi * weights[34], q[9])
        ry.ctrl(math.pi * weights[40], q[9], q[8])
        ry.ctrl(math.pi * weights[41], q[8], q[9])
        ry(math.pi * weights[35], q[11])
        ry.ctrl(math.pi * weights[42], q[11], q[10])
        ry.ctrl(math.pi * weights[43], q[10], q[11])
    b51_0 = mz(q[4]);  b51_1 = mz(q[5])  # bits[22:24] 鍵 5-1
    b52_0 = mz(q[6]);  b52_1 = mz(q[7])  # bits[24:26] 鍵 5-2
    b53_0 = mz(q[8]);  b53_1 = mz(q[9])  # bits[26:28] 鍵 5-3
    b54_0 = mz(q[10]); b54_1 = mz(q[11]) # bits[28:30] 鍵 5-4

    # ========================================================================
    # Phase 5: atom 6   weights[44:62]
    # ========================================================================
    if a5_0:  x(q[2])
    if a5_1:  x(q[3])
    if b51_0: x(q[4])
    if b51_1: x(q[5])
    if b52_0: x(q[6])
    if b52_1: x(q[7])
    if b53_0: x(q[8])
    if b53_1: x(q[9])
    if b54_0: x(q[10])
    if b54_1: x(q[11])

    if a5_0 or a5_1:
        ry(math.pi * weights[44], q[2])
        ry(math.pi * weights[45], q[3])
        ry.ctrl(math.pi * weights[46], q[2], q[3])
    a6_0 = mz(q[2]);  a6_1 = mz(q[3])    # bits[30:32] 原子 6

    if a6_0 or a6_1:
        ry(math.pi * weights[47], q[5])
        ry.ctrl(math.pi * weights[52], q[5], q[4])
        ry.ctrl(math.pi * weights[53], q[4], q[5])
        ry(math.pi * weights[48], q[7])
        ry.ctrl(math.pi * weights[54], q[7], q[6])
        ry.ctrl(math.pi * weights[55], q[6], q[7])
        ry(math.pi * weights[49], q[9])
        ry.ctrl(math.pi * weights[56], q[9], q[8])
        ry.ctrl(math.pi * weights[57], q[8], q[9])
        ry(math.pi * weights[50], q[11])
        ry.ctrl(math.pi * weights[58], q[11], q[10])
        ry.ctrl(math.pi * weights[59], q[10], q[11])
        ry(math.pi * weights[51], q[13])
        ry.ctrl(math.pi * weights[60], q[13], q[12])
        ry.ctrl(math.pi * weights[61], q[12], q[13])
    b61_0 = mz(q[4]);  b61_1 = mz(q[5])  # bits[32:34] 鍵 6-1
    b62_0 = mz(q[6]);  b62_1 = mz(q[7])  # bits[34:36] 鍵 6-2
    b63_0 = mz(q[8]);  b63_1 = mz(q[9])  # bits[36:38] 鍵 6-3
    b64_0 = mz(q[10]); b64_1 = mz(q[11]) # bits[38:40] 鍵 6-4
    b65_0 = mz(q[12]); b65_1 = mz(q[13]) # bits[40:42] 鍵 6-5

    # ========================================================================
    # Phase 6: atom 7   weights[62:83]
    # ========================================================================
    if a6_0:  x(q[2])
    if a6_1:  x(q[3])
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
        ry(math.pi * weights[62], q[2])
        ry(math.pi * weights[63], q[3])
        ry.ctrl(math.pi * weights[64], q[2], q[3])
    a7_0 = mz(q[2]);  a7_1 = mz(q[3])    # bits[42:44] 原子 7

    if a7_0 or a7_1:
        ry(math.pi * weights[65], q[5])
        ry.ctrl(math.pi * weights[71], q[5], q[4])
        ry.ctrl(math.pi * weights[72], q[4], q[5])
        ry(math.pi * weights[66], q[7])
        ry.ctrl(math.pi * weights[73], q[7], q[6])
        ry.ctrl(math.pi * weights[74], q[6], q[7])
        ry(math.pi * weights[67], q[9])
        ry.ctrl(math.pi * weights[75], q[9], q[8])
        ry.ctrl(math.pi * weights[76], q[8], q[9])
        ry(math.pi * weights[68], q[11])
        ry.ctrl(math.pi * weights[77], q[11], q[10])
        ry.ctrl(math.pi * weights[78], q[10], q[11])
        ry(math.pi * weights[69], q[13])
        ry.ctrl(math.pi * weights[79], q[13], q[12])
        ry.ctrl(math.pi * weights[80], q[12], q[13])
        ry(math.pi * weights[70], q[15])
        ry.ctrl(math.pi * weights[81], q[15], q[14])
        ry.ctrl(math.pi * weights[82], q[14], q[15])
    b71_0 = mz(q[4]);  b71_1 = mz(q[5])  # bits[44:46] 鍵 7-1
    b72_0 = mz(q[6]);  b72_1 = mz(q[7])  # bits[46:48] 鍵 7-2
    b73_0 = mz(q[8]);  b73_1 = mz(q[9])  # bits[48:50] 鍵 7-3
    b74_0 = mz(q[10]); b74_1 = mz(q[11]) # bits[50:52] 鍵 7-4
    b75_0 = mz(q[12]); b75_1 = mz(q[13]) # bits[52:54] 鍵 7-5
    b76_0 = mz(q[14]); b76_1 = mz(q[15]) # bits[54:56] 鍵 7-6

    # ========================================================================
    # Phase 7: atom 8   weights[83:107]
    # ========================================================================
    if a7_0:  x(q[2])
    if a7_1:  x(q[3])
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
        ry(math.pi * weights[83], q[2])
        ry(math.pi * weights[84], q[3])
        ry.ctrl(math.pi * weights[85], q[2], q[3])
    a8_0 = mz(q[2]);  a8_1 = mz(q[3])    # bits[56:58] 原子 8

    if a8_0 or a8_1:
        ry(math.pi * weights[86], q[5])
        ry.ctrl(math.pi * weights[93], q[5], q[4])
        ry.ctrl(math.pi * weights[94], q[4], q[5])
        ry(math.pi * weights[87], q[7])
        ry.ctrl(math.pi * weights[95], q[7], q[6])
        ry.ctrl(math.pi * weights[96], q[6], q[7])
        ry(math.pi * weights[88], q[9])
        ry.ctrl(math.pi * weights[97], q[9], q[8])
        ry.ctrl(math.pi * weights[98], q[8], q[9])
        ry(math.pi * weights[89], q[11])
        ry.ctrl(math.pi * weights[99], q[11], q[10])
        ry.ctrl(math.pi * weights[100], q[10], q[11])
        ry(math.pi * weights[90], q[13])
        ry.ctrl(math.pi * weights[101], q[13], q[12])
        ry.ctrl(math.pi * weights[102], q[12], q[13])
        ry(math.pi * weights[91], q[15])
        ry.ctrl(math.pi * weights[103], q[15], q[14])
        ry.ctrl(math.pi * weights[104], q[14], q[15])
        ry(math.pi * weights[92], q[17])
        ry.ctrl(math.pi * weights[105], q[17], q[16])
        ry.ctrl(math.pi * weights[106], q[16], q[17])
    b81_0 = mz(q[4]);  b81_1 = mz(q[5])  # bits[58:60] 鍵 8-1
    b82_0 = mz(q[6]);  b82_1 = mz(q[7])  # bits[60:62] 鍵 8-2
    b83_0 = mz(q[8]);  b83_1 = mz(q[9])  # bits[62:64] 鍵 8-3
    b84_0 = mz(q[10]); b84_1 = mz(q[11]) # bits[64:66] 鍵 8-4
    b85_0 = mz(q[12]); b85_1 = mz(q[13]) # bits[66:68] 鍵 8-5
    b86_0 = mz(q[14]); b86_1 = mz(q[15]) # bits[68:70] 鍵 8-6
    b87_0 = mz(q[16]); b87_1 = mz(q[17]) # bits[70:72] 鍵 8-7

    # ========================================================================
    # Phase 8: atom 9   weights[107:134]
    # ========================================================================
    if a8_0:  x(q[2])
    if a8_1:  x(q[3])
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

    # atom 9 type：條件 = atom_8_existence
    if a8_0 or a8_1:
        ry(math.pi * weights[107], q[2])
        ry(math.pi * weights[108], q[3])
        ry.ctrl(math.pi * weights[109], q[2], q[3])

    # ★ 必須捕捉 atom 9 測量值，供 bond 9 條件判斷使用
    a9_0 = mz(q[2]);  a9_1 = mz(q[3])    # bits[72:74] 原子 9

    # 鍵 9-{1..8}：條件 = atom_9_existence（★ 修正：用 a9 而非 a8）
    if a9_0 or a9_1:
        ry(math.pi * weights[110], q[5])
        ry.ctrl(math.pi * weights[118], q[5], q[4])
        ry.ctrl(math.pi * weights[119], q[4], q[5])
        ry(math.pi * weights[111], q[7])
        ry.ctrl(math.pi * weights[120], q[7], q[6])
        ry.ctrl(math.pi * weights[121], q[6], q[7])
        ry(math.pi * weights[112], q[9])
        ry.ctrl(math.pi * weights[122], q[9], q[8])
        ry.ctrl(math.pi * weights[123], q[8], q[9])
        ry(math.pi * weights[113], q[11])
        ry.ctrl(math.pi * weights[124], q[11], q[10])
        ry.ctrl(math.pi * weights[125], q[10], q[11])
        ry(math.pi * weights[114], q[13])
        ry.ctrl(math.pi * weights[126], q[13], q[12])
        ry.ctrl(math.pi * weights[127], q[12], q[13])
        ry(math.pi * weights[115], q[15])
        ry.ctrl(math.pi * weights[128], q[15], q[14])
        ry.ctrl(math.pi * weights[129], q[14], q[15])
        ry(math.pi * weights[116], q[17])
        ry.ctrl(math.pi * weights[130], q[17], q[16])
        ry.ctrl(math.pi * weights[131], q[16], q[17])
        ry(math.pi * weights[117], q[19])
        ry.ctrl(math.pi * weights[132], q[19], q[18])
        ry.ctrl(math.pi * weights[133], q[18], q[19])

    mz(q[4]);  mz(q[5])   # bits[74:76]  鍵 9-1
    mz(q[6]);  mz(q[7])   # bits[76:78]  鍵 9-2
    mz(q[8]);  mz(q[9])   # bits[78:80]  鍵 9-3
    mz(q[10]); mz(q[11])  # bits[80:82]  鍵 9-4
    mz(q[12]); mz(q[13])  # bits[82:84]  鍵 9-5
    mz(q[14]); mz(q[15])  # bits[84:86]  鍵 9-6
    mz(q[16]); mz(q[17])  # bits[86:88]  鍵 9-7
    mz(q[18]); mz(q[19])  # bits[88:90]  鍵 9-8


# ===========================================================================
# DynamicCircuitBuilderCUDAQ
# ===========================================================================

class DynamicCircuitBuilderCUDAQ:
    """
    CUDA-Q 版 QMG 動態電路建構器。
    公開介面與 Qiskit DynamicCircuitBuilder 完全相同。
    目前支援 N=9；其他 N 可依相同模板擴充。
    """

    _KERNELS = {9: _qmg_dynamic_n9}

    def __init__(
        self,
        num_heavy_atom:            int,
        temperature:               float = 0.2,
        remove_bond_disconnection: bool  = True,
        chemistry_constraint:      bool  = True,
    ):
        if num_heavy_atom not in self._KERNELS:
            raise NotImplementedError(
                f"CUDA-Q 核心目前僅支援 N ∈ {list(self._KERNELS)}；"
                f"請依相同模板實作 N={num_heavy_atom} 的 @cudaq.kernel。"
            )
        self.num_heavy_atom            = num_heavy_atom
        self.temperature               = temperature
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint      = chemistry_constraint
        self.num_clbits                = num_heavy_atom * (num_heavy_atom + 1)   # 90
        self.length_all_weight_vector  = int(
            8 + (num_heavy_atom - 2) * (num_heavy_atom + 3) * 3 / 2
        )  # 134

    def get_kernel(self):
        """回傳 CUDA-Q kernel 函式。"""
        return self._KERNELS[self.num_heavy_atom]

    def softmax_temperature(
        self, weight_vector: np.ndarray, temperature: float = None
    ) -> np.ndarray:
        t    = temperature if temperature is not None else self.temperature
        v    = weight_vector / t
        exps = np.exp(v - np.max(v))
        return exps / np.sum(exps)

    def apply_bond_disconnection_correction(self, bitstring: str) -> str:
        """
        後處理：確保每個存在的原子至少有一條鍵。
        等效於 Qiskit 電路內 bond_disconnection_CR + X gate + re-measure。

        Clbit layout（reverse=False）：
          原子 k：bits[(k-1)²+(k-1):(k-1)²+(k-1)+2]
          鍵  k ：bits[k²-k+2 : k²-k+2+2*(k-1)]
        """
        if not self.remove_bond_disconnection:
            return bitstring

        n    = self.num_heavy_atom
        bits = list(bitstring)

        for k in range(3, n + 1):
            atom_start  = (k - 1) ** 2 + (k - 1)
            atom_exists = bits[atom_start] == '1' or bits[atom_start + 1] == '1'
            if not atom_exists:
                continue
            bond_start = k * k - k + 2
            bond_end   = bond_start + 2 * (k - 1)
            if all(b == '0' for b in bits[bond_start:bond_end]):
                bits[bond_end - 1] = '1'

        return ''.join(bits)