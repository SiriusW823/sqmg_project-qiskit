"""
==============================================================================
build_dynamic_circuit_cudaq.py  (v9 — 根本修正版)
==============================================================================

v8 → v9 修正：
  ★ BUG ROOT CAUSE (CUDA-Q 0.7.1 known bug):
      CUDA-Q 0.7.1 對含 mid-circuit measurement 的 kernel，
      get_sequential_data('__global__') 運作異常（0.8.0 才修）。
      v8 依賴 __global__ 取 bond-9 的 16 bits → 全部落入
      `except Exception: global_data = None` → bond9_bits = '0'*16
      → uniqueness 長期壓在 0.05~0.35

  ★ OOM FIX (Killed at eval ~155):
      v8 的 closure 工廠函式 make_qmg_n9_kernel() 每次呼叫
      都新建一個 MLIR module，del + gc.collect() 只能釋放 Python 層，
      C++ MLIR registry 持續累積 → 約 155 次後 OOM Killed。

  ★ v9 兩大修正：
    (1) 將最後 16 個 bond-9 mz() 全部命名（b91_0...b98_1），
        徹底消除對 __global__ 的依賴。
    (2) 改用 module-level parametric kernel _qmg_n9(w: list[float])，
        @cudaq.kernel 裝飾器在模組載入時只編譯一次 MLIR，
        每次呼叫 cudaq.sample(_qmg_n9, w_list, ...) 重用同一 MLIR，
        完全解決記憶體累積問題。

bitstring 結構（90 bits，對應 _N9_ALL_REGS 的順序）：
  bits[ 0: 2]  a1_0,  a1_1         (atom-1 type)
  bits[ 2: 4]  a2_0,  a2_1         (atom-2 type)
  bits[ 4: 6]  b21_0, b21_1        (bond 2→1)
  bits[ 6: 8]  a3_0,  a3_1
  bits[ 8:12]  b31_0, b31_1, b32_0, b32_1
  ...
  bits[72:74]  a9_0,  a9_1
  bits[74:76]  b91_0, b91_1
  ...
  bits[88:90]  b98_0, b98_1        (bond 9→8)
==============================================================================
"""
from __future__ import annotations

import math
import gc
import numpy as np
from typing import Union, List

import cudaq


# ===========================================================================
# Module-level Parametric Kernel  (MLIR 只編譯一次)
# ===========================================================================
#
# 重要設計原則：
#   - 定義在模組層級，@cudaq.kernel 裝飾器在 import 時一次性編譯 MLIR
#   - 每次 cudaq.sample(_qmg_n9, w_list, ...) 只傳遞不同的 w_list，
#     不新建 MLIR module → 無記憶體累積，徹底解決 OOM
#   - 所有 90 個 mz() 均有命名變數 → 不依賴 __global__ (CUDA-Q 0.7.1 bug)
#
@cudaq.kernel
def _qmg_n9(w: list[float]):
    """
    N=9 QMG 動態電路（parametric，w 長度 134）。
    所有 90 個 mz() 均有命名 → 對應 _N9_ALL_REGS 中的 90 個暫存器。
    """
    q = cudaq.qvector(20)

    # ================================================================
    # Phase 1: build_two_atoms   w[0:8]
    # ================================================================
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

    a1_0 = mz(q[0])
    a1_1 = mz(q[1])
    a2_0 = mz(q[2])
    a2_1 = mz(q[3])

    if a2_0 or a2_1:
        ry(math.pi * w[6], q[4])
        x(q[5])
        x.ctrl(q[4], q[5])
        ry.ctrl(math.pi * w[7], q[4], q[5])
    b21_0 = mz(q[4])
    b21_1 = mz(q[5])

    # ================================================================
    # Phase 2: atom 3   w[8:17]
    # ================================================================
    if a2_0:
        x(q[2])
    if a2_1:
        x(q[3])
    if b21_0:
        x(q[4])
    if b21_1:
        x(q[5])

    if a2_0 or a2_1:
        ry(math.pi * w[8],  q[2])
        ry(math.pi * w[9],  q[3])
        ry.ctrl(math.pi * w[10], q[2], q[3])
    a3_0 = mz(q[2])
    a3_1 = mz(q[3])

    if a3_0 or a3_1:
        ry(math.pi * w[11], q[5])
        ry.ctrl(math.pi * w[13], q[5], q[4])
        ry.ctrl(math.pi * w[14], q[4], q[5])
        ry(math.pi * w[12], q[7])
        ry.ctrl(math.pi * w[15], q[7], q[6])
        ry.ctrl(math.pi * w[16], q[6], q[7])
    b31_0 = mz(q[4])
    b31_1 = mz(q[5])
    b32_0 = mz(q[6])
    b32_1 = mz(q[7])

    # ================================================================
    # Phase 3: atom 4   w[17:29]
    # ================================================================
    if a3_0:
        x(q[2])
    if a3_1:
        x(q[3])
    if b31_0:
        x(q[4])
    if b31_1:
        x(q[5])
    if b32_0:
        x(q[6])
    if b32_1:
        x(q[7])

    if a3_0 or a3_1:
        ry(math.pi * w[17], q[2])
        ry(math.pi * w[18], q[3])
        ry.ctrl(math.pi * w[19], q[2], q[3])
    a4_0 = mz(q[2])
    a4_1 = mz(q[3])

    if a4_0 or a4_1:
        ry(math.pi * w[20], q[5])
        ry.ctrl(math.pi * w[23], q[5], q[4])
        ry.ctrl(math.pi * w[24], q[4], q[5])
        ry(math.pi * w[21], q[7])
        ry.ctrl(math.pi * w[25], q[7], q[6])
        ry.ctrl(math.pi * w[26], q[6], q[7])
        ry(math.pi * w[22], q[9])
        ry.ctrl(math.pi * w[27], q[9], q[8])
        ry.ctrl(math.pi * w[28], q[8], q[9])
    b41_0 = mz(q[4])
    b41_1 = mz(q[5])
    b42_0 = mz(q[6])
    b42_1 = mz(q[7])
    b43_0 = mz(q[8])
    b43_1 = mz(q[9])

    # ================================================================
    # Phase 4: atom 5   w[29:44]
    # ================================================================
    if a4_0:
        x(q[2])
    if a4_1:
        x(q[3])
    if b41_0:
        x(q[4])
    if b41_1:
        x(q[5])
    if b42_0:
        x(q[6])
    if b42_1:
        x(q[7])
    if b43_0:
        x(q[8])
    if b43_1:
        x(q[9])

    if a4_0 or a4_1:
        ry(math.pi * w[29], q[2])
        ry(math.pi * w[30], q[3])
        ry.ctrl(math.pi * w[31], q[2], q[3])
    a5_0 = mz(q[2])
    a5_1 = mz(q[3])

    if a5_0 or a5_1:
        ry(math.pi * w[32], q[5])
        ry.ctrl(math.pi * w[36], q[5], q[4])
        ry.ctrl(math.pi * w[37], q[4], q[5])
        ry(math.pi * w[33], q[7])
        ry.ctrl(math.pi * w[38], q[7], q[6])
        ry.ctrl(math.pi * w[39], q[6], q[7])
        ry(math.pi * w[34], q[9])
        ry.ctrl(math.pi * w[40], q[9], q[8])
        ry.ctrl(math.pi * w[41], q[8], q[9])
        ry(math.pi * w[35], q[11])
        ry.ctrl(math.pi * w[42], q[11], q[10])
        ry.ctrl(math.pi * w[43], q[10], q[11])
    b51_0 = mz(q[4])
    b51_1 = mz(q[5])
    b52_0 = mz(q[6])
    b52_1 = mz(q[7])
    b53_0 = mz(q[8])
    b53_1 = mz(q[9])
    b54_0 = mz(q[10])
    b54_1 = mz(q[11])

    # ================================================================
    # Phase 5: atom 6   w[44:62]
    # ================================================================
    if a5_0:
        x(q[2])
    if a5_1:
        x(q[3])
    if b51_0:
        x(q[4])
    if b51_1:
        x(q[5])
    if b52_0:
        x(q[6])
    if b52_1:
        x(q[7])
    if b53_0:
        x(q[8])
    if b53_1:
        x(q[9])
    if b54_0:
        x(q[10])
    if b54_1:
        x(q[11])

    if a5_0 or a5_1:
        ry(math.pi * w[44], q[2])
        ry(math.pi * w[45], q[3])
        ry.ctrl(math.pi * w[46], q[2], q[3])
    a6_0 = mz(q[2])
    a6_1 = mz(q[3])

    if a6_0 or a6_1:
        ry(math.pi * w[47], q[5])
        ry.ctrl(math.pi * w[52], q[5], q[4])
        ry.ctrl(math.pi * w[53], q[4], q[5])
        ry(math.pi * w[48], q[7])
        ry.ctrl(math.pi * w[54], q[7], q[6])
        ry.ctrl(math.pi * w[55], q[6], q[7])
        ry(math.pi * w[49], q[9])
        ry.ctrl(math.pi * w[56], q[9], q[8])
        ry.ctrl(math.pi * w[57], q[8], q[9])
        ry(math.pi * w[50], q[11])
        ry.ctrl(math.pi * w[58], q[11], q[10])
        ry.ctrl(math.pi * w[59], q[10], q[11])
        ry(math.pi * w[51], q[13])
        ry.ctrl(math.pi * w[60], q[13], q[12])
        ry.ctrl(math.pi * w[61], q[12], q[13])
    b61_0 = mz(q[4])
    b61_1 = mz(q[5])
    b62_0 = mz(q[6])
    b62_1 = mz(q[7])
    b63_0 = mz(q[8])
    b63_1 = mz(q[9])
    b64_0 = mz(q[10])
    b64_1 = mz(q[11])
    b65_0 = mz(q[12])
    b65_1 = mz(q[13])

    # ================================================================
    # Phase 6: atom 7   w[62:83]
    # ================================================================
    if a6_0:
        x(q[2])
    if a6_1:
        x(q[3])
    if b61_0:
        x(q[4])
    if b61_1:
        x(q[5])
    if b62_0:
        x(q[6])
    if b62_1:
        x(q[7])
    if b63_0:
        x(q[8])
    if b63_1:
        x(q[9])
    if b64_0:
        x(q[10])
    if b64_1:
        x(q[11])
    if b65_0:
        x(q[12])
    if b65_1:
        x(q[13])

    if a6_0 or a6_1:
        ry(math.pi * w[62], q[2])
        ry(math.pi * w[63], q[3])
        ry.ctrl(math.pi * w[64], q[2], q[3])
    a7_0 = mz(q[2])
    a7_1 = mz(q[3])

    if a7_0 or a7_1:
        ry(math.pi * w[65], q[5])
        ry.ctrl(math.pi * w[71], q[5], q[4])
        ry.ctrl(math.pi * w[72], q[4], q[5])
        ry(math.pi * w[66], q[7])
        ry.ctrl(math.pi * w[73], q[7], q[6])
        ry.ctrl(math.pi * w[74], q[6], q[7])
        ry(math.pi * w[67], q[9])
        ry.ctrl(math.pi * w[75], q[9], q[8])
        ry.ctrl(math.pi * w[76], q[8], q[9])
        ry(math.pi * w[68], q[11])
        ry.ctrl(math.pi * w[77], q[11], q[10])
        ry.ctrl(math.pi * w[78], q[10], q[11])
        ry(math.pi * w[69], q[13])
        ry.ctrl(math.pi * w[79], q[13], q[12])
        ry.ctrl(math.pi * w[80], q[12], q[13])
        ry(math.pi * w[70], q[15])
        ry.ctrl(math.pi * w[81], q[15], q[14])
        ry.ctrl(math.pi * w[82], q[14], q[15])
    b71_0 = mz(q[4])
    b71_1 = mz(q[5])
    b72_0 = mz(q[6])
    b72_1 = mz(q[7])
    b73_0 = mz(q[8])
    b73_1 = mz(q[9])
    b74_0 = mz(q[10])
    b74_1 = mz(q[11])
    b75_0 = mz(q[12])
    b75_1 = mz(q[13])
    b76_0 = mz(q[14])
    b76_1 = mz(q[15])

    # ================================================================
    # Phase 7: atom 8   w[83:107]
    # ================================================================
    if a7_0:
        x(q[2])
    if a7_1:
        x(q[3])
    if b71_0:
        x(q[4])
    if b71_1:
        x(q[5])
    if b72_0:
        x(q[6])
    if b72_1:
        x(q[7])
    if b73_0:
        x(q[8])
    if b73_1:
        x(q[9])
    if b74_0:
        x(q[10])
    if b74_1:
        x(q[11])
    if b75_0:
        x(q[12])
    if b75_1:
        x(q[13])
    if b76_0:
        x(q[14])
    if b76_1:
        x(q[15])

    if a7_0 or a7_1:
        ry(math.pi * w[83], q[2])
        ry(math.pi * w[84], q[3])
        ry.ctrl(math.pi * w[85], q[2], q[3])
    a8_0 = mz(q[2])
    a8_1 = mz(q[3])

    if a8_0 or a8_1:
        ry(math.pi * w[86], q[5])
        ry.ctrl(math.pi * w[93], q[5], q[4])
        ry.ctrl(math.pi * w[94], q[4], q[5])
        ry(math.pi * w[87], q[7])
        ry.ctrl(math.pi * w[95], q[7], q[6])
        ry.ctrl(math.pi * w[96], q[6], q[7])
        ry(math.pi * w[88], q[9])
        ry.ctrl(math.pi * w[97], q[9], q[8])
        ry.ctrl(math.pi * w[98], q[8], q[9])
        ry(math.pi * w[89], q[11])
        ry.ctrl(math.pi * w[99], q[11], q[10])
        ry.ctrl(math.pi * w[100], q[10], q[11])
        ry(math.pi * w[90], q[13])
        ry.ctrl(math.pi * w[101], q[13], q[12])
        ry.ctrl(math.pi * w[102], q[12], q[13])
        ry(math.pi * w[91], q[15])
        ry.ctrl(math.pi * w[103], q[15], q[14])
        ry.ctrl(math.pi * w[104], q[14], q[15])
        ry(math.pi * w[92], q[17])
        ry.ctrl(math.pi * w[105], q[17], q[16])
        ry.ctrl(math.pi * w[106], q[16], q[17])
    b81_0 = mz(q[4])
    b81_1 = mz(q[5])
    b82_0 = mz(q[6])
    b82_1 = mz(q[7])
    b83_0 = mz(q[8])
    b83_1 = mz(q[9])
    b84_0 = mz(q[10])
    b84_1 = mz(q[11])
    b85_0 = mz(q[12])
    b85_1 = mz(q[13])
    b86_0 = mz(q[14])
    b86_1 = mz(q[15])
    b87_0 = mz(q[16])
    b87_1 = mz(q[17])

    # ================================================================
    # Phase 8: atom 9   w[107:134]
    # ================================================================
    if a8_0:
        x(q[2])
    if a8_1:
        x(q[3])
    if b81_0:
        x(q[4])
    if b81_1:
        x(q[5])
    if b82_0:
        x(q[6])
    if b82_1:
        x(q[7])
    if b83_0:
        x(q[8])
    if b83_1:
        x(q[9])
    if b84_0:
        x(q[10])
    if b84_1:
        x(q[11])
    if b85_0:
        x(q[12])
    if b85_1:
        x(q[13])
    if b86_0:
        x(q[14])
    if b86_1:
        x(q[15])
    if b87_0:
        x(q[16])
    if b87_1:
        x(q[17])

    if a8_0 or a8_1:
        ry(math.pi * w[107], q[2])
        ry(math.pi * w[108], q[3])
        ry.ctrl(math.pi * w[109], q[2], q[3])

    a9_0 = mz(q[2])
    a9_1 = mz(q[3])

    if a9_0 or a9_1:
        ry(math.pi * w[110], q[5])
        ry.ctrl(math.pi * w[118], q[5], q[4])
        ry.ctrl(math.pi * w[119], q[4], q[5])
        ry(math.pi * w[111], q[7])
        ry.ctrl(math.pi * w[120], q[7], q[6])
        ry.ctrl(math.pi * w[121], q[6], q[7])
        ry(math.pi * w[112], q[9])
        ry.ctrl(math.pi * w[122], q[9], q[8])
        ry.ctrl(math.pi * w[123], q[8], q[9])
        ry(math.pi * w[113], q[11])
        ry.ctrl(math.pi * w[124], q[11], q[10])
        ry.ctrl(math.pi * w[125], q[10], q[11])
        ry(math.pi * w[114], q[13])
        ry.ctrl(math.pi * w[126], q[13], q[12])
        ry.ctrl(math.pi * w[127], q[12], q[13])
        ry(math.pi * w[115], q[15])
        ry.ctrl(math.pi * w[128], q[15], q[14])
        ry.ctrl(math.pi * w[129], q[14], q[15])
        ry(math.pi * w[116], q[17])
        ry.ctrl(math.pi * w[130], q[17], q[16])
        ry.ctrl(math.pi * w[131], q[16], q[17])
        ry(math.pi * w[117], q[19])
        ry.ctrl(math.pi * w[132], q[19], q[18])
        ry.ctrl(math.pi * w[133], q[18], q[19])

    # ★ v9 修正：全部命名，徹底消除對 __global__ 的依賴
    # 對應 _N9_ALL_REGS[74:90]
    b91_0 = mz(q[4]);  b91_1 = mz(q[5])
    b92_0 = mz(q[6]);  b92_1 = mz(q[7])
    b93_0 = mz(q[8]);  b93_1 = mz(q[9])
    b94_0 = mz(q[10]); b94_1 = mz(q[11])
    b95_0 = mz(q[12]); b95_1 = mz(q[13])
    b96_0 = mz(q[14]); b96_1 = mz(q[15])
    b97_0 = mz(q[16]); b97_1 = mz(q[17])
    b98_0 = mz(q[18]); b98_1 = mz(q[19])


# ===========================================================================
# DynamicCircuitBuilderCUDAQ  (v9)
# ===========================================================================

class DynamicCircuitBuilderCUDAQ:
    """
    CUDA-Q 0.7.1 版 QMG 動態電路建構器（v9 parametric kernel 版）。

    v9 設計改變：
      - 使用 module-level parametric kernel _qmg_n9(w: list[float])
      - MLIR 只在模組載入時編譯一次，每次評估重用
      - 解決 OOM 問題（v8 closure 方式的根本問題）
    """

    def __init__(
        self,
        num_heavy_atom:            int,
        temperature:               float = 0.2,
        remove_bond_disconnection: bool  = True,
        chemistry_constraint:      bool  = True,
    ):
        if num_heavy_atom != 9:
            raise NotImplementedError(f"目前僅支援 N=9（N={num_heavy_atom} 尚未實作）。")
        self.num_heavy_atom            = num_heavy_atom
        self.temperature               = temperature
        self.remove_bond_disconnection = remove_bond_disconnection
        self.chemistry_constraint      = chemistry_constraint
        self.num_clbits                = num_heavy_atom * (num_heavy_atom + 1)  # 90
        self.length_all_weight_vector  = int(
            8 + (num_heavy_atom - 2) * (num_heavy_atom + 3) * 3 / 2
        )  # 134

    def get_kernel(self):
        """回傳 module-level parametric kernel（已在載入時編譯完畢）。"""
        return _qmg_n9

    def prepare_weights(self, weights) -> list:
        """將 numpy array 或 list 轉換為 Python list[float]（134 個元素）。"""
        if hasattr(weights, 'tolist'):
            w_list = weights.tolist()
        else:
            w_list = list(weights)
        assert len(w_list) == self.length_all_weight_vector, (
            f"weights 長度 {len(w_list)} != {self.length_all_weight_vector}"
        )
        return [float(x) for x in w_list]

    # 保留舊介面以向後相容（但已不再需要新建 closure kernel）
    def build_kernel_from_weights(self, weights):
        """
        [已廢棄，保留向後相容] v9 不再需要每次建立新 kernel。
        請改用 get_kernel() + prepare_weights() 或直接使用 _qmg_n9。
        """
        import warnings
        warnings.warn(
            "build_kernel_from_weights() 已廢棄。"
            "請改用 get_kernel() + prepare_weights()。"
            "v9 使用 parametric kernel，不再需要此方法。",
            DeprecationWarning,
            stacklevel=2,
        )
        return _qmg_n9  # 回傳同一個 parametric kernel（忽略 weights）

    def apply_bond_disconnection_correction(self, bitstring: str) -> str:
        """
        修正孤立重原子（存在但無鍵連接）的 bitstring。
        bitstring 順序與 _N9_ALL_REGS 一致（90 bits）。
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