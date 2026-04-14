"""
cudaq_n9_diagnostic.py  (v8.1)
針對 QMG N=9 kernel 本身的深度診斷

★ v8.1 修正：
  原始版本引用已移除的 _qmg_dynamic_n9（v8 起已不再 export），
  改為正確引用 make_qmg_n9_kernel，並更新對應呼叫方式。

放到 ~/sqmg_project-cudaq/ 後執行：
  PYTHONPATH=~/sqmg_project-cudaq python cudaq_n9_diagnostic.py 2>&1 | tee n9_diag.txt
"""
import math
import cudaq
import numpy as np

print("=" * 70)
print("QMG N=9 Kernel 深度診斷 (v8.1)")
print("=" * 70)

# ── Test A：直接匯入並呼叫 make_qmg_n9_kernel ────────────────────────────
print("\n[Test A] 直接 import make_qmg_n9_kernel 並 sample（不透過 Generator）")
try:
    cudaq.set_target("qpp-cpu")
    # ★ v8.1 修正：v8 起 _qmg_dynamic_n9 已移除，改用工廠函式 make_qmg_n9_kernel
    from qmg.utils.build_dynamic_circuit_cudaq import make_qmg_n9_kernel
    w = [0.5] * 134
    kernel = make_qmg_n9_kernel(w)
    result = cudaq.sample(kernel, shots_count=5)
    print(f"  ✓ 成功：{dict(list(result.items())[:3])}...")
    del kernel
    import gc; gc.collect()
except Exception as e:
    print(f"  ✗ 失敗：{e}")

# ── Test B：印出 kernel 的 MLIR module（看簽名是否正確）─────────────────
print("\n[Test B] 印出 make_qmg_n9_kernel 產生的 MLIR module（前 40 行）")
try:
    from qmg.utils.build_dynamic_circuit_cudaq import make_qmg_n9_kernel
    import gc
    kernel = make_qmg_n9_kernel([0.5] * 134)
    mlir_str = str(kernel.module)
    lines = mlir_str.splitlines()
    for i, line in enumerate(lines[:40], 1):
        print(f"  {i:3d} | {line}")
    if len(lines) > 40:
        print(f"  ... ({len(lines)} lines total)")
    del kernel; gc.collect()
except Exception as e:
    print(f"  ✗ 失敗：{e}")

# ── Test C：找出第一個讓 sample 失敗的操作 ────────────────────────────────
print("\n[Test C] 找出問題節點 — 逐步增加複雜度")

# C1: 只有 ry + mz（no conditional）
@cudaq.kernel
def k_c1(weights: list[float]):
    q = cudaq.qvector(20)
    ry(math.pi * weights[0], q[0])
    x(q[1])
    ry(math.pi * weights[2], q[2])
    a1_0 = mz(q[0])
    a1_1 = mz(q[1])
    a2_0 = mz(q[2])
    a2_1 = mz(q[3])

try:
    r = cudaq.sample(k_c1, [0.5]*134, shots_count=5)
    print(f"  C1 (ry+mz, no cond): ✓ {dict(list(r.items())[:2])}")
except Exception as e:
    print(f"  C1 ✗ {e}")

# C2: + 第一個 if a2_0 or a2_1 條件
@cudaq.kernel
def k_c2(weights: list[float]):
    q = cudaq.qvector(20)
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
    a1_0 = mz(q[0])
    a1_1 = mz(q[1])
    a2_0 = mz(q[2])
    a2_1 = mz(q[3])
    if a2_0 or a2_1:
        ry(math.pi * weights[6], q[4])
        x(q[5])
        x.ctrl(q[4], q[5])
        ry.ctrl(math.pi * weights[7], q[4], q[5])
    b21_0 = mz(q[4])
    b21_1 = mz(q[5])

try:
    r = cudaq.sample(k_c2, [0.5]*134, shots_count=5)
    print(f"  C2 (+ if a2_0 or a2_1): ✓ {dict(list(r.items())[:2])}")
except Exception as e:
    print(f"  C2 ✗ {e}")

# C3: + Phase 2（包含 if a2_0: x(q[2]) 的 reset pattern）
@cudaq.kernel
def k_c3(weights: list[float]):
    q = cudaq.qvector(20)
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
    a1_0 = mz(q[0])
    a1_1 = mz(q[1])
    a2_0 = mz(q[2])
    a2_1 = mz(q[3])
    if a2_0 or a2_1:
        ry(math.pi * weights[6], q[4])
        x(q[5])
        x.ctrl(q[4], q[5])
        ry.ctrl(math.pi * weights[7], q[4], q[5])
    b21_0 = mz(q[4])
    b21_1 = mz(q[5])
    # Phase 2 reset pattern
    if a2_0:
        x(q[2])
    if a2_1:
        x(q[3])
    if b21_0:
        x(q[4])
    if b21_1:
        x(q[5])
    if a2_0 or a2_1:
        ry(math.pi * weights[8],  q[2])
        ry(math.pi * weights[9],  q[3])
        ry.ctrl(math.pi * weights[10], q[2], q[3])
    a3_0 = mz(q[2])
    a3_1 = mz(q[3])

try:
    r = cudaq.sample(k_c3, [0.5]*134, shots_count=5)
    print(f"  C3 (+ Phase 2 reset+atom3): ✓ {dict(list(r.items())[:2])}")
except Exception as e:
    print(f"  C3 ✗ {e}")

# ── Test D：測試 make_qmg_n9_kernel 的 MLIR 是否能正常 compile ──────────
print("\n[Test D] 嘗試強制 compile make_qmg_n9_kernel([0.5]*134)")
try:
    import gc
    from qmg.utils.build_dynamic_circuit_cudaq import make_qmg_n9_kernel
    kernel = make_qmg_n9_kernel([0.5] * 134)
    kernel.compile()
    print("  compile() 成功")
    del kernel; gc.collect()
except Exception as e:
    print(f"  compile() 失敗：{e}")

# ── Test E：完整 N=9 kernel 採樣驗證（GPU + CPU）─────────────────────────
print("\n[Test E] 完整 N=9 kernel 採樣驗證（10 shots）")
import gc, time
from qmg.utils.build_dynamic_circuit_cudaq import make_qmg_n9_kernel

for target in ["qpp-cpu", "nvidia"]:
    print(f"\n  [Target: {target}]")
    try:
        cudaq.set_target(target)
        kernel = make_qmg_n9_kernel([0.5] * 134)
        t0 = time.time()
        r = cudaq.sample(kernel, shots_count=10)
        elapsed = time.time() - t0
        items = dict(list(r.items())[:3])
        print(f"  ✓ 成功：{items}  ({elapsed:.2f}s)")
        # 驗證 bitstring 長度
        sample_bs = list(r.items())
        if sample_bs:
            bs_len = len(sample_bs[0][0])
            print(f"  bitstring 長度: {bs_len} (預期 ≤ 90)")
        del kernel; gc.collect()
    except Exception as e:
        print(f"  ✗ 失敗：{e}")

# ── Test F：查 sample.py 完整內容 ────────────────────────────────────────
print("\n[Test F] sample.py 完整內容")
import importlib.util, os
spec = importlib.util.find_spec("cudaq")
sample_py = os.path.join(os.path.dirname(spec.origin), "runtime", "sample.py")
if os.path.exists(sample_py):
    with open(sample_py) as f:
        lines = f.readlines()
    print(f"  共 {len(lines)} 行，前 40 行：")
    for i, line in enumerate(lines[:40], 1):
        print(f"  {i:4d} | {line}", end='')
else:
    print(f"  {sample_py} 不存在")

print("\n" + "=" * 70)
print("診斷完成 (v8.1)")
print("=" * 70)