"""
cudaq_n9_diagnostic.py  (v9.2)
QMG N=9 深度診斷

★ v9.2 關鍵說明：
  CUDA-Q 0.7.1 的 broadcast dispatch bug：
    - 跨模組 import 的 list[float] kernel → broadcast 觸發 → RuntimeError
    - 同檔案定義的 list[float] kernel → 正常呼叫
  v9.2 的解法：MoleculeGeneratorCUDAQ 在 generator_cudaq.py 中
               使用本檔定義的 _qmg_n9_v9，不 import 跨模組 kernel。
  
  診斷腳本驗證方式：
    - 直接 sample 測試：透過 MoleculeGeneratorCUDAQ.sample_molecule()（間接呼叫本檔 kernel）
    - 不直接呼叫 cudaq.sample(imported_kernel, w, ...)（會觸發 broadcast bug）

執行方式：
  PYTHONPATH=~/sqmg_project-cudaq python cudaq_n9_diagnostic.py 2>&1 | tee n9_diag_v92.txt
"""
import math
import time
import gc
import re
import cudaq
import numpy as np

print("=" * 70)
print("QMG N=9 Kernel 深度診斷 (v9.2)")
print("=" * 70)

# ── Pre-check ─────────────────────────────────────────────────────────────────
print("\n[Pre-check] 版本與 GPU 確認")
import sys
try:
    ver = cudaq.__version__
    m = re.search(r'(\d+\.\d+\.\d+)', ver)
    print(f"  CUDA-Q version : {m.group(1) if m else ver}")
    # ★ 修正：正確解析 target name（target string 含 newline）
    targets_raw = [str(t) for t in cudaq.get_targets()]
    target_names = [t.split('\n')[0].replace('Target','').strip() for t in targets_raw]
    print(f"  Target names   : {target_names}")
    gpu_ok = 'nvidia' in target_names
    print(f"  GPU (nvidia)   : {'✓ 可用' if gpu_ok else '✗ 不可用'}")
    try:
        n = cudaq.num_available_gpus()
        print(f"  GPU count      : {n}")
    except Exception:
        pass
except Exception as e:
    print(f"  ✗ {e}"); sys.exit(1)

try:
    from qmg.generator_cudaq import MoleculeGeneratorCUDAQ, _N9_ALL_REGS, _qmg_n9_v9
    from qmg.utils.weight_generator import ConditionalWeightsGenerator
    assert len(_N9_ALL_REGS) == 90
    print(f"  _N9_ALL_REGS   : {len(_N9_ALL_REGS)} 個暫存器 ✓")
except Exception as e:
    print(f"  ✗ import 失敗：{e}"); sys.exit(1)

# ── Test A：局部小型 kernel（確認 list[float] + mid-circuit 基本功能）──────────
print("\n[Test A] 局部 list[float] + mid-circuit kernel（確認 CUDA-Q 0.7.1 基本功能）")

@cudaq.kernel
def _local_test_kernel(params: list[float]):
    q = cudaq.qvector(6)
    ry(math.pi * params[0], q[0])
    ry(math.pi * params[1], q[1])
    x.ctrl(q[0], q[1])
    ry.ctrl(math.pi * params[2], q[1], q[2])
    a0 = mz(q[0]); a1 = mz(q[1]); a2 = mz(q[2])
    if a0 or a1:
        ry(math.pi * params[3], q[3]); x(q[4])
        x.ctrl(q[3], q[4])
        ry.ctrl(math.pi * params[4], q[3], q[4])
    b0 = mz(q[3]); b1 = mz(q[4]); b2 = mz(q[5])

cudaq.set_target("qpp-cpu")
try:
    r = cudaq.sample(_local_test_kernel, [0.5, 0.3, 0.2, 0.8, 0.4], shots_count=20)
    print(f"  ✓ 成功：{dict(list(r.items())[:3])}")
    # 命名暫存器驗證
    for reg in ['a0', 'b0']:
        d = r.get_sequential_data(reg)
        print(f"  register '{reg}': len={len(d)}, shots={d[:5]}")
    print("  ✓ mid-circuit + 命名暫存器 均正常")
except Exception as e:
    print(f"  ✗ 失敗：{e}")

# ── Test B：說明 broadcast bug ───────────────────────────────────────────────
print("\n[Test B] CUDA-Q 0.7.1 broadcast bug 說明（預期失敗，僅記錄）")
print("  注意：cudaq.sample(imported_kernel, list, ...) 在 CUDA-Q 0.7.1 中，")
print("        若 kernel 從其他模組 import，list 會被誤解為 broadcast set → TypeError")
print("  v9.2 解法：kernel 直接定義在 generator_cudaq.py 中，MoleculeGeneratorCUDAQ")
print("              內部呼叫同檔 kernel → broadcast 不觸發")
try:
    # 故意測試：從別的模組 import 的 _qmg_n9_v9 呼叫
    w = [0.5] * 134
    r = cudaq.sample(_qmg_n9_v9, w, shots_count=5)
    print(f"  結果：成功（{dict(list(r.items())[:2])}）— 此版本 broadcast bug 已修正或行為不同")
except Exception as e:
    print(f"  結果：失敗（預期）— {str(e)[:80]}")
    print("  → 確認：v9.2 正確透過 MoleculeGeneratorCUDAQ 內部呼叫規避此問題")

# ── Test C：MoleculeGeneratorCUDAQ 端到端（qpp-cpu，200 shots）──────────────
print("\n[Test C] MoleculeGeneratorCUDAQ 端到端（qpp-cpu，200 shots）")
try:
    cwg = ConditionalWeightsGenerator(9, smarts=None)
    w   = cwg.generate_conditional_random_weights(random_seed=42)
    assert len(w) == 134, f"weight 長度錯誤：{len(w)}"
    print(f"  weight_generator OK: len={len(w)}, range=[{w.min():.3f}, {w.max():.3f}]")

    gen = MoleculeGeneratorCUDAQ(9, all_weight_vector=w, backend_name="cudaq_qpp")
    t0  = time.time()
    sd, v, u = gen.sample_molecule(200)
    elapsed  = time.time() - t0
    valid    = [k for k in sd if k and k != "None"]

    print(f"  V={v:.3f}  U={u:.3f}  V×U={v*u:.4f}  ({elapsed:.1f}s)")
    print(f"  有效分子種數：{len(valid)}")
    if valid:
        print(f"  範例 SMILES：{valid[:5]}")

    if v > 0.3 and u > 0.3:
        print("  ✓ v9.2 CPU 正常，V>0.3 且 U>0.3 ✓")
    elif v > 0:
        print("  ⚠ 有結果但 V 或 U 偏低（參數 seed=42 初始值，optimizer 會調整）")
    else:
        print("  ✗ validity=0，請回報 CUDA-Q 版本及暫存器診斷結果")
except Exception as e:
    print(f"  ✗ 失敗：{e}")
    import traceback; traceback.print_exc()

# ── Test D：GPU 測試（nvidia，100 shots）──────────────────────────────────
print("\n[Test D] MoleculeGeneratorCUDAQ GPU 測試（nvidia，100 shots）")
try:
    gen_gpu = MoleculeGeneratorCUDAQ(9, all_weight_vector=w, backend_name="cudaq_nvidia")
    t0      = time.time()
    sd_gpu, v_gpu, u_gpu = gen_gpu.sample_molecule(100)
    elapsed = time.time() - t0
    valid_gpu = [k for k in sd_gpu if k and k != "None"]

    print(f"  V={v_gpu:.3f}  U={u_gpu:.3f}  V×U={v_gpu*u_gpu:.4f}  ({elapsed:.1f}s)")
    print(f"  有效分子種數：{len(valid_gpu)}")
    if elapsed < 30:
        print(f"  ✓ GPU 加速正常（{elapsed:.1f}s < 30s）")
    else:
        print(f"  ⚠ 速度偏慢（{elapsed:.1f}s），可能仍在 CPU 執行")
    if v_gpu > 0:
        print(f"  ✓ GPU 有效結果 ✓")
    else:
        print(f"  ✗ GPU validity=0")
except Exception as e:
    print(f"  ✗ GPU 失敗：{e}")

# ── Test E：10000 shots CPU 速度預估 ─────────────────────────────────────
print("\n[Test E] 10000 shots 速度預估（qpp-cpu，50 shots 計時）")
try:
    cudaq.set_target("qpp-cpu")
    gen_spd = MoleculeGeneratorCUDAQ(9, all_weight_vector=w, backend_name="cudaq_qpp")
    t0 = time.time()
    gen_spd.sample_molecule(50)
    rate = (time.time() - t0) / 50  # s/shot
    est_10k = rate * 10000
    print(f"  CPU: {rate*1000:.1f} ms/shot → 10000 shots ≈ {est_10k:.0f}s ({est_10k/60:.1f} min)")
    print(f"  完整實驗 520 evals: ≈ {est_10k*520/3600:.0f}h → 強烈建議用 GPU")
except Exception as e:
    print(f"  ✗ {e}")

print("\n" + "=" * 70)
print("診斷完成 (v9.2)")
print("=" * 70)