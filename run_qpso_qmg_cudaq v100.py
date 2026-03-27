"""
==============================================================================
run_qpso_qmg_cudaq_v100.py
8× V100-16GB GPU 最佳化設定版（對應論文 V×U = 0.8834）
==============================================================================

此檔為 run_qpso_qmg_cudaq.py 的超參數最佳化版本。
電路邏輯、Logger 格式、QPSO 演算法完全不變，
只調整 --particles、--iterations、--backend 參數。

論文設定（BO 基線，供對齊）：
  總評估次數 ≈ 355（BO 達到峰值時）
  方法：GP-EI，30 個 Sobol 初始點
  最佳結果：V=0.955, U=0.925, V×U=0.8834

QPSO 策略（8× V100 對齊論文）：
  總評估次數 = M × (T+1)，建議 400~600 evals
  ────────────────────────────────────────
  【方案 A】穩健版（推薦首選）
    M=20 粒子, T=25 迭代 → 520 evals
    執行指令：見下方

  【方案 B】精細版（更多評估，較慢）
    M=15 粒子, T=40 迭代 → 615 evals

  【方案 C】快速驗證版（功能確認用）
    M=5 粒子,  T=10 迭代 → 55 evals（~10 分鐘）
==============================================================================
"""

# ===========================================================================
# 執行指令
# ===========================================================================

COMMANDS = """
# ── 方案 A：穩健版（推薦，對齊論文評估數）───────────────────────────────────
python run_qpso_qmg_cudaq.py \\
    --backend       cudaq_nvidia \\
    --particles     20           \\
    --iterations    25           \\
    --num_sample    10000        \\
    --num_heavy_atom 9           \\
    --alpha_max     1.2          \\
    --alpha_min     0.4          \\
    --mutation_prob 0.10         \\
    --stagnation_limit 12        \\
    --seed          42           \\
    --task_name     unconditional_9_qpso_cudaq_A \\
    --data_dir      results_cudaq_A

# ── 方案 B：精細版（更多粒子，更好覆蓋 134D 空間）──────────────────────────
python run_qpso_qmg_cudaq.py \\
    --backend       cudaq_nvidia \\
    --particles     15           \\
    --iterations    40           \\
    --num_sample    10000        \\
    --alpha_max     1.2          \\
    --alpha_min     0.4          \\
    --mutation_prob 0.10         \\
    --stagnation_limit 15        \\
    --seed          42           \\
    --task_name     unconditional_9_qpso_cudaq_B \\
    --data_dir      results_cudaq_B

# ── 快速功能驗證（5 分鐘內確認程式碼正確）──────────────────────────────────
python run_qpso_qmg_cudaq.py \\
    --backend       cudaq_nvidia \\
    --particles     5            \\
    --iterations    10           \\
    --num_sample    1000         \\
    --seed          42           \\
    --task_name     sanity_check \\
    --data_dir      results_sanity
"""

# ===========================================================================
# 參數調整說明
# ===========================================================================

PARAM_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════╗
║              QPSO 超參數 vs 論文目標 V×U ≥ 0.8834 對照表               ║
╠══════════════════════════╦═══════════════╦═══════════════════════════════╣
║ 參數                     ║ 原設定（M=5） ║ 建議設定（對齊論文）          ║
╠══════════════════════════╬═══════════════╬═══════════════════════════════╣
║ --particles (M)          ║ 5             ║ 20（134D 空間最低有效覆蓋）   ║
║ --iterations (T)         ║ 120           ║ 25（配合 M=20 維持總 evals）  ║
║ Total evals = M*(T+1)    ║ 605           ║ 520（接近論文 355 evals）     ║
║ --alpha_max              ║ 1.2           ║ 1.2（不變）                   ║
║ --alpha_min              ║ 0.4           ║ 0.4（不變）                   ║
║ --mutation_prob          ║ 0.12          ║ 0.10（降低避免破壞收斂）       ║
║ --stagnation_limit       ║ 8             ║ 12（避免過早重初始化）         ║
║ --num_sample             ║ 10000         ║ 10000（對齊論文）             ║
║ --backend                ║ cudaq_qpp     ║ cudaq_nvidia（V100 GPU）      ║
╚══════════════════════════╩═══════════════╩═══════════════════════════════╝

重要說明：
  1. M 從 5 → 20 是最關鍵的改動。
     論文 BO 等效在 134D 空間有全局代理模型，M=5 的 QPSO 完全無法比擬。
     M=20 讓初始 Phase 0 有 20 個探索點，對 134D 參數空間才有基本覆蓋。

  2. stagnation_limit 從 8 → 12：
     原設定 8 代表 8 次 QPSO 迭代（M=5 時 = 40 次評估）無進展即重初始化。
     M=20 時，每次迭代 = 20 次評估，stagnation_limit=12 = 240 次評估無進展
     才觸發重初始化，更合理。

  3. 8× V100 的 backend 選擇：
     - cudaq_nvidia：單 GPU 模擬（最穩定，推薦先用此確認結果）
     - cudaq_mqpu：自動將 10,000 shots 分配到 8 張 V100 並行
       理論上速度提升 ~8x，但每次評估的電路重複執行
       建議：確認 cudaq_nvidia 結果正確後再嘗試 cudaq_mqpu
"""

# ===========================================================================
# generator_cudaq.py 的 multi-GPU 設定說明
# ===========================================================================

MULTIGPU_PATCH = """
# 在 generator_cudaq.py 的 _CUDAQ_TARGET_MAP 中，
# "cudaq_mqpu" 已正確對應到 "nvidia-mqpu"。
# 只需執行時改 --backend cudaq_mqpu 即可。

# 若 cudaq_mqpu 在你的系統上找不到，可用以下方式確認：
python3 -c "import cudaq; print(cudaq.get_targets())"

# 預期輸出應包含：
#   qpp-cpu        （CPU 模擬）
#   nvidia         （單 GPU，cuStateVec）
#   nvidia-mqpu    （多 GPU）
"""

# ===========================================================================
# V100 compute capability 確認
# ===========================================================================

GPU_CHECK = """
# 執行前確認 V100 被 CUDA-Q 正確識別：
python3 -c "
import cudaq
cudaq.set_target('nvidia')
print('GPU target OK')
print('Available targets:', cudaq.get_targets())
"

# 若出現 cuStateVec 錯誤，確認 CUDA 版本：
nvidia-smi  # V100 應顯示 CUDA Version >= 11.8
nvcc --version

# CUDA-Q GPU 版安裝（選一）：
pip install cuda-quantum-cu11   # CUDA 11.x
pip install cuda-quantum-cu12   # CUDA 12.x
"""

if __name__ == "__main__":
    print("=== 8× V100 CUDA-Q 執行指令 ===")
    print(COMMANDS)
    print("=== 參數設定說明 ===")
    print(PARAM_GUIDE)
    print("=== GPU 確認指令 ===")
    print(GPU_CHECK)