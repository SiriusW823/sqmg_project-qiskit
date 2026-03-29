# sqmg_project-cudaq

**QMG Dynamic Circuit + SOQPSO Optimizer (CUDA-Q 實作版)**
基於 NVIDIA CUDA-Q 框架加速的量子分子生成實驗，並使用 SOQPSO (Quantum Particle Swarm Optimization) 進行優化。

---

## 專案定位

本專案將量子分子生成模型 (Quantum Molecular Generation, QMG) 從原有的 Qiskit 實驗改寫為 **CUDA-Q** 版本，從而利用 GPU 後端進行高效率的量子電路模擬。同時結合自訂的 **SOQPSO (Delta 勢阱 + Cauchy 變異)** 取代傳統的貝葉斯優化 (Bayesian Optimization)，加速量子電路參數搜尋。

| 面向 | 原版 QMG (Chen et al. 2025) | 本專案 (CUDA-Q + SOQPSO) |
|------|---------------------------|------------------------|
| 量子電路 | Qiskit 動態電路 (DynamicCircuitBuilder) | **CUDA-Q** 動態電路 (`cudaq.make_kernel`) |
| 優化器 | Bayesian Optimization (Ax/BoTorch) | **SOQPSO** (Quantum Particle Swarm Optimization) |
| 框架 | Qiskit | **CUDA-Q** + 原生 Python 程式實作 |
| 目標 | validity × uniqueness (maximize) | **相同** |
| 後端模擬器 | Qiskit Aer (CPU local) | **CUDA-Q** （支援 `nvidia` 或 `nvidia-mgpu` 標靶）|

---

## 資料夾結構

```text
sqmg_project-cudaq/
├── qmg/                          # 將分子生成流程適配 CUDA-Q 的核心模組
│   ├── __init__.py
│   ├── generator_cudaq.py        # MoleculeGeneratorCUDAQ 主類別處理模型生成邏輯
│   └── utils/
│       ├── build_dynamic_circuit_cudaq.py  # 建立 N=9 的 CUDA-Q 動態電路（包含測量及重置）
│       ├── (其他依賴模組，例如 chemistry_data_processing 與 fitness_calculator 將在此專案外部支援或被整合)
│
├── qpso_optimizer_qmg.py         # SOQPSO 優化器實作 (管理群體適應度與迭代更新)
├── run_qpso_qmg_cudaq.py         # ★ 主程式入口：直接組合 CUDA-Q 電路與 SOQPSO
├── run_qpso_qmg_cudaq v100.py    # 主程式的備用實驗腳本變體
├── requirements.txt              # 環境依賴檔案 (cudaq, rdkit 等)
└── README.md                     # 本檔案
```

---

## 環境配置

此專案強烈依賴 `cudaq` 和化學開源庫 `rdkit`。建議設置獨立的 Conda 虛擬環境：

```bash
# 建立並啟動 conda 環境
conda create -n cudaq_qmg python=3.10 -y
conda activate cudaq_qmg

# 安裝所需依賴 (包含 CUDA-Q, RDKit, NumPy, Pandas, 等等)
pip install -r requirements.txt
```

---

## 執行實驗

本專案透過 `run_qpso_qmg_cudaq.py` 進行量子狀態採樣與 QPSO 最佳化尋優。您可以透過設定 CLI 參數快速調整實驗規模。

**標準執行範例**：
```bash
# 跑 5 個粒子、120 次迭代 (對應約 600 次的評估總量)
python run_qpso_qmg_cudaq.py --particles 5 --iterations 120 --task_name "cudaq_v1"
```

**快速驗證測試**：
```bash
python run_qpso_qmg_cudaq.py --particles 3 --iterations 5 --task_name "debug_test"
```

---

## 預期輸出

在您啟動優化腳本後，系統預設會在 `results_qpso/` (或者您帶入的 `--data_dir` 指定資料夾) 產出對應的模型記錄檔，例如：
1. **[TaskName].log**：記錄 QPSO 每一迭代最佳解的詳細歷程 (Best V×U Score 等)。
2. **[TaskName].csv**：記錄每個粒子各個世代的完整適應度指標 (Validity, Uniqueness) 與位置訊息。
3. **[TaskName]_best_params.npy**：記錄最終尋得的最佳量子電路映射參數陣列，可供後續直接重現高穩定性分子。

