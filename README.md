# sqmg_project-cudaq

**QMG Dynamic Circuit + SOQPSO Optimizer (CUDA-Q 實作版)**
基於 NVIDIA CUDA-Q 框架加速的量子分子生成實驗，並使用 SOQPSO (Quantum Particle Swarm Optimization) 進行優化。

---

## 專案定位

本專案旨在將量子分子生成模型 (Quantum Molecular Generation, QMG) 改寫為 **CUDA-Q** 版本，以利用 GPU 進行高效的量子電路模擬，並結合 **SOQPSO (Delta 勢阱 + Cauchy 變異)** 取代傳統的 Bayesian Optimization 進行參數最佳化。

| 面向 | 原版 QMG (Chen et al. 2025) | 本專案 (CUDA-Q + QPSO) |
|------|---------------------------|------------------------|
| 量子電路 | Qiskit 動態電路 | **CUDA-Q** 動態電路 (`cudaq.make_kernel`) |
| 優化器 | Bayesian Optimization | **SOQPSO** (Quantum Particle Swarm Optimization) |
| 框架 | Qiskit + Ax/BoTorch | **CUDA-Q** + 自訂 QPSO |
| 目標 | validity × uniqueness (maximize) | **相同** |
| 後端 | Qiskit Aer (CPU local) | **CUDA-Q** (支援 GPU 加速 `nvidia` target) |

---

## 資料夾結構

```text
sqmg_project-cudaq/
├── qmg/                          # 改寫為 CUDA-Q 的分子生成模型
│   ├── generator_cudaq.py        # MoleculeGeneratorCUDAQ 主類別
│   └── utils/
│       ├── build_dynamic_circuit_cudaq.py  # CUDA-Q N=9 動態電路（包含 measure & reset）
│       ├── (其他輔助模組，如 chemistry_data_processing 等)
│
├── qpso_optimizer_qmg.py         # SOQPSO 優化器實作
├── run_qpso_qmg_cudaq.py         # ★ 主入口：CUDA-Q 電路建構 + QPSO 求解
├── run_qpso_qmg_cudaq v100.py    # 支援不同設定的主入口變體
├── requirements.txt              # 環境依賴（包含 cudaq, rdkit 等）
└── README.md                     # 本說明檔
```

---

## 環境安裝

本專案強烈依賴 `cudaq`。建議建立獨立的 Python 環境：

```bash
# 建立 conda 環境
conda create -n cudaq_qmg python=3.10 -y
conda activate cudaq_qmg

# 安裝依賴套件
pip install -r requirements.txt
```

---

## 執行實驗

使用 CUDA-Q 架構並透過 CPU 或 GPU 進行量子狀態採樣與分子生成：

```bash
# 執行完整的 QPSO 結合 CUDA-Q 生成實驗
python run_qpso_qmg_cudaq.py
```

可以透過修改 `run_qpso_qmg_cudaq.py` 中的參數來調整粒子數 (`particles`) 與迭代次數 (`iterations`)。

---

## 預期輸出

執行完畢後，結果會儲存於指定的資料夾 (例如 `results_qpso/`)：
1. **日誌檔 (.log)**：記錄 QPSO 每次迭代的最佳適應度 (V×U 分數)。
2. **CSV 檔 (.csv)**：記錄每一代每個粒子的詳細評估指標 (Validity, Uniqueness, V×U)。
3. **最佳參數矩陣 (.npy)**：優化收斂後得到的最佳量子電路參數。

