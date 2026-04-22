#!/bin/bash
# ==============================================================================
# cutn-qmg_mpi_8g.slurm
# QMG AE-SOQPSO + MPI 並行實驗 — DGX V100-SXM2-16GB × 8
# 融合：SQMG tensornet 後端 + AE-QTS 調和加權 + MPI 並行
#
# 參考：cutn-qsvm_check_sub_n1g1.slurm（原始 SLURM 設定）
# ==============================================================================
#
# 提交：sbatch cutn-qmg_mpi_8g.slurm
# 監控：squeue -u $USER
#       tail -f results_mpi/unconditional_9_ae_mpi.log
#
# 預估時間（8 卡 V100，tensornet 後端）：
#   每次評估：~0.5s（tensornet）vs ~105s（cuStateVec）
#   50 粒子 / 8 GPU = 7 粒子/rank → 7×0.5s = 3.5s/iter
#   40 迭代 → 總計約 40×3.5s + 初始化 ≈ 5分鐘（！）
#
#   若 tensornet 速度不如預期（fallback 到 nvidia）：
#   每次評估：~105s，50/8=7粒子 → 7×105s=735s/iter × 40iter ≈ 8h
# ==============================================================================

#SBATCH -J qmg.ae.mpi.tn        # 作業名稱
#SBATCH -p nchc                 # 佇列（與原 slurm 一致）
#SBATCH --nodes=1               # 單節點 DGX（8 張 V100 在同一機器）
#SBATCH --ntasks-per-node=8     # 8 個 MPI rank = 8 張 GPU
#SBATCH --cpus-per-task=4       # 每 rank 4 CPU（RDKit 處理用）
#SBATCH --gres=gpu:8            # 申請 8 張 GPU
#SBATCH --mem-bind=no           # 與原 slurm 設定一致

# ── 環境初始化（與 cutn-qsvm_check_sub_n1g1.slurm 完全一致）────────────────
ml purge
ml cuq/12

srun hostname

source /beegfs/_venv/cuq23cu12/bin/activate

# ── GPU 設定 ────────────────────────────────────────────────────────────────
# 讓全部 8 張 GPU 可見，run_qpso_qmg_mpi.py 根據 RANK 自動選擇
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ── CUDA-Q 設定 ─────────────────────────────────────────────────────────────
export CUDAQ_REQUIRE_CUDA=1          # 強制 GPU，不 fallback 到 CPU

# ── Python 路徑 ─────────────────────────────────────────────────────────────
export PYTHONPATH=/beegfs/home/${USER}/sqmg_project-cudaq:${PYTHONPATH}

# ── mpi4py 設定（確保使用正確的 MPI 實作）───────────────────────────────────
export MPICC=$(which mpicc)

# ==============================================================================
# ★ 方案 A：tensornet 後端（推薦首選，SQMG 論文推薦，速度最快）
# 50 粒子 × 41 輪 = 2050 次評估
# 估計時間：~5 分鐘（tensornet）或 ~8h（fallback nvidia）
# ==============================================================================
mpirun -n 8 python run_qpso_qmg_mpi.py \
    --backend           cudaq_tensornet \
    --num_heavy_atom    9               \
    --num_sample        10000           \
    --particles         50              \
    --iterations        40              \
    --alpha_max         1.2             \
    --alpha_min         0.4             \
    --mutation_prob     0.15            \
    --stagnation_limit  8               \
    --reinit_fraction   0.20            \
    --ae_weighting                      \
    --pair_interval     5               \
    --rotate_factor     0.01            \
    --seed              42              \
    --task_name         unconditional_9_ae_mpi_tn \
    --data_dir          results_mpi

echo "---- Walltime: ${SECONDS} sec."

# ==============================================================================
# ★ 方案 B：nvidia（cuStateVec）後端（若 tensornet 遇到問題的備用方案）
# ==============================================================================
# mpirun -n 8 python run_qpso_qmg_mpi.py \
#     --backend           cudaq_nvidia    \
#     --num_heavy_atom    9               \
#     --num_sample        10000           \
#     --particles         50              \
#     --iterations        40              \
#     --alpha_max         1.2             \
#     --alpha_min         0.4             \
#     --mutation_prob     0.15            \
#     --stagnation_limit  8               \
#     --ae_weighting                      \
#     --pair_interval     5               \
#     --seed              42              \
#     --task_name         unconditional_9_ae_mpi_nv \
#     --data_dir          results_mpi_nv
# echo "---- Walltime (nvidia): ${SECONDS} sec."

# ==============================================================================
# ★ 方案 C：大規模（tensornet，80 粒子 × 100 迭代）
# 對應論文目標 V×U ≥ 0.8834，評估次數 8080 次
# 估計時間：~20 分鐘（tensornet）
# ==============================================================================
# mpirun -n 8 python run_qpso_qmg_mpi.py \
#     --backend           cudaq_tensornet \
#     --num_heavy_atom    9               \
#     --num_sample        10000           \
#     --particles         80              \
#     --iterations        100             \
#     --alpha_max         1.2             \
#     --alpha_min         0.4             \
#     --mutation_prob     0.15            \
#     --stagnation_limit  10              \
#     --reinit_fraction   0.15            \
#     --ae_weighting                      \
#     --pair_interval     5               \
#     --rotate_factor     0.008           \
#     --seed              42              \
#     --task_name         unconditional_9_ae_mpi_large \
#     --data_dir          results_mpi_large
# echo "---- Walltime (large): ${SECONDS} sec."

# ==============================================================================
# 快速功能驗證（在正式提交前，先在互動 session 跑）：
#   srun --nodes=1 --ntasks-per-node=8 --gres=gpu:8 --pty bash -c \
#     "source /beegfs/_venv/cuq23cu12/bin/activate && \
#      export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
#      export PYTHONPATH=/beegfs/home/${USER}/sqmg_project-cudaq && \
#      mpirun -n 8 python run_qpso_qmg_mpi.py \
#          --backend cudaq_tensornet \
#          --particles 8 --iterations 2 --num_sample 5 \
#          --task_name sanity_mpi --data_dir results_sanity"
# ==============================================================================
