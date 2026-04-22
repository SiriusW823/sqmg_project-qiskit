"""
==============================================================================
qpso_optimizer_ae.py
AE-SOQPSO — Amplitude-Ensemble Single-Objective Quantum PSO
融合 AE-QTS（arXiv:2311.12867v2）的調和加權與配對更新機制
支援單粒子（subprocess）與批次（MPI）兩種評估模式
==============================================================================

核心改進：

  【1】AE-QTS 調和加權 mbest
      QTS 對排序後的 N/2 個 best-worst 配對，分別以 Δθ/k（k=1..N/2）
      做量子旋轉，核心思想是「按排名衰減的影響力」。
      映射到連續 QPSO：
        w_k = 1/k（k=1..M），歸一化後加權 pbest 計算 mbest
        第 1 名的 pbest 影響力是第 M 名的 M 倍（vs 均等均值）
      效果：勢阱中心偏向歷史高品質區域，加速收斂

  【2】AE-QTS Best-Worst 配對微調
      每 pair_interval 迭代做一次：
        Best  粒子：以 rotate_factor/k 幅度向 gbest 拉近（利用）
        Worst 粒子：以 rotate_factor/k 做 Cauchy 擾動（探索）
      對應 AE-QTS 中利用全族群資訊的旋轉更新

  【3】批次評估介面（MPI 並行支援）
      batch_evaluate_fn(positions: np.ndarray) → list[(v, u)]
        - 輸入：形狀 (M, D) 的位置矩陣
        - 輸出：M 個 (validity, uniqueness) 元組
      MPI 版本在此函式內部做 scatter/gather，對 optimizer 完全透明

  【4】雙目標解耦監控
      分別追蹤 V_best_ever 和 U_best_ever，
      診斷「V 高 U 低」（模型坍縮）或「V 低 U 高」（無效多樣）情況

放置位置：qpso_optimizer_ae.py（專案根目錄）

參考文獻：
  [1] Tseng et al., AE-QTS, arXiv:2311.12867v2, 2024
  [2] Sun et al., QPSO, CEC 2012
  [3] Chen et al., QMG, JCTC 2025
  [4] Xiao et al., SQMG, arXiv:2604.13877v1, 2026
==============================================================================
"""
from __future__ import annotations

import csv
import logging
import math
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class AESOQPSOOptimizer:
    """
    Amplitude-Ensemble Single-Objective QPSO。

    評估模式（擇一傳入）：
      evaluate_fn:       (pos: np.ndarray[D]) → (validity, uniqueness)
                         單粒子評估，逐粒子呼叫（subprocess 模式）
      batch_evaluate_fn: (positions: np.ndarray[M,D]) → list[(v,u)]
                         批次評估，一次評估全部粒子（MPI 並行模式）

    Args:
        n_params:          參數維度 D（N=9 時為 134）
        n_particles:       粒子數 M
        max_iterations:    最大迭代次數 T
        evaluate_fn:       單粒子適應度函式（與 batch_evaluate_fn 擇一）
        batch_evaluate_fn: 批次適應度函式（MPI 模式，優先級高於 evaluate_fn）
        logger:            logging.Logger（MPI 時只有 rank 0 傳入有效 logger）
        ae_weighting:      是否使用 AE-QTS 調和加權 mbest（預設 True）
        pair_interval:     每幾個迭代做一次 AE 配對更新（0=關閉）
        rotate_factor:     AE 配對更新基礎幅度（對應 AE-QTS 的 Δθ）
        seed:              隨機種子
        alpha_max/min:     QPSO 收斂係數範圍
        stagnation_limit:  連續多少迭代無進展觸發重初始化
        reinit_fraction:   重初始化的粒子比例
        mutation_prob:     Cauchy 變異機率（per-particle per-iteration）
        mutation_scale:    Cauchy 變異幅度
        alpha_perturb_std: α 排程的隨機擾動標準差
        alpha_stag_boost:  停滯時 α 的額外增量
        data_dir:          結果輸出目錄
        task_name:         實驗名稱
    """

    def __init__(
        self,
        n_params:            int,
        n_particles:         int,
        max_iterations:      int,
        logger:              logging.Logger,
        evaluate_fn:         Optional[Callable[[np.ndarray], Tuple[float, float]]] = None,
        batch_evaluate_fn:   Optional[Callable[[np.ndarray], List[Tuple[float, float]]]] = None,
        seed:                int   = 42,
        alpha_max:           float = 1.2,
        alpha_min:           float = 0.4,
        data_dir:            str   = "results_ae_qpso",
        task_name:           str   = "unconditional_9_ae_qpso",
        stagnation_limit:    int   = 8,
        reinit_fraction:     float = 0.20,
        mutation_prob:       float = 0.15,
        mutation_scale:      float = 0.15,
        alpha_perturb_std:   float = 0.04,
        alpha_stag_boost:    float = 0.20,
        ae_weighting:        bool  = True,
        pair_interval:       int   = 5,
        rotate_factor:       float = 0.01,
    ):
        if evaluate_fn is None and batch_evaluate_fn is None:
            raise ValueError("必須提供 evaluate_fn 或 batch_evaluate_fn 其中一個。")

        self.D                 = n_params
        self.M                 = n_particles
        self.T                 = max_iterations
        self.logger            = logger
        self.evaluate_fn       = evaluate_fn
        self.batch_evaluate_fn = batch_evaluate_fn
        self.alpha_max         = alpha_max
        self.alpha_min         = alpha_min
        self.data_dir          = data_dir
        self.task_name         = task_name
        self.stagnation_limit  = stagnation_limit
        self.reinit_fraction   = reinit_fraction
        self.mutation_prob     = mutation_prob
        self.mutation_scale    = mutation_scale
        self.alpha_perturb_std = alpha_perturb_std
        self.alpha_stag_boost  = alpha_stag_boost
        self.ae_weighting      = ae_weighting
        self.pair_interval     = pair_interval
        self.rotate_factor     = rotate_factor

        # 參數空間邊界：全部 [0, 1]
        self.lb = np.zeros(self.D, dtype=np.float64)
        self.ub = np.ones(self.D,  dtype=np.float64)
        self._mut_range = mutation_scale * (self.ub - self.lb)

        # 粒子群初始化
        self.rng       = np.random.default_rng(seed)
        self.positions = self._rand_pos(self.M)
        self.pbest     = self.positions.copy()
        self.pbest_fit = np.full(self.M, -np.inf)

        # 全局最優
        self.gbest_pos  = None
        self.gbest_fit  = -np.inf
        self.gbest_val  = 0.0
        self.gbest_uniq = 0.0
        # 雙目標獨立追蹤（AE 多目標視角）
        self._best_v_ever = 0.0
        self._best_u_ever = 0.0

        # 運行統計
        self.history:          List[Dict] = []
        self._stag_counter     = 0
        self._prev_best        = -np.inf
        self._global_eval_cnt  = 0
        self._total_reinits    = 0
        self._total_mutations  = 0
        self._total_ae_updates = 0

        os.makedirs(data_dir, exist_ok=True)
        self._csv_path = os.path.join(data_dir, f"{task_name}.csv")
        self._init_csv()

    # ================================================================
    # 基礎工具
    # ================================================================

    def _rand_pos(self, n: int) -> np.ndarray:
        return self.lb + self.rng.random((n, self.D)) * (self.ub - self.lb)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def _get_alpha(self, t: int) -> float:
        """Cosine annealing + 隨機擾動 + 停滯 boost。"""
        progress = t / max(self.T - 1, 1)
        base = (self.alpha_min
                + 0.5 * (self.alpha_max - self.alpha_min)
                * (1.0 + math.cos(math.pi * progress)))
        perturb = self.rng.normal(0.0, self.alpha_perturb_std)
        boost   = self.alpha_stag_boost if self._stag_counter >= self.stagnation_limit else 0.0
        return float(np.clip(base + perturb + boost,
                             self.alpha_min,
                             self.alpha_max + self.alpha_stag_boost))

    def _update_pos_single(self, x, pbest_i, gbest, mbest, alpha) -> np.ndarray:
        """Delta 勢阱位置更新（Sun et al. 2012 Eq.12）。"""
        phi       = self.rng.uniform(0.0, 1.0, size=self.D)
        attractor = phi * pbest_i + (1.0 - phi) * gbest
        u         = np.maximum(self.rng.uniform(0.0, 1.0, size=self.D), 1e-10)
        step      = alpha * np.abs(mbest - x) * np.log(1.0 / u)
        sign      = np.where(self.rng.uniform(0.0, 1.0, size=self.D) < 0.5, 1.0, -1.0)
        return self._clip(attractor + sign * step)

    def _cauchy_mutation(self, x: np.ndarray) -> np.ndarray:
        """Cauchy 重尾變異，用於跳脫局部最優。"""
        x_mut = x.copy()
        n_dim = max(1, int(self.D * self.rng.uniform(0.15, 0.35)))
        dims  = self.rng.choice(self.D, size=n_dim, replace=False)
        noise = self.rng.standard_cauchy(size=n_dim) * self._mut_range[dims]
        x_mut[dims] += noise
        return self._clip(x_mut)

    # ================================================================
    # ★ AE-QTS 核心機制
    # ================================================================

    def _ae_weighted_mbest(self) -> np.ndarray:
        """
        AE-QTS Amplitude-Ensemble 調和加權 mbest。

        AE-QTS 演算法第 k 對的旋轉幅度為 Δθ/k，
        意即第 1 對（最佳-最差）影響力最大，第 M/2 對最小。
        本方法將此「按排名衰減的影響力」映射到 QPSO 的 mbest 計算：
          w_k = 1/k（k=1..M），歸一化後加權排名第 k 的粒子之 pbest。

        相較均等均值（mbest = mean(pbest)）：
          - 最佳粒子的 pbest 對 mbest 影響力是最差粒子的 H_M 倍
            （H_M = 1 + 1/2 + ... + 1/M ≈ ln(M)，調和數）
          - 使 Delta 勢阱中心向高品質區域偏移
          - AE-QTS 論文在 knapsack 問題上平均提升 20%，最高 30%

        Returns:
            weighted_mbest: (D,) 加權後的均值位置向量
        """
        sorted_idx = np.argsort(self.pbest_fit)[::-1]  # fitness 由高到低
        # 調和數序列：1, 1/2, 1/3, ..., 1/M
        harmonic_w = np.array([1.0 / (k + 1) for k in range(self.M)])
        harmonic_w /= harmonic_w.sum()   # 歸一化
        return np.sum(
            self.pbest[sorted_idx] * harmonic_w[:, np.newaxis], axis=0
        )

    def _ae_paired_update(self, alpha: float):
        """
        AE-QTS Best-Worst 配對更新，連續化映射。

        AE-QTS 原始：對第 k 對 (best_k, worst_k)，
          sign_k = best_k - worst_k（+1 或 -1）
          theta_update += sign_k × (Δθ / k)
        連續空間映射：
          Best_k：以 step=rotate_factor/k 向 gbest 拉近（利用高品質區域）
          Worst_k：以 step 做 Cauchy 擾動（增加探索多樣性）

        這模擬了 AE-QTS 中「用全部族群資訊調整量子振幅」的效果。
        """
        if self.gbest_pos is None:
            return
        half      = self.M // 2
        sorted_i  = np.argsort(self.pbest_fit)[::-1]

        for k in range(1, half + 1):
            best_idx  = sorted_i[k - 1]
            worst_idx = sorted_i[self.M - k]
            step      = self.rotate_factor / k

            # Best：向 gbest 方向移動
            direction = self.gbest_pos - self.positions[best_idx]
            scale     = self.rng.uniform(0.5, 1.5, size=self.D)
            self.positions[best_idx] = self._clip(
                self.positions[best_idx] + step * direction * scale
            )

            # Worst：Cauchy 重尾擾動
            noise = self.rng.standard_cauchy(size=self.D) * step * 0.5
            self.positions[worst_idx] = self._clip(
                self.positions[worst_idx] + noise
            )

        self._total_ae_updates += 1

    # ================================================================
    # 停滯偵測與重初始化
    # ================================================================

    def _update_stagnation(self, score: float):
        if score > self._prev_best + 1e-8:
            self._stag_counter = 0
        else:
            self._stag_counter += 1
        self._prev_best = score

    def _maybe_reinit(self):
        if self._stag_counter < self.stagnation_limit:
            return
        n_reinit  = max(1, int(self.M * self.reinit_fraction))
        worst_idx = np.argsort(self.pbest_fit)[:n_reinit]
        for off, idx in enumerate(worst_idx):
            if off < n_reinit // 2 or self.gbest_pos is None:
                new_pos = self._rand_pos(1)[0]
            else:
                noise   = self.rng.normal(0.0, 0.10, size=self.D) * (self.ub - self.lb)
                new_pos = self._clip(self.gbest_pos + noise)
            self.positions[idx] = new_pos
            self.pbest[idx]     = new_pos.copy()
            self.pbest_fit[idx] = -np.inf
        self._stag_counter   = 0
        self._total_reinits += 1
        self.logger.info(
            f"  [停滯偵測] 重初始化 {n_reinit} 個粒子"
            f"（累計第 {self._total_reinits} 次）"
        )

    # ================================================================
    # pbest / gbest 更新
    # ================================================================

    def _update_pbest(self, i: int, fit: float):
        if fit > self.pbest_fit[i]:
            self.pbest[i]     = self.positions[i].copy()
            self.pbest_fit[i] = fit

    def _update_gbest(self, i: int, fit: float, val: float, uniq: float):
        if fit > self.gbest_fit:
            self.gbest_pos  = self.positions[i].copy()
            self.gbest_fit  = fit
            self.gbest_val  = val
            self.gbest_uniq = uniq
            self.logger.info(
                f"  🔥 New gbest!  V={val:.4f}  U={uniq:.4f}  V×U={fit:.4f}"
                f"{'  ✓ 超越 BO 基線 0.8834!' if fit > 0.8834 else ''}"
            )
        # 雙目標獨立追蹤
        if val  > self._best_v_ever:
            self._best_v_ever = val
        if uniq > self._best_u_ever:
            self._best_u_ever = uniq

    # ================================================================
    # CSV 記錄
    # ================================================================

    _CSV_FIELDS = [
        'eval_index', 'qpso_iter', 'particle',
        'validity', 'uniqueness', 'fitness',
        'gbest_fitness', 'best_v_ever', 'best_u_ever',
        'alpha', 'stagnation', 'elapsed_s',
    ]

    def _init_csv(self):
        with open(self._csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self._CSV_FIELDS).writeheader()

    def _write_csv(self, row: dict):
        with open(self._csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self._CSV_FIELDS).writerow(row)

    def _log_eval(
        self, eval_label: int, qpso_iter: int, particle_id: int,
        v: float, u: float, f: float, alpha: float, elapsed: float,
    ):
        """記錄單次評估結果（格式與原版 log 相容）。"""
        self.logger.info(f"Iteration number: {eval_label}")
        self.logger.info(f"validity (maximize): {v:.3f}")
        self.logger.info(f"uniqueness (maximize): {u:.3f}")
        self.logger.info(
            f"  [AE-QPSO] iter={qpso_iter}  p={particle_id}  "
            f"fit={f:.4f}  gbest={self.gbest_fit:.4f}  "
            f"stag={self._stag_counter}  α={alpha:.4f}  t={elapsed:.1f}s  "
            f"[V⋆={self._best_v_ever:.3f} U⋆={self._best_u_ever:.3f}]"
        )
        self._write_csv({
            'eval_index':   eval_label,
            'qpso_iter':    qpso_iter,
            'particle':     particle_id,
            'validity':     round(v, 4),
            'uniqueness':   round(u, 4),
            'fitness':      round(f, 6),
            'gbest_fitness': round(self.gbest_fit, 6),
            'best_v_ever':  round(self._best_v_ever, 4),
            'best_u_ever':  round(self._best_u_ever, 4),
            'alpha':        round(alpha, 4),
            'stagnation':   self._stag_counter,
            'elapsed_s':    round(elapsed, 1),
        })

    # ================================================================
    # 主優化迴圈（支援 evaluate_fn 與 batch_evaluate_fn）
    # ================================================================

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        執行 AE-SOQPSO 優化。

        自動選擇評估模式：
          - 有 batch_evaluate_fn → 批次模式（MPI 並行）
          - 否則 → 逐粒子模式（subprocess 或單 GPU）
        """
        use_batch = self.batch_evaluate_fn is not None
        total_evals = self.M * (self.T + 1)

        self.logger.info("=" * 70)
        self.logger.info("AE-SOQPSO（Amplitude-Ensemble QPSO）優化啟動")
        self.logger.info(f"  粒子數 M               : {self.M}")
        self.logger.info(f"  參數維度 D              : {self.D}")
        self.logger.info(f"  最大迭代 T              : {self.T}")
        self.logger.info(f"  總評估次數              : {total_evals}")
        self.logger.info(f"  評估模式               : {'批次（MPI 並行）' if use_batch else '逐粒子（subprocess）'}")
        self.logger.info(f"  α 排程                : [{self.alpha_min}, {self.alpha_max}] cosine")
        self.logger.info(f"  AE 調和加權 mbest       : {'✓ 開啟' if self.ae_weighting else '✗ 關閉'}")
        self.logger.info(f"  AE 配對更新間隔（迭代） : {self.pair_interval if self.pair_interval > 0 else '關閉'}")
        self.logger.info(f"  rotate_factor          : {self.rotate_factor}")
        self.logger.info(f"  停滯門檻               : {self.stagnation_limit} iters")
        self.logger.info(f"  Cauchy 變異機率         : {self.mutation_prob:.0%}")
        self.logger.info(f"  BO 基線 (V×U)          : 0.8834")
        self.logger.info("=" * 70)

        # ──────────────────────────────────────────────────────────────────
        # Phase 0：初始化評估
        # ──────────────────────────────────────────────────────────────────
        self.logger.info("[Phase 0] 初始粒子評估...")
        t0 = time.time()

        if use_batch:
            batch_results = self.batch_evaluate_fn(self.positions)
            elapsed_batch = time.time() - t0
            for i, (v, u) in enumerate(batch_results):
                f = float(v) * float(u)
                self._log_eval(self._global_eval_cnt, 0, i, v, u, f,
                               self._get_alpha(0), elapsed_batch)
                self._global_eval_cnt += 1
                self._update_pbest(i, f)
                self._update_gbest(i, f, v, u)
        else:
            for i in range(self.M):
                t_i = time.time()
                v, u = self.evaluate_fn(self.positions[i])
                f    = float(v) * float(u)
                elapsed = time.time() - t_i
                self._log_eval(self._global_eval_cnt, 0, i, v, u, f,
                               self._get_alpha(0), elapsed)
                self._global_eval_cnt += 1
                self._update_pbest(i, f)
                self._update_gbest(i, f, v, u)

        self._prev_best = self.gbest_fit

        # ──────────────────────────────────────────────────────────────────
        # 主迭代
        # ──────────────────────────────────────────────────────────────────
        for t in range(self.T):
            alpha = self._get_alpha(t)

            # ── ★ AE-QTS 調和加權 mbest ───────────────────────────────────
            if self.ae_weighting:
                mbest = self._ae_weighted_mbest()
            else:
                mbest = np.mean(self.pbest, axis=0)

            gbest = self.gbest_pos if self.gbest_pos is not None else self.positions[0]

            # ── 位置更新（標準 SOQPSO）────────────────────────────────────
            for i in range(self.M):
                self.positions[i] = self._update_pos_single(
                    self.positions[i], self.pbest[i], gbest, mbest, alpha
                )
                if self.rng.random() < self.mutation_prob:
                    self.positions[i] = self._cauchy_mutation(self.positions[i])
                    self._total_mutations += 1

            # ── ★ AE-QTS 配對微調（每 pair_interval 迭代）────────────────
            if self.pair_interval > 0 and (t + 1) % self.pair_interval == 0:
                self._ae_paired_update(alpha)
                self.logger.info(
                    f"  [AE 配對更新] iter={t+1}  "
                    f"rotate_factor={self.rotate_factor:.4f}  "
                    f"累計 {self._total_ae_updates} 次"
                )

            # ── 評估（批次 or 逐粒子）─────────────────────────────────────
            iter_fits = []
            t_iter = time.time()

            if use_batch:
                batch_results = self.batch_evaluate_fn(self.positions)
                elapsed_batch = time.time() - t_iter
                for i, (v, u) in enumerate(batch_results):
                    f = float(v) * float(u)
                    iter_fits.append(f)
                    self._log_eval(self._global_eval_cnt, t + 1, i, v, u, f,
                                   alpha, elapsed_batch)
                    self._global_eval_cnt += 1
                    self._update_pbest(i, f)
                    self._update_gbest(i, f, v, u)
            else:
                for i in range(self.M):
                    t_i = time.time()
                    v, u = self.evaluate_fn(self.positions[i])
                    f    = float(v) * float(u)
                    elapsed = time.time() - t_i
                    iter_fits.append(f)
                    self._log_eval(self._global_eval_cnt, t + 1, i, v, u, f,
                                   alpha, elapsed)
                    self._global_eval_cnt += 1
                    self._update_pbest(i, f)
                    self._update_gbest(i, f, v, u)

            self._update_stagnation(self.gbest_fit)
            self._maybe_reinit()

            mean_fit = float(np.mean(iter_fits))
            max_fit  = float(np.max(iter_fits))
            self.history.append({
                'qpso_iter':        t + 1,
                'n_evals':          self._global_eval_cnt,
                'gbest_fitness':    self.gbest_fit,
                'gbest_validity':   self.gbest_val,
                'gbest_uniqueness': self.gbest_uniq,
                'best_v_ever':      self._best_v_ever,
                'best_u_ever':      self._best_u_ever,
                'mean_fitness':     mean_fit,
                'max_fitness':      max_fit,
                'alpha':            alpha,
            })
            self.logger.info(
                f"  [AE-QPSO Iter {t+1:3d}/{self.T}] "
                f"α={alpha:.3f}  "
                f"gbest={self.gbest_fit:.4f} (V={self.gbest_val:.3f} U={self.gbest_uniq:.3f})  "
                f"mean={mean_fit:.4f}  max={max_fit:.4f}  "
                f"stag={self._stag_counter}  evals={self._global_eval_cnt}  "
                f"V⋆={self._best_v_ever:.3f}  U⋆={self._best_u_ever:.3f}"
            )

        # ──────────────────────────────────────────────────────────────────
        # 最終摘要
        # ──────────────────────────────────────────────────────────────────
        self.logger.info("=" * 70)
        self.logger.info("AE-SOQPSO 優化完成")
        self.logger.info(f"  Best V×U        : {self.gbest_fit:.6f}")
        self.logger.info(f"  Best V          : {self.gbest_val:.4f}")
        self.logger.info(f"  Best U          : {self.gbest_uniq:.4f}")
        self.logger.info(f"  V_best_ever     : {self._best_v_ever:.4f}  "
                         f"（{'正常' if self._best_v_ever > 0.8 else '偏低，建議增加 chemistry constraint'}）")
        self.logger.info(f"  U_best_ever     : {self._best_u_ever:.4f}  "
                         f"（{'正常' if self._best_u_ever > 0.7 else '偏低，建議增加粒子數或迭代數'}）")
        self.logger.info(
            f"  BO Baseline     : 0.8834  "
            + ("✓ 超越基線!" if self.gbest_fit > 0.8834
               else f"✗ 差距 {0.8834 - self.gbest_fit:.4f}")
        )
        self.logger.info(f"  Total evals     : {self._global_eval_cnt}")
        self.logger.info(f"  Reinits         : {self._total_reinits}")
        self.logger.info(f"  Mutations       : {self._total_mutations}")
        self.logger.info(f"  AE pair updates : {self._total_ae_updates}")
        self.logger.info(f"  AE weighting    : {'開啟' if self.ae_weighting else '關閉'}")
        self.logger.info("=" * 70)

        best = self.gbest_pos.copy() if self.gbest_pos is not None else np.zeros(self.D)
        return best, self.gbest_fit
