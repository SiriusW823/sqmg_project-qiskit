"""
==============================================================================
qpso_optimizer_qmg.py
SOQPSO — 單目標量子粒子群優化，適配 QMG Qiskit 電路介面
==============================================================================

設計原則：
  1. 完全不依賴 CUDA-Q；參數空間為 [0,1]^D，對應 QMG weight convention。
  2. fitness = validity × uniqueness，maximize。
  3. 每個粒子評估對應 log 的 "Iteration number: X"，
     格式與 unconditional_9.log 完全一致，方便直接比較。
  4. Chemistry constraint 由 run_qpso_qmg.py 的 evaluate_fn 套用
     （透過 cwg.apply_chemistry_constraint），optimizer 本身不感知電路細節。

核心演算法（Sun et al. 2012, Eq. 12）：
  mbest_d  = mean_i(pbest_{i,d})
  p_{i,d}  = φ·pbest_{i,d} + (1−φ)·gbest_d,   φ ~ U(0,1)
  x_{i,d}  = p ± α·|mbest_d − x|·ln(1/u),     u ~ U(0,1)
  α(t)     = α_min + 0.5·(α_max−α_min)·(1+cos(πt/T))
==============================================================================
"""
from __future__ import annotations

import csv
import logging
import math
import os
import time
from typing import Callable, Dict, List, Tuple

import numpy as np


class QMGSOQPSOOptimizer:
    """
    單目標 SOQPSO，用於最大化 validity × uniqueness。

    Args:
        n_params:          參數維度（N=9 無條件生成時為 134）
        n_particles:       粒子數 M
        max_iterations:    最大 QPSO 迭代次數 T
        evaluate_fn:       適應度函式 (params) → (validity: float, uniqueness: float)
        logger:            logging.Logger 實例
        seed:              隨機種子
        alpha_max/min:     收斂係數上下界
        data_dir:          結果輸出目錄
        task_name:         實驗名稱（對應 BO 的 task_name）
        stagnation_limit:  停滯偵測門檻（單位：QPSO 迭代次數）
        reinit_fraction:   停滯時重初始化的粒子比例
        mutation_prob:     Cauchy 變異機率（per-particle per-iteration）
        mutation_scale:    Cauchy 變異幅度（相對於 [0,1] 的比例）
        alpha_perturb_std: α 排程的隨機擾動標準差
        alpha_stag_boost:  停滯時 α 的額外增量，增加探索力度
    """

    def __init__(
        self,
        n_params:          int,
        n_particles:       int,
        max_iterations:    int,
        evaluate_fn:       Callable[[np.ndarray], Tuple[float, float]],
        logger:            logging.Logger,
        seed:              int   = 42,
        alpha_max:         float = 1.2,
        alpha_min:         float = 0.4,
        data_dir:          str   = "results_qpso",
        task_name:         str   = "unconditional_9_qpso",
        stagnation_limit:  int   = 8,
        reinit_fraction:   float = 0.20,
        mutation_prob:     float = 0.12,
        mutation_scale:    float = 0.15,
        alpha_perturb_std: float = 0.04,
        alpha_stag_boost:  float = 0.20,
    ):
        self.D  = n_params
        self.M  = n_particles
        self.T  = max_iterations
        self.evaluate_fn       = evaluate_fn
        self.logger            = logger
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

        # QMG 參數邊界：全部 [0, 1]
        self.lb = np.zeros(self.D, dtype=np.float64)
        self.ub = np.ones(self.D,  dtype=np.float64)
        self._mut_range = mutation_scale * (self.ub - self.lb)

        # 粒子群初始化
        self.rng = np.random.default_rng(seed)
        self.positions  = self._rand_pos(self.M)
        self.pbest      = self.positions.copy()
        self.pbest_fit  = np.full(self.M, -np.inf)

        self.gbest_pos  = None
        self.gbest_fit  = -np.inf
        self.gbest_val  = 0.0
        self.gbest_uniq = 0.0

        # 運行統計
        self.history:         List[Dict] = []
        self._stag_counter    = 0
        self._prev_best       = -np.inf
        self._global_eval_cnt = 0
        self._total_reinits   = 0
        self._total_mutations = 0

        os.makedirs(data_dir, exist_ok=True)
        self._csv_path = os.path.join(data_dir, f"{task_name}.csv")
        self._init_csv()

    # ================================================================
    # 工具方法
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

    def _update_pos(self, x, pbest_i, gbest, mbest, alpha) -> np.ndarray:
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
        # 修正：用 dims 索引 _mut_range，確保 shape 一致 (n_dim,)
        noise = self.rng.standard_cauchy(size=n_dim) * self._mut_range[dims]
        x_mut[dims] += noise
        return self._clip(x_mut)

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

    # ================================================================
    # CSV 記錄
    # ================================================================

    _CSV_FIELDS = [
        'eval_index', 'qpso_iter', 'particle',
        'validity', 'uniqueness', 'fitness',
        'gbest_fitness', 'alpha', 'stagnation', 'elapsed_s',
    ]

    def _init_csv(self):
        with open(self._csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self._CSV_FIELDS).writeheader()

    def _write_csv(self, row: dict):
        with open(self._csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self._CSV_FIELDS).writerow(row)

    # ================================================================
    # 單粒子評估（log 格式與 unconditional_9.log 完全對齊）
    # ================================================================

    def _eval_particle(
        self,
        pos:         np.ndarray,
        eval_label:  int,
        qpso_iter:   int,
        particle_id: int,
        alpha:       float,
    ) -> Tuple[float, float, float]:
        """
        呼叫 evaluate_fn，記錄結果，回傳 (validity, uniqueness, fitness)。
        eval_label 對應 BO log 的 "Iteration number: X"。
        """
        self.logger.info(f"Iteration number: {eval_label}")
        t0 = time.time()
        validity, uniqueness = self.evaluate_fn(pos)
        elapsed = time.time() - t0
        fitness = float(validity) * float(uniqueness)

        # 與 unconditional_9.log 完全一致的兩行
        self.logger.info(f"validity (maximize): {validity:.3f}")
        self.logger.info(f"uniqueness (maximize): {uniqueness:.3f}")
        # QPSO 額外診斷（不影響 log 格式比較）
        self.logger.info(
            f"  [QPSO] iter={qpso_iter}  p={particle_id}  "
            f"fit={fitness:.4f}  gbest={self.gbest_fit:.4f}  "
            f"stag={self._stag_counter}  α={alpha:.4f}  t={elapsed:.1f}s"
        )
        self._write_csv({
            'eval_index':   eval_label,
            'qpso_iter':    qpso_iter,
            'particle':     particle_id,
            'validity':     round(float(validity),   4),
            'uniqueness':   round(float(uniqueness), 4),
            'fitness':      round(fitness,            6),
            'gbest_fitness': round(self.gbest_fit,    6),
            'alpha':        round(alpha,              4),
            'stagnation':   self._stag_counter,
            'elapsed_s':    round(elapsed,            1),
        })
        return float(validity), float(uniqueness), fitness

    # ================================================================
    # 主優化迴圈
    # ================================================================

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        執行 SOQPSO 優化。

        每個 QPSO 迭代包含 M 次粒子評估；
        每次評估記錄一個 "Iteration number: X"，與 BO log 格式相同。
        Total evaluations = M×(T+1)（含初始化 Phase 0）。

        Returns:
            (best_params: np.ndarray, best_fitness: float)
        """
        total_evals = self.M * (self.T + 1)
        self.logger.info("=" * 65)
        self.logger.info("SOQPSO 量子粒子群優化啟動")
        self.logger.info(f"  粒子數 M            : {self.M}")
        self.logger.info(f"  參數維度 D           : {self.D}")
        self.logger.info(f"  最大迭代 T           : {self.T}")
        self.logger.info(f"  總評估次數           : {total_evals}")
        self.logger.info(f"  α 排程              : [{self.alpha_min}, {self.alpha_max}] cosine")
        self.logger.info(f"  停滯門檻             : {self.stagnation_limit} QPSO iters")
        self.logger.info(f"  Cauchy 變異機率      : {self.mutation_prob:.0%}")
        self.logger.info(f"  BO 基線 (Best V×U)  : 0.8834")
        self.logger.info("=" * 65)

        # ── Phase 0：初始評估（對應 BO 的 Sobol 初始化）───────────
        self.logger.info("[Phase 0] 初始粒子評估（隨機初始化）...")
        for i in range(self.M):
            v, u, f = self._eval_particle(
                self.positions[i], self._global_eval_cnt, 0, i, self._get_alpha(0)
            )
            self._global_eval_cnt += 1
            self._update_pbest(i, f)
            self._update_gbest(i, f, v, u)
        self._prev_best = self.gbest_fit

        # ── 主迭代 ──────────────────────────────────────────────
        for t in range(self.T):
            alpha = self._get_alpha(t)
            mbest = np.mean(self.pbest, axis=0)
            gbest = self.gbest_pos if self.gbest_pos is not None else self.positions[0]

            iter_fits = []
            for i in range(self.M):
                # 位置更新
                self.positions[i] = self._update_pos(
                    self.positions[i], self.pbest[i], gbest, mbest, alpha
                )
                # Cauchy 變異
                if self.rng.random() < self.mutation_prob:
                    self.positions[i] = self._cauchy_mutation(self.positions[i])
                    self._total_mutations += 1

                # 評估
                v, u, f = self._eval_particle(
                    self.positions[i], self._global_eval_cnt, t + 1, i, alpha
                )
                self._global_eval_cnt += 1
                iter_fits.append(f)
                self._update_pbest(i, f)
                self._update_gbest(i, f, v, u)

            self._update_stagnation(self.gbest_fit)
            self._maybe_reinit()

            mean_fit = float(np.mean(iter_fits))
            max_fit  = float(np.max(iter_fits))
            self.history.append({
                'qpso_iter':       t + 1,
                'n_evals':         self._global_eval_cnt,
                'gbest_fitness':   self.gbest_fit,
                'gbest_validity':  self.gbest_val,
                'gbest_uniqueness': self.gbest_uniq,
                'mean_fitness':    mean_fit,
                'max_fitness':     max_fit,
                'alpha':           alpha,
            })
            self.logger.info(
                f"  [QPSO Iter {t+1:3d}/{self.T}] "
                f"α={alpha:.3f}  "
                f"gbest={self.gbest_fit:.4f} (V={self.gbest_val:.3f} U={self.gbest_uniq:.3f})  "
                f"mean={mean_fit:.4f}  max={max_fit:.4f}  "
                f"stag={self._stag_counter}  evals={self._global_eval_cnt}"
            )

        # ── 最終摘要 ─────────────────────────────────────────────
        self.logger.info("=" * 65)
        self.logger.info("SOQPSO 優化完成")
        self.logger.info(f"  Best V×U   : {self.gbest_fit:.6f}")
        self.logger.info(f"  Best V     : {self.gbest_val:.4f}")
        self.logger.info(f"  Best U     : {self.gbest_uniq:.4f}")
        self.logger.info(
            f"  BO Baseline: 0.8834  "
            + ("✓ 超越基線!" if self.gbest_fit > 0.8834
               else "✗ 未超越 — 建議增加粒子數或迭代次數")
        )
        self.logger.info(f"  Total evals: {self._global_eval_cnt}")
        self.logger.info(f"  Reinits    : {self._total_reinits}")
        self.logger.info(f"  Mutations  : {self._total_mutations}")
        self.logger.info("=" * 65)

        best = self.gbest_pos.copy() if self.gbest_pos is not None else np.zeros(self.D)
        return best, self.gbest_fit