"""
==============================================================================
qpso_optimizer_ae.py
AE-SOQPSO — Amplitude-Ensemble Single-Objective Quantum PSO
融合 AE-QTS（arXiv:2311.12867v2）的調和加權與配對更新機制
支援單粒子（subprocess）與批次（MPI）兩種評估模式
==============================================================================

v1.0 → v1.1 修正（對照論文 arXiv:2311.12867v2）：

  ★ [BUG-FIX 1] _ae_weighted_mbest() 加權方案錯誤：
      v1.0：單調遞減加權 w_k = 1/(k+1)（最好粒子影響力最大，依序遞減）
      論文：AE-QTS Algorithm 3 中，第 k 對 (best_k, worst_k) 的旋轉幅度
            均為 Δθ/k，代表第 k 名「和」第 M+1-k 名各有影響力 1/k。
            這是「兩端高、中間低」的 U 形加權，排名中間的粒子影響力最小。
      v1.1 修正：改為對稱 U 形調和加權。

  ★ [BUG-FIX 2] _ae_paired_update() 語意偏差：
      v1.0：best 粒子向 gbest 靠近，worst 粒子做 Cauchy 重尾擾動。
      論文：AE-QTS 的 Table I 旋轉查找表對 best 和 worst 粒子做的是
            對稱的同向調整——兩者都依據 (s_best ⊕ s_worst) 的符號
            向「高適應度方向」旋轉。Grover Search 的核心是「振幅放大」，
            讓所有個體的目標態機率都增加，而不是 worst 做隨機探索。
            Cauchy 擾動是 SOQPSO 自身的 mutation 機制，不屬於 AE 配對。
      v1.1 修正：best 和 worst 均向各自的 local attractor（pbest + gbest
            的凸組合）移動，幅度為 rotate_factor/k，符合論文精神。

核心改進（修正後）：

  【1】AE-QTS 對稱 U 形調和加權 mbest
      映射論文 Δθ/k 的配對機制到連續 QPSO：
        排名第 k 名和第 M+1-k 名各貢獻 1/k 的加權影響力
        => 排名兩端（前幾名和後幾名）影響力大，中間最小
      物理意義：接近 gbest 的粒子（high rank）提供收斂方向；
                遠離 gbest 的粒子（low rank）提供多樣性方向；
                中間粒子（已探索但未收斂）提供最少資訊。

  【2】AE-QTS Best-Worst 配對更新（忠實映射論文 Algorithm 3）
      對第 k 對 (best_k, worst_k)，兩者均向各自的 attractor 移動：
        attractor_i = φ·pbest_i + (1-φ)·gbest，φ ~ U(0,1)
        x_i ← x_i + (rotate_factor/k) · (attractor_i - x_i)
      對應論文：兩個粒子都用相同幅度 Δθ/k 做量子旋轉，方向由 Table I 決定。

  【3】批次評估介面（MPI 並行支援）
      batch_evaluate_fn(positions: np.ndarray[M,D]) → list[(v, u)]

  【4】雙目標解耦監控
      分別追蹤 V_best_ever 和 U_best_ever

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
    Amplitude-Ensemble Single-Objective QPSO（v1.1 修正版）。

    評估模式（擇一傳入）：
      evaluate_fn:       (pos: np.ndarray[D]) → (validity, uniqueness)
      batch_evaluate_fn: (positions: np.ndarray[M,D]) → list[(v,u)]
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

        self.lb = np.zeros(self.D, dtype=np.float64)
        self.ub = np.ones(self.D,  dtype=np.float64)
        self._mut_range = mutation_scale * (self.ub - self.lb)

        self.rng       = np.random.default_rng(seed)
        self.positions = self._rand_pos(self.M)
        self.pbest     = self.positions.copy()
        self.pbest_fit = np.full(self.M, -np.inf)

        self.gbest_pos  = None
        self.gbest_fit  = -np.inf
        self.gbest_val  = 0.0
        self.gbest_uniq = 0.0
        self._best_v_ever = 0.0
        self._best_u_ever = 0.0

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
        """Cauchy 重尾變異，用於跳脫局部最優（屬於 SOQPSO 自身機制，非 AE 配對）。"""
        x_mut = x.copy()
        n_dim = max(1, int(self.D * self.rng.uniform(0.15, 0.35)))
        dims  = self.rng.choice(self.D, size=n_dim, replace=False)
        noise = self.rng.standard_cauchy(size=n_dim) * self._mut_range[dims]
        x_mut[dims] += noise
        return self._clip(x_mut)

    # ================================================================
    # ★ AE-QTS 核心機制（v1.1 修正版）
    # ================================================================

    def _ae_weighted_mbest(self) -> np.ndarray:
        """
        ★ v1.1 修正：AE-QTS 對稱 U 形調和加權 mbest。

        論文 AE-QTS Algorithm 3 的結構：
          對第 k 對 (best_k, worst_k)，k = 1..M/2：
            旋轉幅度 = Δθ/k（best_k 和 worst_k 各用同一幅度）

        這意味著：
          - 排名第 k 名（best_k）的影響力貢獻 = 1/k
          - 排名第 M+1-k 名（worst_k）的影響力貢獻 = 1/k
          - 排名在 [M/2+1, M/2] 中間的粒子：沒有配對，影響力最小

        => 加權分佈形狀為 U 形：兩端高（前幾名 + 後幾名），中間低
           與 v1.0 的單調遞減加權（只有前幾名有高影響力）根本不同。

        物理意義：
          - 高排名粒子：提供收斂方向（gbest 附近）
          - 低排名粒子：提供多樣性方向（探索空間的另一端）
          - 中間粒子：資訊重複，影響力最小
        """
        sorted_idx = np.argsort(self.pbest_fit)[::-1]  # fitness 由高到低
        half = self.M // 2
        weights = np.zeros(self.M, dtype=np.float64)

        # 第 k 對：best_k（排名 k-1）和 worst_k（排名 M-k）各貢獻 1/k
        for k in range(1, half + 1):
            w_k = 1.0 / k
            weights[k - 1]        += w_k   # best_k：排名 k（0-indexed: k-1）
            weights[self.M - k]   += w_k   # worst_k：排名 M+1-k（0-indexed: M-k）

        # M 為奇數時，正中間那個粒子沒有配對（影響力 = 0，已是預設值）
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            # 防禦性 fallback：均等加權（不應發生）
            weights[:] = 1.0 / self.M

        return np.sum(self.pbest[sorted_idx] * weights[:, np.newaxis], axis=0)

    def _ae_paired_update(self, alpha: float):
        """
        ★ v1.1 修正：AE-QTS Best-Worst 配對更新，忠實映射論文 Algorithm 3。

        論文精神（Grover Search 振幅放大類比）：
          AE-QTS 的旋轉查找表（Table I）對 best_k 和 worst_k 均按照
          「向目標態方向調整振幅」的原則操作：
            sign = best_j ⊕ worst_j 決定旋轉正負方向
            兩者都朝「增加目標態機率」的方向旋轉，幅度 Δθ/k

          連續空間映射：
            best_k 的 attractor  = φ·pbest_{best_k} + (1-φ)·gbest
            worst_k 的 attractor = φ·pbest_{worst_k} + (1-φ)·gbest
            兩者都向各自的 attractor 移動，步長 rotate_factor/k

          這與 v1.0 的「worst 做 Cauchy 擾動」本質不同：
            v1.0 的 worst 做擾動 → 偏向隨機探索（PSO mutation 機制）
            論文的 worst 做旋轉 → 偏向收斂（振幅放大）
            Cauchy mutation 在 SOQPSO 主迴圈中已有，不應在 AE 配對中重複。

        注意：此函式在每 pair_interval 個 QPSO 迭代執行一次，
              在主位置更新（_update_pos_single）之後、評估之前呼叫。
        """
        if self.gbest_pos is None:
            return

        half     = self.M // 2
        sorted_i = np.argsort(self.pbest_fit)[::-1]   # fitness 由高到低

        for k in range(1, half + 1):
            best_idx  = sorted_i[k - 1]
            worst_idx = sorted_i[self.M - k]
            step = self.rotate_factor / k              # 1/k 衰減，對應論文 Δθ/k

            # 兩者都向各自的 local attractor 移動（Grover 振幅放大類比）
            # attractor_i = φ·pbest_i + (1-φ)·gbest
            for idx in (best_idx, worst_idx):
                phi       = self.rng.uniform(0.0, 1.0, size=self.D)
                attractor = phi * self.pbest[idx] + (1.0 - phi) * self.gbest_pos
                direction = attractor - self.positions[idx]
                self.positions[idx] = self._clip(
                    self.positions[idx] + step * direction
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
            'eval_index':    eval_label,
            'qpso_iter':     qpso_iter,
            'particle':      particle_id,
            'validity':      round(v, 4),
            'uniqueness':    round(u, 4),
            'fitness':       round(f, 6),
            'gbest_fitness': round(self.gbest_fit, 6),
            'best_v_ever':   round(self._best_v_ever, 4),
            'best_u_ever':   round(self._best_u_ever, 4),
            'alpha':         round(alpha, 4),
            'stagnation':    self._stag_counter,
            'elapsed_s':     round(elapsed, 1),
        })

    # ================================================================
    # 主優化迴圈
    # ================================================================

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        執行 AE-SOQPSO 優化。

        自動選擇評估模式：
          - 有 batch_evaluate_fn → 批次模式（MPI 並行）
          - 否則 → 逐粒子模式（subprocess 或單 GPU）
        """
        use_batch   = self.batch_evaluate_fn is not None
        total_evals = self.M * (self.T + 1)

        self.logger.info("=" * 70)
        self.logger.info("AE-SOQPSO（v1.1 修正版）優化啟動")
        self.logger.info(f"  粒子數 M               : {self.M}")
        self.logger.info(f"  參數維度 D              : {self.D}")
        self.logger.info(f"  最大迭代 T              : {self.T}")
        self.logger.info(f"  總評估次數              : {total_evals}")
        self.logger.info(f"  評估模式               : {'批次（MPI）' if use_batch else '逐粒子（subprocess）'}")
        self.logger.info(f"  α 排程                : [{self.alpha_min}, {self.alpha_max}] cosine")
        self.logger.info(f"  AE 加權 mbest（v1.1）  : {'✓ U形對稱調和加權' if self.ae_weighting else '✗ 關閉（均等均值）'}")
        self.logger.info(f"  AE 配對更新（v1.1）    : {'✓ 兩端→attractor，間隔' + str(self.pair_interval) + '迭代' if self.pair_interval > 0 else '✗ 關閉'}")
        self.logger.info(f"  rotate_factor          : {self.rotate_factor}")
        self.logger.info(f"  停滯門檻               : {self.stagnation_limit} iters")
        self.logger.info(f"  Cauchy 變異機率（主迴圈）: {self.mutation_prob:.0%}")
        self.logger.info(f"  BO 基線（Chen 2025）   : 0.8834 (V=0.955, U=0.925)")
        self.logger.info("=" * 70)

        # ── Phase 0：初始化評估 ───────────────────────────────────────────
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

        # ── 主迭代 ────────────────────────────────────────────────────────
        for t in range(self.T):
            alpha = self._get_alpha(t)

            # ── ★ v1.1：U 形對稱調和加權 mbest ──────────────────────────
            if self.ae_weighting:
                mbest = self._ae_weighted_mbest()
            else:
                mbest = np.mean(self.pbest, axis=0)

            gbest = self.gbest_pos if self.gbest_pos is not None else self.positions[0]

            # ── 標準 SOQPSO 位置更新 + Cauchy mutation ──────────────────
            for i in range(self.M):
                self.positions[i] = self._update_pos_single(
                    self.positions[i], self.pbest[i], gbest, mbest, alpha
                )
                if self.rng.random() < self.mutation_prob:
                    self.positions[i] = self._cauchy_mutation(self.positions[i])
                    self._total_mutations += 1

            # ── ★ v1.1：AE-QTS 對稱配對更新（每 pair_interval 迭代）────
            # 在位置更新後、評估前執行，讓 AE 調整的位置直接被評估
            if self.pair_interval > 0 and (t + 1) % self.pair_interval == 0:
                self._ae_paired_update(alpha)
                self.logger.info(
                    f"  [AE 配對更新 v1.1] iter={t+1}  "
                    f"rotate_factor={self.rotate_factor:.4f}  "
                    f"累計 {self._total_ae_updates} 次"
                )

            # ── 評估 ──────────────────────────────────────────────────────
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

        # ── 最終摘要 ──────────────────────────────────────────────────────
        self.logger.info("=" * 70)
        self.logger.info("AE-SOQPSO 優化完成（v1.1）")
        self.logger.info(f"  Best V×U        : {self.gbest_fit:.6f}")
        self.logger.info(f"  Best V          : {self.gbest_val:.4f}")
        self.logger.info(f"  Best U          : {self.gbest_uniq:.4f}")
        self.logger.info(
            f"  V_best_ever     : {self._best_v_ever:.4f}  "
            f"({'正常' if self._best_v_ever > 0.8 else '偏低，建議增加 chemistry constraint'})"
        )
        self.logger.info(
            f"  U_best_ever     : {self._best_u_ever:.4f}  "
            f"({'正常' if self._best_u_ever > 0.7 else '偏低，建議增加粒子數或迭代數'})"
        )
        self.logger.info(
            f"  BO Baseline     : 0.8834  "
            + ("✓ 超越基線!" if self.gbest_fit > 0.8834
               else f"✗ 差距 {0.8834 - self.gbest_fit:.4f}")
        )
        self.logger.info(f"  Total evals     : {self._global_eval_cnt}")
        self.logger.info(f"  Reinits         : {self._total_reinits}")
        self.logger.info(f"  Mutations       : {self._total_mutations}")
        self.logger.info(f"  AE pair updates : {self._total_ae_updates}")
        self.logger.info(f"  AE weighting    : {'v1.1 U形對稱' if self.ae_weighting else '關閉'}")
        self.logger.info("=" * 70)

        best = self.gbest_pos.copy() if self.gbest_pos is not None else np.zeros(self.D)
        return best, self.gbest_fit