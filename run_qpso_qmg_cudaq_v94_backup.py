"""
==============================================================================
run_qpso_qmg_cudaq.py  ─ CUDA-Q 0.7.1 + SOQPSO 主入口（v9.6 subprocess 隔離版）
==============================================================================
v9.5 → v9.6 記憶體根本修正：

  ★ 問題確認（test_reinit.py 輸出）：
      每次評估（10k shots）洩漏 +2494 MB
      del generator + gc.collect() + malloc_trim → freed 0 MB
      generator 重建 → freed 0 MB
      
      根本原因：CUDA-Q 0.7.1 cuStateVec 後端使用 CUDA pinned memory
      （cudaMallocHost），此類記憶體由 CUDA driver 直接管理，
      完全繞過 glibc heap，任何 Python 層面的釋放手段均無效。
      唯一釋放方式是 CUDA context 銷毀，而 context 綁定到行程，
      只有行程結束才會觸發。

  ★ 解法：subprocess 隔離評估
      每次 evaluate_fn 呼叫都啟動一個獨立子行程執行 worker_eval.py，
      子行程結束時 CUDA driver 強制銷毀 context，所有 pinned memory
      完全釋放。主行程的記憶體使用保持穩定（< 1 GB）。

      代價：每次評估多 ~1-3s 的 subprocess 啟動開銷。
      對比每次評估 ~80s 的採樣時間，額外開銷 < 4%，完全可接受。

依賴：
  worker_eval.py 必須與本檔案放在同一目錄（PYTHONPATH 根目錄）
==============================================================================
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

try:
    import cudaq
except ImportError:
    print("[ERROR] 無法 import cudaq。請執行：pip install cuda-quantum-cu12==0.7.1")
    sys.exit(1)

try:
    from qmg.utils import ConditionalWeightsGenerator
except ImportError as e:
    print(f"[ERROR] 無法 import qmg.utils: {e}")
    sys.exit(1)

try:
    from qpso_optimizer_qmg import QMGSOQPSOOptimizer
except ImportError as e:
    print(f"[ERROR] 無法 import qpso_optimizer_qmg: {e}")
    sys.exit(1)


def _get_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024
    except Exception:
        pass
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        pass
    return -1.0


def log_memory(logger: logging.Logger, label: str = "") -> float:
    rss = _get_rss_mb()
    if rss >= 0:
        logger.info(f"[MEM] {label}  RSS={rss:.0f} MB")
    return rss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QMG CUDA-Q 0.7.1 + SOQPSO（v9.6 subprocess 隔離版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num_heavy_atom",      type=int,   default=9)
    p.add_argument("--num_sample",          type=int,   default=10000)
    p.add_argument("--particles",           type=int,   default=50)
    p.add_argument("--iterations",          type=int,   default=200)
    p.add_argument("--alpha_max",           type=float, default=1.5)
    p.add_argument("--alpha_min",           type=float, default=0.5)
    p.add_argument("--mutation_prob",       type=float, default=0.12)
    p.add_argument("--stagnation_limit",    type=int,   default=8)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument(
        "--backend", type=str, default="cudaq_nvidia",
        choices=["cudaq_qpp", "cudaq_nvidia", "cudaq_nvidia_fp64"],
    )
    p.add_argument(
        "--subprocess_timeout", type=int, default=600,
        help="每次子行程評估的逾時秒數（預設 600s）",
    )
    p.add_argument(
        "--reinit_every", type=int, default=0,
        help="subprocess 模式下此參數無效，保留供相容性使用",
    )
    p.add_argument("--task_name", type=str, default="unconditional_9_qpso")
    p.add_argument("--data_dir",  type=str, default="results_dgx1_gpu_final")
    return p.parse_args()


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("CUDAQQPSOLogger")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s,%(msecs)03d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for h in [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def log_gpu_info(logger: logging.Logger) -> None:
    try:
        import subprocess as _sp, re
        out = _sp.check_output(
            "nvidia-smi --query-gpu=index,name,memory.total,driver_version "
            "--format=csv,noheader",
            shell=True, stderr=_sp.DEVNULL,
        ).decode().strip()
        for line in out.splitlines():
            logger.info(f"  GPU: {line.strip()}")
    except Exception:
        logger.info("  GPU info: 無法取得")
    try:
        import re
        ver_str = cudaq.__version__
        match = re.search(r'(\d+\.\d+\.\d+)', ver_str)
        logger.info(f"  CUDA-Q: {match.group(1) if match else ver_str}")
    except Exception:
        pass
    cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "（未設定）")
    logger.info(f"  CUDA_VISIBLE_DEVICES: {cuda_dev}")


def make_subprocess_evaluate_fn(
    args:          argparse.Namespace,
    cwg:           ConditionalWeightsGenerator,
    logger:        logging.Logger,
    worker_script: str,
    cuda_device:   str,
) -> callable:
    """
    每次 evaluate_fn 呼叫都啟動獨立子行程執行 worker_eval.py。
    子行程結束後 CUDA context 銷毀，pinned memory 完全釋放。
    """
    pythonpath = os.environ.get("PYTHONPATH", ".")
    eval_count = [0]

    def evaluate_fn(pos: np.ndarray) -> tuple:
        eval_idx = eval_count[0]
        eval_count[0] += 1

        uid         = uuid.uuid4().hex[:8]
        weight_path = os.path.join(tempfile.gettempdir(), f"qmg_w_{uid}.npy")
        result_path = os.path.join(tempfile.gettempdir(), f"qmg_r_{uid}.npy")

        try:
            # chemistry constraint 在主行程套用
            w_constrained = cwg.apply_chemistry_constraint(pos.copy())
            np.save(weight_path, w_constrained)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = cuda_device
            env["PYTHONPATH"]           = pythonpath

            cmd = [
                sys.executable,
                worker_script,
                "--weight_path",    weight_path,
                "--result_path",    result_path,
                "--num_heavy_atom", str(args.num_heavy_atom),
                "--num_sample",     str(args.num_sample),
                "--backend",        args.backend,
            ]

            t0  = time.time()
            ret = subprocess.run(
                cmd,
                env            = env,
                timeout        = args.subprocess_timeout,
                capture_output = True,
            )
            elapsed = time.time() - t0

            if ret.returncode != 0:
                stderr_msg = ret.stderr.decode("utf-8", errors="replace")[-500:]
                logger.warning(
                    f"[subprocess] eval #{eval_idx} 失敗（exit {ret.returncode}，{elapsed:.1f}s）\n"
                    f"  stderr tail: {stderr_msg}"
                )
                return 0.0, 0.0

            result_arr = np.load(result_path)
            return float(result_arr[0]), float(result_arr[1])

        except subprocess.TimeoutExpired:
            logger.warning(
                f"[subprocess] eval #{eval_idx} 逾時（>{args.subprocess_timeout}s），回傳 V=0, U=0"
            )
            return 0.0, 0.0

        except Exception as e:
            logger.warning(f"[subprocess] eval #{eval_idx} 例外：{e}")
            return 0.0, 0.0

        finally:
            for path in [weight_path, result_path]:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

    return evaluate_fn


def main() -> None:
    args = parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    log_path = os.path.join(args.data_dir, f"{args.task_name}.log")
    logger   = setup_logger(log_path)

    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Task: ['validity', 'uniqueness']")
    logger.info(f"Condition: ['None', 'None']")
    logger.info(f"objective: ['maximize', 'maximize']")
    logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
    logger.info(f"# of samples: {args.num_sample}")
    logger.info(f"smarts: None")
    logger.info(f"disable_connectivity_position: []")
    logger.info(f"CUDA-Q backend: {args.backend}")
    logger.info(f"[v9.6] 評估模式: subprocess 隔離（解決 CUDA pinned memory 洩漏）")
    logger.info(f"[v9.6] subprocess_timeout: {args.subprocess_timeout}s")
    log_gpu_info(logger)
    log_memory(logger, "啟動時")

    # 確認 worker_eval.py 存在
    script_dir    = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(script_dir, "worker_eval.py")
    if not os.path.exists(worker_script):
        logger.error(
            f"[ERROR] worker_eval.py 不存在：{worker_script}\n"
            f"  請確認 worker_eval.py 與 run_qpso_qmg_cudaq.py 在同一目錄。"
        )
        sys.exit(1)
    logger.info(f"  worker_eval.py: {worker_script} ✓")

    cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    logger.info(f"  Using CUDA_VISIBLE_DEVICES={cuda_device}")

    cwg = ConditionalWeightsGenerator(
        args.num_heavy_atom,
        smarts=None,
        disable_connectivity_position=[],
    )
    n_flexible = int((cwg.parameters_indicator == 0.0).sum())
    logger.info(f"Number of flexible parameters: {n_flexible}")

    assert n_flexible == cwg.length_all_weight_vector, (
        f"[BUG] n_flexible={n_flexible} != {cwg.length_all_weight_vector}"
    )

    total_evals = args.particles * (args.iterations + 1)
    logger.info(
        f"[CUDAQ-QPSO config] M={args.particles}  T={args.iterations}  "
        f"total_evals={total_evals}  seed={args.seed}  backend={args.backend}"
    )

    # worker 功能測試
    logger.info("[v9.6] 執行 worker_eval.py 功能測試（5 shots）...")
    w_test  = cwg.generate_conditional_random_weights(random_seed=42)
    uid     = uuid.uuid4().hex[:8]
    wt_path = os.path.join(tempfile.gettempdir(), f"qmg_test_w_{uid}.npy")
    rt_path = os.path.join(tempfile.gettempdir(), f"qmg_test_r_{uid}.npy")
    np.save(wt_path, w_test)

    env_test = os.environ.copy()
    env_test["CUDA_VISIBLE_DEVICES"] = cuda_device
    env_test["PYTHONPATH"]           = os.environ.get("PYTHONPATH", ".")

    t0 = time.time()
    ret = subprocess.run(
        [
            sys.executable, worker_script,
            "--weight_path", wt_path,
            "--result_path", rt_path,
            "--num_heavy_atom", str(args.num_heavy_atom),
            "--num_sample", "5",
            "--backend", args.backend,
        ],
        env=env_test, timeout=120, capture_output=True,
    )
    elapsed_test = time.time() - t0

    for path in [wt_path, rt_path]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    if ret.returncode != 0:
        logger.error(
            f"[ERROR] worker_eval.py 測試失敗（exit {ret.returncode}）\n"
            f"  stderr: {ret.stderr.decode('utf-8', errors='replace')[-500:]}"
        )
        sys.exit(1)

    logger.info(f"[v9.6] worker_eval.py 測試通過（{elapsed_test:.1f}s）✓")
    log_memory(logger, "worker 測試後")

    evaluate_fn = make_subprocess_evaluate_fn(
        args          = args,
        cwg           = cwg,
        logger        = logger,
        worker_script = worker_script,
        cuda_device   = cuda_device,
    )

    class MemoryAwareOptimizer(QMGSOQPSOOptimizer):
        def optimize(self):
            total_evals_inner = self.M * (self.T + 1)
            self.logger.info("=" * 65)
            self.logger.info("SOQPSO 量子粒子群優化啟動（v9.6 subprocess 隔離版）")
            self.logger.info(f"  粒子數 M            : {self.M}")
            self.logger.info(f"  參數維度 D           : {self.D}")
            self.logger.info(f"  最大迭代 T           : {self.T}")
            self.logger.info(f"  總評估次數           : {total_evals_inner}")
            self.logger.info(f"  α 排程              : [{self.alpha_min}, {self.alpha_max}] cosine")
            self.logger.info(f"  停滯門檻             : {self.stagnation_limit} QPSO iters")
            self.logger.info(f"  Cauchy 變異機率      : {self.mutation_prob:.0%}")
            self.logger.info(f"  評估模式            : subprocess 隔離（無 OOM 風險）")
            self.logger.info(f"  BO 基線 (Best V×U)  : 0.8834")
            self.logger.info("=" * 65)

            self.logger.info("[Phase 0] 初始粒子評估...")
            for i in range(self.M):
                v, u, f = self._eval_particle(
                    self.positions[i], self._global_eval_cnt, 0, i, self._get_alpha(0)
                )
                self._global_eval_cnt += 1
                self._update_pbest(i, f)
                self._update_gbest(i, f, v, u)
            self._prev_best = self.gbest_fit
            log_memory(logger, f"Phase 0 結束（{self.M} evals）")

            for t in range(self.T):
                alpha = self._get_alpha(t)
                mbest = np.mean(self.pbest, axis=0)
                gbest = self.gbest_pos if self.gbest_pos is not None else self.positions[0]

                iter_fits = []
                for i in range(self.M):
                    self.positions[i] = self._update_pos(
                        self.positions[i], self.pbest[i], gbest, mbest, alpha
                    )
                    if self.rng.random() < self.mutation_prob:
                        self.positions[i] = self._cauchy_mutation(self.positions[i])
                        self._total_mutations += 1

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
                    'qpso_iter':        t + 1,
                    'n_evals':          self._global_eval_cnt,
                    'gbest_fitness':    self.gbest_fit,
                    'gbest_validity':   self.gbest_val,
                    'gbest_uniqueness': self.gbest_uniq,
                    'mean_fitness':     mean_fit,
                    'max_fitness':      max_fit,
                    'alpha':            alpha,
                })
                self.logger.info(
                    f"  [QPSO Iter {t+1:3d}/{self.T}] "
                    f"α={alpha:.3f}  "
                    f"gbest={self.gbest_fit:.4f} "
                    f"(V={self.gbest_val:.3f} U={self.gbest_uniq:.3f})  "
                    f"mean={mean_fit:.4f}  max={max_fit:.4f}  "
                    f"stag={self._stag_counter}  evals={self._global_eval_cnt}"
                )
                log_memory(logger, f"Iter {t+1}/{self.T}")

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
            log_memory(logger, "優化完成")

            best = self.gbest_pos.copy() if self.gbest_pos is not None else np.zeros(self.D)
            return best, self.gbest_fit

    optimizer = MemoryAwareOptimizer(
        n_params          = n_flexible,
        n_particles       = args.particles,
        max_iterations    = args.iterations,
        evaluate_fn       = evaluate_fn,
        logger            = logger,
        seed              = args.seed,
        alpha_max         = args.alpha_max,
        alpha_min         = args.alpha_min,
        data_dir          = args.data_dir,
        task_name         = args.task_name,
        stagnation_limit  = args.stagnation_limit,
        mutation_prob     = args.mutation_prob,
    )

    best_params, best_fitness = optimizer.optimize()

    best_npy = os.path.join(args.data_dir, f"{args.task_name}_best_params.npy")
    np.save(best_npy, best_params)
    logger.info(f"最佳參數已儲存: {best_npy}")
    logger.info(
        f"最終結果: V×U={best_fitness:.6f}  "
        + ("✓ 超越 BO 基線 0.8834!" if best_fitness > 0.8834
           else "✗ 未超越 — 建議增加 --particles 或 --iterations")
    )
    log_memory(logger, "程序結束前")


if __name__ == "__main__":
    main()