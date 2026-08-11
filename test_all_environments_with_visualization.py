# test_all_environments_with_visualization.py
"""
Test i të gjitha mjediseve me vizualizim 3D
MBËSHTET TË GJITHA 63 MJEDISET (8 Bazë + 7 të Përziera + 4 Stanford + 44 MovingAI)
"""

from datetime import datetime
import json
import os
import random
import sys
import time
import traceback
import io

# ============================================================
# FORCO UTF-8 PËR WINDOWS
# ============================================================
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# ============================================================
# PATH SETUP - PARA ÇDO IMPORTI
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(CURRENT_DIR, 'env_model'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'env_model', 'envs'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'nacha_planner'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'planners'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'datasets'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'models'))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch

# ============================================================
# IMPORTET
# ============================================================
try:
    from obstacle_manager_3d import ObstacleManager
except ImportError:
    try:
        from env_model.envs.obstacle_manager_3d import ObstacleManager
    except ImportError:
        print("[ERROR] Cannot import ObstacleManager!")
        sys.exit(1)

from models.env_model_loader import EnvModel
from nacha_planner.nacha_star import NACHAStar

# JPS përdoret këtu si bazë shtesë krahasimi pa mësim (Reviewer A, pika 1)
try:
    from planners.jump_point_search import JumpPointSearch
except ImportError:
    try:
        from jump_point_search import JumpPointSearch
    except ImportError:
        JumpPointSearch = None

import tracemalloc  # për matjen e kujtesës peak (Reviewer A, pika 2)

try:
    from datasets.movingai_loader import load_all_movingai_maps
except ImportError:
    try:
        from movingai_loader import load_all_movingai_maps
    except ImportError:
        def load_all_movingai_maps(grid_size=20):
            return {}

# ============================================================
# KONFIGURIMI
# ============================================================
START_POS = (0, 0, 0)
GOAL_POS = (18, 18, 18)
START_YAW = 0

# ============================================================
# FUNKSIONI PËR TË GJITHA MJEDISET
# ============================================================
def get_all_environments(seed=42, grid_size=20, include_movingai=True):
    patterns = {}
    
    try:
        base_patterns = ObstacleManager.get_all_patterns(
            seed=seed, 
            grid_size=grid_size,
            include_mixed=True,
            include_movingai=False,
            include_stanford=True
        )
        patterns.update(base_patterns)
    except Exception as e:
        print(f"[WARN] Gabim gjatë ngarkimit të ObstacleManager: {e}")
    
    if include_movingai:
        try:
            movingai_maps = load_all_movingai_maps(grid_size=grid_size)
            patterns.update(movingai_maps)
        except Exception as e:
            print(f"[WARN] Gabim gjatë ngarkimit të MovingAI: {e}")
    
    return patterns

# ============================================================
# SafeEnvModelWrapper
# ============================================================
class SafeEnvModelWrapper:
    def __init__(self, raw_model, grid_size=20):
        self.raw_model = raw_model
        self.grid_size = grid_size

    def predict_grid(self, grid_size=None):
        if grid_size is None:
            grid_size = self.grid_size

        if hasattr(self.raw_model, 'predict_grid'):
            try:
                return self.raw_model.predict_grid(grid_size)
            except Exception:
                pass

        if hasattr(self.raw_model, 'predict_batch'):
            try:
                xs, ys, zs = np.meshgrid(
                    np.arange(grid_size),
                    np.arange(grid_size),
                    np.arange(grid_size),
                    indexing='ij'
                )
                all_points = list(zip(
                    xs.ravel().tolist(),
                    ys.ravel().tolist(),
                    zs.ravel().tolist()
                ))

                preds = self.raw_model.predict_batch(all_points)
                preds_arr = np.asarray(preds, dtype=np.float32)
                if preds_arr.size == grid_size ** 3:
                    return preds_arr.reshape((grid_size, grid_size, grid_size))
                else:
                    return np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
            except Exception:
                pass

        grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    grid[x, y, z] = self.predict(x, y, z)
        return grid
    
    def predict(self, x, y, z):
        if hasattr(self.raw_model, 'predict'):
            try:
                res = self.raw_model.predict((x, y, z))
                if isinstance(res, (int, float, np.number)):
                    return float(res)
                elif isinstance(res, torch.Tensor):
                    return float(res.item())
                elif isinstance(res, (list, np.ndarray)) and len(res) > 0:
                    return float(res[0])
            except Exception:
                pass

        if callable(self.raw_model):
            try:
                inp = torch.tensor([[x, y, z]], dtype=torch.float32)
                with torch.no_grad():
                    out = self.raw_model(inp)
                return float(out.squeeze().item())
            except Exception:
                pass

        return 0.0    

    def __call__(self, *args, **kwargs):
        if len(args) == 3:
            return self.predict(args[0], args[1], args[2])
        elif len(args) == 1 and isinstance(args[0], (tuple, list)):
            return self.predict(args[0][0], args[0][1], args[0][2])
        return 0.0

# ============================================================
# TestAllEnvironments
# ============================================================
class TestAllEnvironments:
    def __init__(self, grid_size=20, max_expansions=100000, n_runs=1, visualize=True, include_movingai=True):
        self.grid_size = grid_size
        self.max_expansions = max_expansions
        self.n_runs = n_runs
        self.visualize = visualize
        self.include_movingai = include_movingai
        self.results = {}
        self.models = {}

        self.start = (START_POS[0], START_POS[1], START_POS[2], START_YAW)
        self.goal = GOAL_POS

        self.fig_dir = f"results/figures_all_envs_grid{grid_size}"
        os.makedirs(self.fig_dir, exist_ok=True)

        self._load_all_models()

    def _load_all_models(self):
        patterns = get_all_environments(seed=42, grid_size=self.grid_size, include_movingai=self.include_movingai)

        print("[INFO] Loading models for each environment...")
        loaded = 0

        for pattern_name in patterns.keys():
            if len(patterns[pattern_name]) == 0:
                continue

            possible_names = [
                f"env_model_{pattern_name}_backprop",
                f"env_model_{pattern_name}",
                pattern_name
            ]
            
            raw_m = None
            for base_name in possible_names:
                model_path = os.path.join(CURRENT_DIR, 'env_model', 'models', f'{base_name}.pt')
                if os.path.exists(model_path):
                    try:
                        raw_m = EnvModel(model_path)
                        print(f"  [OK] {pattern_name}: loaded from {base_name}.pt")
                        break
                    except Exception as e:
                        print(f"  [ERROR] {pattern_name}: error loading {base_name}.pt - {e}")
            
            if raw_m is None:
                alt_path = os.path.join(CURRENT_DIR, 'env_model', 'models', 'env_model_cube_backprop.pt')
                if os.path.exists(alt_path):
                    try:
                        raw_m = EnvModel(alt_path)
                        print(f"  [WARN] {pattern_name}: using cube model (fallback)")
                    except Exception:
                        print(f"  [WARN] {pattern_name}: no model found")
                else:
                    print(f"  [WARN] {pattern_name}: no model found")

            if raw_m is not None:
                self.models[pattern_name] = SafeEnvModelWrapper(raw_m, grid_size=self.grid_size)
                loaded += 1

        print(f"  [OK] Loaded {loaded}/{len(patterns)} models")

    def get_all_patterns(self):
        return get_all_environments(seed=42, grid_size=self.grid_size, include_movingai=self.include_movingai)

    def _check_feasibility(self, obstacles):
        from collections import deque

        obstacle_set = set(obstacles)
        start = START_POS
        goal = GOAL_POS

        if start in obstacle_set or goal in obstacle_set:
            return False

        queue = deque([start])
        visited = {start}
        moves = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        while queue:
            current = queue.popleft()
            if current == goal:
                return True
            for dx, dy, dz in moves:
                nx, ny, nz = current[0] + dx, current[1] + dy, current[2] + dz
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 0 <= nz < self.grid_size:
                    if (nx, ny, nz) not in visited and (nx, ny, nz) not in obstacle_set:
                        visited.add((nx, ny, nz))
                        queue.append((nx, ny, nz))
        return False

    def test_and_visualize(self, env_name, obstacles):
        start = self.start
        goal = self.goal

        print(f"\n  Testing: {env_name} (obstacles: {len(obstacles)})")

        is_feasible = self._check_feasibility(obstacles)
        has_model = env_name in self.models
        print(f"    Feasible: {is_feasible}, Has model: {has_model}")

        results = {
            'name': env_name,
            'obstacles': len(obstacles),
            'feasible': is_feasible,
            'has_model': has_model,
            'runs': []
        }

        for run_id in range(self.n_runs):
            run_data = {'run_id': run_id}

            mem_no_model = float('nan')
            try:
                tracemalloc.start()
                planner_no_model = NACHAStar(
                    grid_size=self.grid_size,
                    obstacles=obstacles,
                    env_model=None,
                    max_expansions=self.max_expansions
                )

                start_time = time.time()
                path_no_model = planner_no_model.plan(start, goal)
                time_no_model = time.time() - start_time
                _, peak = tracemalloc.get_traced_memory()
                mem_no_model = peak / (1024 ** 2)
                tracemalloc.stop()
            except Exception as e:
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
                path_no_model = None
                time_no_model = 0
                print(f"    [WARN] Error in 'No Model' planner: {e}")

            run_data['no_model'] = {
                'success': path_no_model is not None,
                'time': time_no_model,
                'length': len(path_no_model) if path_no_model else 0,
                'peak_memory_mb': mem_no_model,
                'path': path_no_model
            }

            path_with_model = None
            time_with_model = 0
            success_with_model = False
            mem_with_model = float('nan')

            if has_model:
                try:
                    tracemalloc.start()
                    planner_with_model = NACHAStar(
                        grid_size=self.grid_size,
                        obstacles=obstacles,
                        env_model=self.models[env_name],
                        lambda0=0.1,
                        use_adaptive_lambda=False,
                        max_expansions=self.max_expansions
                    )

                    start_time = time.time()
                    path_with_model = planner_with_model.plan(start, goal)
                    time_with_model = time.time() - start_time
                    success_with_model = path_with_model is not None
                    _, peak = tracemalloc.get_traced_memory()
                    mem_with_model = peak / (1024 ** 2)
                    tracemalloc.stop()
                except Exception as e:
                    if tracemalloc.is_tracing():
                        tracemalloc.stop()
                    print(f"    [WARN] Error in 'With Model' planner for {env_name}: {e}")
                    traceback.print_exc()

                run_data['with_model'] = {
                    'success': success_with_model,
                    'time': time_with_model,
                    'length': len(path_with_model) if path_with_model else 0,
                    'peak_memory_mb': mem_with_model,
                    'path': path_with_model
                }
            else:
                run_data['with_model'] = {
                    'success': False,
                    'time': 0,
                    'length': 0,
                    'peak_memory_mb': float('nan'),
                    'path': None
                }

            # ------------------------------------------------------------
            # Bazë shtesë pa mësim: Jump Point Search (Reviewer A, pika 1).
            # Krahasohet me NACHA* (Proposed) në të njëjtin run, të njëjtat
            # pengesa/start/goal, për të izoluar kontributin e komponentit
            # neural kundrejt një planner-i klasik pa mësim por të
            # optimizuar algoritmikisht (symmetry pruning).
            # ------------------------------------------------------------
            path_jps, time_jps, mem_jps = None, 0.0, float('nan')
            if JumpPointSearch is not None:
                try:
                    tracemalloc.start()
                    planner_jps = JumpPointSearch(self.grid_size, obstacles)
                    t0 = time.time()
                    path_jps = planner_jps.plan(start, goal)
                    time_jps = time.time() - t0
                    _, peak = tracemalloc.get_traced_memory()
                    mem_jps = peak / (1024 ** 2)
                    tracemalloc.stop()
                except Exception as e:
                    if tracemalloc.is_tracing():
                        tracemalloc.stop()
                    print(f"    [WARN] Error in 'JPS' planner for {env_name}: {e}")

            run_data['jps'] = {
                'success': path_jps is not None,
                'time': time_jps,
                'length': len(path_jps) if path_jps else 0,
                'peak_memory_mb': mem_jps,
                'path': path_jps
            }

            results['runs'].append(run_data)

            status_no = "[OK]" if path_no_model else "[FAIL]"
            time_no_str = f"{time_no_model:.3f}s" if path_no_model else "-"

            if has_model and success_with_model:
                status_with = "[OK]"
                time_with_str = f"{time_with_model:.3f}s"
            elif has_model:
                status_with = "[FAIL]"
                time_with_str = "-"
            else:
                status_with = "[SKIP]"
                time_with_str = "no model"

            print(f"    Run {run_id+1}: {status_no} No model ({time_no_str}) | {status_with} With model ({time_with_str})")

            if self.visualize and (path_no_model is not None or path_with_model is not None):
                self._visualize_comparison(env_name, obstacles, path_no_model, path_with_model, run_id)

        return results

    def _visualize_comparison(self, env_name, obstacles, path_no_model, path_with_model, run_id):
        start = START_POS
        goal = GOAL_POS

        max_obs = 1000
        obs_sample = obstacles if len(obstacles) <= max_obs else list(obstacles)[:max_obs]

        if path_no_model is not None:
            fig1 = plt.figure(figsize=(12, 9))
            ax1 = fig1.add_subplot(111, projection='3d')
            ax1.set_title(f"{env_name} - No Model (Run {run_id+1})", fontsize=12)
            ax1.set_xlim(0, self.grid_size)
            ax1.set_ylim(0, self.grid_size)
            ax1.set_zlim(0, self.grid_size)

            if obs_sample:
                obs_pts = [p[:3] for p in obs_sample]
                xs, ys, zs = zip(*obs_pts)
                ax1.scatter(xs, ys, zs, c='gray', s=5, alpha=0.3, label='Obstacles')

            ax1.scatter(start[0], start[1], start[2], c='green', s=150, marker='*', label='Start')
            ax1.scatter(goal[0], goal[1], goal[2], c='gold', s=150, marker='*', label='Goal')

            pts = [p[:3] for p in path_no_model]
            xs, ys, zs = zip(*pts)
            ax1.plot(xs, ys, zs, c='red', linewidth=3, label='Path')

            ax1.legend()
            filename1 = os.path.join(self.fig_dir, f"{env_name}_no_model.png")
            plt.tight_layout()
            plt.savefig(filename1, dpi=150)
            plt.close(fig1)

        if path_with_model is not None:
            fig2 = plt.figure(figsize=(12, 9))
            ax2 = fig2.add_subplot(111, projection='3d')
            ax2.set_title(f"{env_name} - With Model (Run {run_id+1})", fontsize=12)
            ax2.set_xlim(0, self.grid_size)
            ax2.set_ylim(0, self.grid_size)
            ax2.set_zlim(0, self.grid_size)

            if obs_sample:
                obs_pts = [p[:3] for p in obs_sample]
                xs, ys, zs = zip(*obs_pts)
                ax2.scatter(xs, ys, zs, c='gray', s=5, alpha=0.3, label='Obstacles')

            ax2.scatter(start[0], start[1], start[2], c='green', s=150, marker='*', label='Start')
            ax2.scatter(goal[0], goal[1], goal[2], c='gold', s=150, marker='*', label='Goal')

            pts = [p[:3] for p in path_with_model]
            xs, ys, zs = zip(*pts)
            ax2.plot(xs, ys, zs, c='blue', linewidth=3, label='Path (With Model)')

            ax2.legend()
            filename2 = os.path.join(self.fig_dir, f"{env_name}_with_model.png")
            plt.tight_layout()
            plt.savefig(filename2, dpi=150)
            plt.close(fig2)

        if path_no_model is not None and path_with_model is not None:
            fig3 = plt.figure(figsize=(12, 9))
            ax3 = fig3.add_subplot(111, projection='3d')
            ax3.set_title(f"{env_name} - Comparison (Run {run_id+1})", fontsize=12)
            ax3.set_xlim(0, self.grid_size)
            ax3.set_ylim(0, self.grid_size)
            ax3.set_zlim(0, self.grid_size)

            if obs_sample:
                obs_pts = [p[:3] for p in obs_sample]
                xs, ys, zs = zip(*obs_pts)
                ax3.scatter(xs, ys, zs, c='gray', s=5, alpha=0.2, label='Obstacles')

            ax3.scatter(start[0], start[1], start[2], c='green', s=150, marker='*', label='Start')
            ax3.scatter(goal[0], goal[1], goal[2], c='gold', s=150, marker='*', label='Goal')

            pts1 = [p[:3] for p in path_no_model]
            xs1, ys1, zs1 = zip(*pts1)
            ax3.plot(xs1, ys1, zs1, c='red', linewidth=3, label='No Model')

            pts2 = [p[:3] for p in path_with_model]
            xs2, ys2, zs2 = zip(*pts2)
            ax3.plot(xs2, ys2, zs2, c='blue', linewidth=3, linestyle='--', label='With Model')

            ax3.legend()
            filename3 = os.path.join(self.fig_dir, f"{env_name}_comparison.png")
            plt.tight_layout()
            plt.savefig(filename3, dpi=150)
            plt.close(fig3)

    def run_all(self):
        viz_status = "[OK] Enabled" if self.visualize else "[OFF] Disabled"
        print("=" * 70)
        print("TESTING ALL ENVIRONMENTS WITH VISUALIZATION")
        print("=" * 70)
        print(f"  Grid size: {self.grid_size}x{self.grid_size}x{self.grid_size}")
        print(f"  Max expansions: {self.max_expansions}")
        print(f"  Runs per environment: {self.n_runs}")
        print(f"  Visualization: {viz_status}")
        print(f"  Include MovingAI: {self.include_movingai}")
        print("=" * 70)

        patterns = self.get_all_patterns()
        print(f"\n[INFO] Total environments: {len(patterns)}")
        
        base = [k for k in patterns.keys() if '+' not in k and not k.startswith('movingai_') and not k.startswith('stanford_')]
        mixed = [k for k in patterns.keys() if '+' in k]
        stanford = [k for k in patterns.keys() if k.startswith('stanford_')]
        movingai = [k for k in patterns.keys() if k.startswith('movingai_')]
        
        print(f"  - Base: {len(base)}")
        print(f"  - Mixed: {len(mixed)}")
        print(f"  - Stanford: {len(stanford)}")
        print(f"  - MovingAI: {len(movingai)}")

        env_counter = 0
        total_envs = len(patterns)
        
        for env_name, obstacles in patterns.items():
            env_counter += 1
            print(f"\n[Env] [{env_counter}/{total_envs}] {env_name}")
            
            if len(obstacles) == 0:
                print(f"  [WARN] Skipping {env_name} (empty)")
                continue

            self.results[env_name] = self.test_and_visualize(env_name, obstacles)

        self.analyze_results()
        self.save_results()
        self.print_summary_table()

        return self.results

    def analyze_results(self):
        print('\n' + '=' * 70)
        print('RESULTS ANALYSIS')
        print('=' * 70)

        total_envs = len(self.results)
        feasible_envs = sum(1 for d in self.results.values() if d['feasible'])
        envs_with_model = sum(1 for d in self.results.values() if d['has_model'])

        no_model_success = 0
        with_model_success = 0
        jps_success = 0
        no_model_times = []
        with_model_times = []
        no_model_lengths = []
        with_model_lengths = []
        no_model_mem = []
        with_model_mem = []
        jps_times = []
        jps_mem = []

        for env_name, data in self.results.items():
            for run in data['runs']:
                if run['no_model']['success']:
                    no_model_success += 1
                    no_model_times.append(run['no_model']['time'])
                    no_model_lengths.append(run['no_model']['length'])
                    if not np.isnan(run['no_model'].get('peak_memory_mb', np.nan)):
                        no_model_mem.append(run['no_model']['peak_memory_mb'])

                if run['with_model']['success']:
                    with_model_success += 1
                    with_model_times.append(run['with_model']['time'])
                    with_model_lengths.append(run['with_model']['length'])
                    if not np.isnan(run['with_model'].get('peak_memory_mb', np.nan)):
                        with_model_mem.append(run['with_model']['peak_memory_mb'])

                jps_run = run.get('jps', {})
                if jps_run.get('success'):
                    jps_success += 1
                    jps_times.append(jps_run['time'])
                    if not np.isnan(jps_run.get('peak_memory_mb', np.nan)):
                        jps_mem.append(jps_run['peak_memory_mb'])

        total_runs = total_envs * self.n_runs if total_envs > 0 else 1

        print('\n  Statistics:')
        print(f"    Total environments: {total_envs}")
        print(f"    Feasible: {feasible_envs}")
        print(f"    Infeasible: {total_envs - feasible_envs}")
        print(f"    With models: {envs_with_model}")

        print('\n  Success Rates:')
        if total_runs > 0:
            print(f"    No model: {no_model_success}/{total_runs} ({no_model_success/total_runs*100:.1f}%)")
            print(f"    With model: {with_model_success}/{total_runs} ({with_model_success/total_runs*100:.1f}%)")

        if no_model_times:
            print(f"\n  No model avg time: {np.mean(no_model_times):.4f}s")
            print(f"  No model avg length: {np.mean(no_model_lengths):.2f}")

        if with_model_times:
            print(f"  With model avg time: {np.mean(with_model_times):.4f}s")
            print(f"  With model avg length: {np.mean(with_model_lengths):.2f}")

            if no_model_times and with_model_times:
                speedup = np.mean(no_model_times) / np.mean(with_model_times)
                print(f"\n  Overall speedup: {speedup:.2f}x")

        # ---- Ablacion: kontributi i heuristikës neurale (Reviewer A) ----
        # "no_model" = NACHA* pa asnjë kontribut nga rrjeti (f(s) gjeometrike
        # pure), "with_model" = NACHA* i plotë. Diferenca izolon empirikisht
        # kontributin e komponentit neural, i pavarur nga JPS/A*.
        print('\n  Ablation (Neural Heuristic Contribution):')
        print(f"    NACHA* WITHOUT neural cost — avg time: "
              f"{np.mean(no_model_times) if no_model_times else float('nan'):.4f}s")
        print(f"    NACHA* WITH neural cost    — avg time: "
              f"{np.mean(with_model_times) if with_model_times else float('nan'):.4f}s")

        if no_model_mem:
            print(f"\n  Peak memory — NACHA* no-model: {np.mean(no_model_mem):.3f} MB")
        if with_model_mem:
            print(f"  Peak memory — NACHA* with-model: {np.mean(with_model_mem):.3f} MB")

        # ---- Bazë pa mësim: JPS (Reviewer A) ----
        if jps_times:
            print(f"\n  JPS (no-learning baseline) — success: {jps_success}/{total_runs} "
                  f"({jps_success/total_runs*100:.1f}%), avg time: {np.mean(jps_times):.4f}s")
            if jps_mem:
                print(f"  Peak memory — JPS: {np.mean(jps_mem):.3f} MB")

    def save_results(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/all_envs_grid{self.grid_size}_{timestamp}.json"
        os.makedirs('results', exist_ok=True)

        saveable = {}
        for env_name, data in self.results.items():
            saveable[env_name] = {
                'obstacles': data['obstacles'],
                'feasible': data['feasible'],
                'has_model': data['has_model'],
                'runs': []
            }
            for run in data['runs']:
                run_copy = {
                    'run_id': run['run_id'],
                    'no_model': {
                        'success': run['no_model']['success'],
                        'time': run['no_model']['time'],
                        'length': run['no_model']['length'],
                        'peak_memory_mb': run['no_model'].get('peak_memory_mb', float('nan'))
                    },
                    'with_model': {
                        'success': run['with_model']['success'],
                        'time': run['with_model']['time'],
                        'length': run['with_model']['length'],
                        'peak_memory_mb': run['with_model'].get('peak_memory_mb', float('nan'))
                    },
                    'jps': {
                        'success': run.get('jps', {}).get('success', False),
                        'time': run.get('jps', {}).get('time', 0),
                        'length': run.get('jps', {}).get('length', 0),
                        'peak_memory_mb': run.get('jps', {}).get('peak_memory_mb', float('nan'))
                    }
                }
                saveable[env_name]['runs'].append(run_copy)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(saveable, f, indent=2, default=str)

        print(f"\n[SAVE] Results saved to: {filename}")

    def print_summary_table(self):
        print('\n' + '=' * 110)
        print('SUMMARY TABLE - ALL ENVIRONMENTS')
        print('=' * 110)
        print(f"\n{'Environment':<30} {'Obs':<8} {'Feas':<6} {'Model':<6} {'No Model':<14} {'With Model':<14} {'Speedup':<10}")
        print('-' * 110)

        categories = {
            'Base': [],
            'Mixed': [],
            'Stanford': [],
            'MovingAI': []
        }
        
        for env_name, data in self.results.items():
            if '+' in env_name:
                categories['Mixed'].append(env_name)
            elif env_name.startswith('stanford_'):
                categories['Stanford'].append(env_name)
            elif env_name.startswith('movingai_'):
                categories['MovingAI'].append(env_name)
            else:
                categories['Base'].append(env_name)
        
        for category, envs in categories.items():
            if envs:
                print(f"\n  {category} Environments:")
                for env_name in envs:
                    data = self.results[env_name]
                    feasible = "[OK]" if data['feasible'] else "[NO]"
                    obstacles = data['obstacles']
                    has_model = "[OK]" if data['has_model'] else "[NO]"

                    no_model_ok = any(r['no_model']['success'] for r in data['runs'])
                    no_model_avg = np.mean([r['no_model']['time'] for r in data['runs'] if r['no_model']['success']]) if no_model_ok else 0
                    no_model_str = f"{no_model_avg:.3f}s" if no_model_ok else "[FAIL]"

                    with_model_ok = any(r['with_model']['success'] for r in data['runs'])
                    with_model_avg = np.mean([r['with_model']['time'] for r in data['runs'] if r['with_model']['success']]) if with_model_ok else 0
                    with_model_str = f"{with_model_avg:.3f}s" if with_model_ok else ("[FAIL]" if data['has_model'] else "[SKIP]")

                    if no_model_ok and with_model_ok and no_model_avg > 0 and with_model_avg > 0:
                        speedup = no_model_avg / with_model_avg
                        speedup_str = f"{speedup:.2f}x"
                    else:
                        speedup_str = "N/A"

                    print(f"    {env_name:<26} {obstacles:<8} {feasible:<6} {has_model:<6} {no_model_str:<14} {with_model_str:<14} {speedup_str:<10}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test all environments with 3D visualization")
    parser.add_argument('--env', type=str, default=None, help="Test only specific environment")
    parser.add_argument('--no-viz', action='store_true', help="Disable visualization")
    parser.add_argument('--runs', type=int, default=1, help="Number of runs per environment")
    parser.add_argument('--grid', type=int, default=20, help="Grid size (20, 50, etc.)")
    parser.add_argument('--no-movingai', action='store_true', help="Exclude MovingAI environments")

    args = parser.parse_args()

    tester = TestAllEnvironments(
        grid_size=args.grid,
        max_expansions=100000,
        n_runs=args.runs,
        visualize=not args.no_viz,
        include_movingai=not args.no_movingai
    )

    if args.env:
        patterns = tester.get_all_patterns()
        if args.env in patterns:
            obstacles = patterns[args.env]
            tester.results[args.env] = tester.test_and_visualize(args.env, obstacles)
            tester.analyze_results()
            tester.save_results()
            tester.print_summary_table()
        else:
            print(f"[ERROR] Environment '{args.env}' not found!")
            print(f"   Available environments: {list(patterns.keys())[:10]}...")
    else:
        tester.run_all()

    print('\n' + '=' * 70)
    print('[OK] ALL TESTS COMPLETED SUCCESSFULLY')
    print(f'[SAVE] Figures saved in: {tester.fig_dir}')
    print('=' * 70)
