"""
benchmark_runner_environments_with_visualization.py
Skripti i Rindërtuar për Ekzekutimin e Metrikave dhe Generimin e Grafiqeve 3D ME VETE PËR ÇDO MJEDIS (ENVIRONMENT)
MBËSHTET TË GJITHA 63 MJEDISET (8 Bazë + 7 të Përziera + 4 Stanford + 44 MovingAI)
"""

import sys
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
# PATH SETUP
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.join(CURRENT_DIR, 'env_model'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'env_model', 'envs'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'nacha_planner'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'planners'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'datasets'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'models'))

# ============================================================
# IMPORTET
# ============================================================
# Importet e mjedisit
try:
    from env_model.envs.obstacle_manager_3d import ObstacleManager
except ImportError:
    try:
        from obstacle_manager_3d import ObstacleManager
    except ImportError:
        from envs.obstacle_manager_3d import ObstacleManager

# Importet e MovingAI
try:
    from datasets.movingai_loader import load_all_movingai_maps
except ImportError:
    try:
        from movingai_loader import load_all_movingai_maps
    except ImportError:
        def load_all_movingai_maps(grid_size=20):
            return {}

# Importet e Planner-ave
from planners.astar_baseline import AStarBaseline
from nacha_planner.nacha_star import NACHAStar
from planners.astar_neural import AStarNeural
from planners.rrt_star import RRTStar
from planners.bit_star import BITStar
from planners.astar_all import EnvironmentModel

# ------------------------------------------------------------------
# Planner-a shtesë për krahasim me literaturën (Reviewer A, pika 1):
#   - Jump Point Search (JPS): optimizim simetrik i A* pa mësim
#   - D* Lite: planifikues inkremental/dinamik klasik
#   - DQN: planifikues i bazuar në Reinforcement Learning (opsional,
#     kërkon para-trajnim; përfshihet vetëm si referencë cilësore,
#     shih shënimin te README/artikull mbi kufizimet e tij)
# ------------------------------------------------------------------
try:
    from planners.jump_point_search import JumpPointSearch
except ImportError:
    try:
        from jump_point_search import JumpPointSearch
    except ImportError:
        JumpPointSearch = None

try:
    from planners.d_star_lite import DStarLite
except ImportError:
    try:
        from d_star_lite import DStarLite
    except ImportError:
        DStarLite = None

try:
    from planners.dqn_planner import DQNPlanner
except ImportError:
    try:
        from dqn_planner import DQNPlanner
    except ImportError:
        DQNPlanner = None

# ============================================================
# FUNKSIONET
# ============================================================
def get_all_environments(seed=42, grid_size=20, include_movingai=True):
    """Merr të gjitha mjediset e disponueshme (63 total)."""
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
        print(f"[INFO] Ngarkova {len(base_patterns)} mjedise nga ObstacleManager")
    except Exception as e:
        print(f"[WARN] Gabim gjatë ngarkimit të ObstacleManager: {e}")
    
    if include_movingai:
        try:
            movingai_maps = load_all_movingai_maps(grid_size=grid_size)
            patterns.update(movingai_maps)
            print(f"[INFO] Ngarkova {len(movingai_maps)} mjedise MovingAI")
        except Exception as e:
            print(f"[WARN] Gabim gjatë ngarkimit të MovingAI: {e}")
    
    return patterns

def get_valid_point(point, obstacles, grid_size):
    """Gjen pikën më të afërt të lirë nëse pika fillestare është e bllokuar."""
    if point not in obstacles:
        return point

    x, y, z = point
    for radius in range(1, grid_size):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
                        candidate = (nx, ny, nz)
                        if candidate not in obstacles:
                            return candidate
    return point

def calculate_path_length(path):
    """Llogarit gjatësinë reale Evklidiane 3D të shtegut."""
    if path is None or len(path) < 2:
        return float('inf')
    
    length = 0.0
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])
        length += np.linalg.norm(p2 - p1)
    return length

def extract_nodes_expanded(planner):
    """Nxjerr numrin e nyjeve të zgjeruara/mostruara nga planner-i."""
    possible_attrs = [
        'nodes_expanded', 'expanded_nodes', 'num_expansions', 
        'visited_count', 'iterations', 'tree_size', 'samples_count'
    ]
    for attr in possible_attrs:
        if hasattr(planner, attr):
            return getattr(planner, attr)
    return 0

def plot_3d_paths_for_env(grid_size, obstacles, start, goal, paths_dict, env_name):
    """Gjeneron dhe ruan paraqitjen grafike 3D."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    if obstacles:
        obs_sample = obstacles if len(obstacles) <= 2000 else list(obstacles)[:2000]
        obs_x, obs_y, obs_z = zip(*obs_sample) if obs_sample else ([], [], [])
        if obs_x:
            ax.scatter(obs_x, obs_y, obs_z, c='gray', alpha=0.15, s=20, marker='s', label='Pengesat')

    color_map = {
        "NACHA* (Proposed)":            ("#D62728", "-", 3.0, "o"),
        "NACHA* (Ablation: No Neural)": ("#8C564B", "-",  2.0, "P"),
        "A* Baseline":                  ("#1F77B4", "--", 2.0, "^"),
        "Neural A*":                    ("#2CA02C", "-.", 2.0, "s"),
        "BIT*":                         ("#FF7F0E", ":",  2.0, "d"),
        "RRT*":                         ("#9467BD", ":",  2.0, "x"),
        "JPS":                          ("#17BECF", "--", 2.0, "v"),
        "D* Lite":                      ("#BCBD22", "--", 2.0, "*"),
        "DQN (RL)":                     ("#7F7F7F", ":",  2.0, "."),
    }

    for name, path in paths_dict.items():
        if path is not None and len(path) > 0:
            path_np = np.array(path)
            color, linestyle, linewidth, marker = color_map.get(name, ("black", "-", 1.5, None))
            
            ax.plot(
                path_np[:, 0], path_np[:, 1], path_np[:, 2],
                label=name, color=color, linestyle=linestyle, 
                linewidth=linewidth, alpha=0.85
            )
            ax.scatter(
                path_np[:, 0], path_np[:, 1], path_np[:, 2],
                color=color, s=15, marker=marker
            )

    ax.scatter([start[0]], [start[1]], [start[2]], color='lime', s=120, marker='*', label='Start')
    ax.scatter([goal[0]], [goal[1]], [goal[2]], color='magenta', s=120, marker='X', label='Goal')

    ax.set_title(f"3D Paths - Environment: '{env_name}'", fontsize=14, fontweight='bold')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([0, grid_size])
    ax.set_ylim([0, grid_size])
    ax.set_zlim([0, grid_size])
    
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
    plt.tight_layout()
    
    output_img = f"3d_paths_{env_name}.png"
    plt.savefig(output_img, dpi=300)
    print(f"[INFO] Grafiku 3D u ruajt si: '{output_img}'")
    plt.close(fig)

def run_benchmark_for_environment(env_name, planners, grid_size, obstacles, start, goal, num_runs=5):
    """Ekzekuton testin krahasues dhe llogarit metrikat."""
    raw_metrics = []
    best_paths = {}

    print("\n" + "#" * 85)
    print(f"   TESTING ENVIRONMENT: [{env_name.upper()}]")
    print(f"   Grid Size: {grid_size}x{grid_size}x{grid_size} | Runs: {num_runs} | Obstacles: {len(obstacles)}")
    print(f"   Start: {start} | Goal: {goal}")
    print("#" * 85)

    for name, planner_class, kwargs in planners:
        print(f"\n---> Algorithm: {name}")
        times = []
        path_lengths = []
        nodes_expanded_list = []
        successes = 0
        last_successful_path = None

        for run in range(num_runs):
            try:
                planner = planner_class(grid_size, obstacles, **kwargs)
                
                t0 = time.perf_counter()
                path = planner.plan(start, goal)
                t1 = time.perf_counter()

                exec_time = t1 - t0
                
                if path is not None:
                    successes += 1
                    p_len = calculate_path_length(path)
                    nodes_exp = extract_nodes_expanded(planner)
                    
                    path_lengths.append(p_len)
                    times.append(exec_time)
                    nodes_expanded_list.append(nodes_exp)
                    last_successful_path = path
                    
                    print(f"  [Run {run+1}/{num_runs}] Success | Time: {exec_time:.4f}s | Length: {p_len:.2f} | Nodes: {nodes_exp}")
                else:
                    print(f"  [Run {run+1}/{num_runs}] Failed | Time: {exec_time:.4f}s")
            except Exception as e:
                print(f"  [Run {run+1}/{num_runs}] ERROR: {e}")
                continue

        avg_time = np.mean(times) if times else float('nan')
        std_time = np.std(times) if times else float('nan')
        avg_len = np.mean(path_lengths) if path_lengths else float('nan')
        avg_nodes = int(np.mean(nodes_expanded_list)) if nodes_expanded_list else 0
        success_rate = (successes / num_runs) * 100

        best_paths[name] = last_successful_path

        raw_metrics.append({
            "Environment": env_name,
            "Algorithm": name,
            "Success Rate (%)": success_rate,
            "Avg Time (s)": avg_time,
            "Std Time (s)": std_time,
            "Avg Path Length": avg_len,
            "Avg Nodes Expanded": avg_nodes
        })

    df = pd.DataFrame(raw_metrics)

    min_optimal_len = df["Avg Path Length"].min()
    df["Sub-optimality (%)"] = df["Avg Path Length"].apply(
        lambda l: round(((l - min_optimal_len) / min_optimal_len) * 100, 2) if not np.isnan(l) else float('nan')
    )

    baseline_time_series = df.loc[df["Algorithm"] == "A* Baseline", "Avg Time (s)"].values
    if len(baseline_time_series) > 0 and not np.isnan(baseline_time_series[0]) and baseline_time_series[0] > 0:
        t_base = baseline_time_series[0]
        df["Speedup vs Base"] = df["Avg Time (s)"].apply(
            lambda t: f"{t_base / t:.2f}x" if not np.isnan(t) and t > 0 else "N/A"
        )
    else:
        df["Speedup vs Base"] = "N/A"

    df_display = df.copy()
    df_display["Avg Time (s)"] = df_display["Avg Time (s)"].map("{:.4f}".format)
    df_display["Std Time (s)"] = df_display["Std Time (s)"].map("{:.4f}".format)
    df_display["Avg Path Length"] = df_display["Avg Path Length"].map("{:.2f}".format)

    print("\n" + "=" * 95)
    print(f"          RESULTS FOR ENVIRONMENT: [{env_name}]")
    print("=" * 95)
    print(df_display.to_string(index=False))
    print("=" * 95)
    
    plot_3d_paths_for_env(grid_size, obstacles, start, goal, best_paths, env_name)
    
    return df

# ============================================================
# ANALIZA E SHKALLËZUESHMËRISË (Reviewer A, pika 2)
# Mat kohën e ekzekutimit DHE kujtesën peak (MB) ndërsa madhësia e
# rrjetës 3D rritet eksponencialisht (N^3 nyje). Përdor tracemalloc
# për gjurmim të kujtesës Python "peak allocation" gjatë plan().
# ============================================================
import tracemalloc

def run_scalability_analysis(model_loader_fn, grid_sizes=(10, 20, 30, 40, 50, 64, 80),
                              obstacle_density=0.15, num_runs=3, seed=42):
    """
    Ekzekuton A* Baseline, NACHA* (Proposed) dhe JPS (nëse është i disponueshëm)
    në grid me madhësi rritëse dhe densitet konstant pengesash, duke
    regjistruar: kohën mesatare (s), memorien peak (MB), dhe nyjet e zgjeruara.

    Kthen një pandas.DataFrame dhe e ruan si 'scalability_analysis.csv'.
    """
    rows = []
    rng = np.random.RandomState(seed)

    for grid_size in grid_sizes:
        n_total = grid_size ** 3
        n_obstacles = int(n_total * obstacle_density)
        # Pengesa të rastit (të riprodhueshme) përjashtuar start/goal
        start = (0, 0, 0)
        goal = (grid_size - 1, grid_size - 1, grid_size - 1)
        coords = rng.randint(0, grid_size, size=(n_obstacles, 3))
        obstacles = set(tuple(c) for c in coords)
        obstacles.discard(start)
        obstacles.discard(goal)

        # Modeli rilidhet për këtë grid_size specifik (rinormalizim input-i)
        try:
            model = model_loader_fn()
            if model is not None and hasattr(model, "set_grid_size"):
                model.set_grid_size(grid_size)
        except Exception:
            model = None

        candidates = [
            ("A* Baseline", AStarBaseline, {}),
            ("NACHA* (Proposed)", NACHAStar, {"env_model": model, "lambda0": 0.10, "use_adaptive_lambda": True}),
        ]
        if JumpPointSearch is not None:
            candidates.append(("JPS", JumpPointSearch, {}))

        for name, planner_class, kwargs in candidates:
            times, peak_mems, nodes_list, ok = [], [], [], 0
            for _ in range(num_runs):
                try:
                    tracemalloc.start()
                    planner = planner_class(grid_size, obstacles, **kwargs)
                    t0 = time.perf_counter()
                    path = planner.plan(start, goal)
                    t1 = time.perf_counter()
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    if path is not None:
                        ok += 1
                        times.append(t1 - t0)
                        peak_mems.append(peak / (1024 ** 2))  # MB
                        nodes_list.append(extract_nodes_expanded(planner))
                except Exception as e:
                    if tracemalloc.is_tracing():
                        tracemalloc.stop()
                    print(f"  [WARN] Scalability run failed for {name} @ grid={grid_size}: {e}")

            rows.append({
                "Grid Size": grid_size,
                "Total Voxels (N^3)": n_total,
                "Algorithm": name,
                "Success Rate (%)": 100.0 * ok / num_runs,
                "Avg Time (s)": float(np.mean(times)) if times else float("nan"),
                "Peak Memory (MB)": float(np.mean(peak_mems)) if peak_mems else float("nan"),
                "Avg Nodes Expanded": float(np.mean(nodes_list)) if nodes_list else float("nan"),
            })
            print(f"  [Scalability] grid={grid_size:>3} | {name:<18} | "
                  f"time={rows[-1]['Avg Time (s)']:.4f}s | "
                  f"mem={rows[-1]['Peak Memory (MB)']:.2f}MB")

    df_scal = pd.DataFrame(rows)
    df_scal.to_csv("scalability_analysis.csv", index=False)
    print("\n[SAVE] Scalability results saved to: 'scalability_analysis.csv'")
    return df_scal


# ============================================================
# FUNKSIONI PËR NGARKIMIN E MODELIT
# ============================================================
def find_model_file():
    """Kërkon dhe kthen rrugën e skedarit të modelit .pt."""
    possible_dirs = [
        os.path.join(PARENT_DIR, "env_model", "models"),
        os.path.join(CURRENT_DIR, "env_model", "models"),
        os.path.join(CURRENT_DIR, "..", "env_model", "models"),
        os.path.join(CURRENT_DIR, "models"),
        os.path.join(PARENT_DIR, "models"),
    ]
    
    for models_dir in possible_dirs:
        if os.path.exists(models_dir):
            files = [f for f in os.listdir(models_dir) if f.endswith('.pt') or f.endswith('.pth')]
            if files:
                return os.path.join(models_dir, files[0])
    return None

def load_model():
    """Ngarkon modelin PyTorch .pt."""
    model_path = find_model_file()
    
    if model_path and os.path.exists(model_path):
        print(f"[INFO] Modeli PyTorch (.pt) u ngarkua nga: {model_path}")
        try:
            return EnvironmentModel(model_path)
        except Exception as e:
            print(f"[ERROR] Ngarkimi i modelit: {e}")
            return None
    else:
        print("[WARN] Modeli .pt nuk u gjet. Përdoren vlerat fallback.")
        return None

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse

    _cli = argparse.ArgumentParser(description="Benchmark runner (all environments)")
    _cli.add_argument('--grid', type=int, default=50, help="Grid size for main benchmark")
    _cli.add_argument('--runs', type=int, default=5, help="Runs per environment")
    _cli.add_argument('--no-movingai', action='store_true', help="Exclude MovingAI environments")
    _cli.add_argument('--scalability', action='store_true',
                       help="Run additional grid-size scalability/memory analysis (Reviewer A)")
    _cli.add_argument('--scalability-grids', type=str, default="10,20,30,40,50,64,80",
                       help="Comma-separated grid sizes for scalability analysis")
    _args, _ = _cli.parse_known_args()

    # ============================================================
    # KONFIGURIMI
    # ============================================================
    GRID_SIZE = _args.grid  # Kjo mund të ndryshohet nga run_all_benchmarks.py --fix-grid
    DEFAULT_START = (0, 0, 0)
    DEFAULT_GOAL = (GRID_SIZE - 1, GRID_SIZE - 1, GRID_SIZE - 1)
    NUM_RUNS = _args.runs
    INCLUDE_MOVINGAI = not _args.no_movingai
    
    print("=" * 95)
    print("BENCHMARK RUNNER - ALL ENVIRONMENTS")
    print("=" * 95)
    print(f"  Grid Size: {GRID_SIZE}x{GRID_SIZE}x{GRID_SIZE}")
    print(f"  Runs per environment: {NUM_RUNS}")
    print(f"  Include MovingAI: {INCLUDE_MOVINGAI}")
    print("=" * 95)
    
    print("\n[INFO] Loading environments...")
    patterns = get_all_environments(seed=42, grid_size=GRID_SIZE, include_movingai=INCLUDE_MOVINGAI)
    
    print(f"\n[OK] Total environments loaded: {len(patterns)}")
    
    base_envs = [k for k in patterns.keys() if '+' not in k and not k.startswith('movingai_') and not k.startswith('stanford_')]
    mixed_envs = [k for k in patterns.keys() if '+' in k]
    stanford_envs = [k for k in patterns.keys() if k.startswith('stanford_')]
    movingai_envs = [k for k in patterns.keys() if k.startswith('movingai_')]
    
    print(f"  - Base environments: {len(base_envs)}")
    print(f"  - Mixed environments: {len(mixed_envs)}")
    print(f"  - Stanford environments: {len(stanford_envs)}")
    print(f"  - MovingAI environments: {len(movingai_envs)}")
    
    print("\n[INFO] Loading model...")
    model = load_model()
    
    # ------------------------------------------------------------------
    # Lista e planner-ave. U shtuan (Reviewer A):
    #   1) JPS dhe D* Lite si baza krahasimi nga literatura klasike/moderne.
    #   2) "NACHA* (Ablation: No Neural)" = NACHA* me lambda0=0, çka e
    #      redukton saktësisht te f(s) = g(s) + w*h(s) gjeometrike (pa
    #      kontributin e rrjetit nervor). Ky variant izolon empirikisht
    #      kontributin e heuristikës/kostos neurale kundrejt bazës.
    # DQN mbetet OPSIONAL (INCLUDE_RL_BASELINE) sepse pa para-trajnim
    # afatgjatë specifik për çdo mjedis nuk konvergjon; e trajtojmë si
    # kufizim i njohur dhe e diskutojmë në përgjigjen ndaj recensuesve.
    # ------------------------------------------------------------------
    INCLUDE_RL_BASELINE = False  # vendos True vetëm nëse ke kohë/burime për para-trajnim DQN

    planners_to_test = [
        ("A* Baseline", AStarBaseline, {}),
        ("NACHA* (Proposed)", NACHAStar, {"env_model": model, "lambda0": 0.10, "use_adaptive_lambda": True}),
        ("NACHA* (Ablation: No Neural)", NACHAStar, {"env_model": model, "lambda0": 0.0, "use_adaptive_lambda": False}),
        ("Neural A*", AStarNeural, {"model": model, "lambda0": 0.10}),
        ("BIT*", BITStar, {"max_iter": 1000, "batch_size": 50}),
        ("RRT*", RRTStar, {"max_iter": 1000, "step_size": 1.0, "goal_bias": 0.10, "timeout": 2.0}),
    ]

    if JumpPointSearch is not None:
        planners_to_test.append(("JPS", JumpPointSearch, {}))
    else:
        print("[WARN] JumpPointSearch nuk u gjet, do të anashkalohet.")

    if DStarLite is not None:
        planners_to_test.append(("D* Lite", DStarLite, {}))
    else:
        print("[WARN] DStarLite nuk u gjet, do të anashkalohet.")

    if INCLUDE_RL_BASELINE and DQNPlanner is not None:
        planners_to_test.append(("DQN (RL)", DQNPlanner, {"use_pretrained": False}))
    
    all_env_results = []
    env_counter = 0
    total_envs = len(patterns)
    
    print("\n" + "=" * 95)
    print("RUNNING BENCHMARK FOR ALL ENVIRONMENTS")
    print("=" * 95)
    
    for env_name, raw_obstacles in patterns.items():
        env_counter += 1
        print(f"\n[Env] {env_counter}/{total_envs}: {env_name}")
        
        obstacles = set(raw_obstacles)
        obstacles.discard(DEFAULT_START)
        obstacles.discard(DEFAULT_GOAL)
        
        if len(obstacles) > GRID_SIZE ** 3 * 0.9:
            print(f"  [WARN] Too many obstacles ({len(obstacles)}), skipping...")
            continue

        start = get_valid_point(DEFAULT_START, obstacles, GRID_SIZE)
        goal = get_valid_point(DEFAULT_GOAL, obstacles, GRID_SIZE)

        try:
            df_env = run_benchmark_for_environment(
                env_name=env_name,
                planners=planners_to_test,
                grid_size=GRID_SIZE,
                obstacles=obstacles,
                start=start,
                goal=goal,
                num_runs=NUM_RUNS
            )
            all_env_results.append(df_env)
        except Exception as e:
            print(f"  [ERROR] Benchmark failed for {env_name}: {e}")
            continue
    
    if all_env_results:
        final_df = pd.concat(all_env_results, ignore_index=True)
        
        csv_filename = f"benchmark_all_environments_results_grid{GRID_SIZE}.csv"
        final_df.to_csv(csv_filename, index=False)
        print(f"\n[SAVE] Results saved to: '{csv_filename}'")
        
        try:
            excel_filename = f"benchmark_all_environments_results_grid{GRID_SIZE}.xlsx"
            final_df.to_excel(excel_filename, index=False)
            print(f"[SAVE] Results saved to: '{excel_filename}'")
        except Exception:
            pass
        
        print("\n" + "=" * 95)
        print("SUMMARY")
        print("=" * 95)
        
        summary = final_df.groupby("Algorithm").agg({
            "Success Rate (%)": "mean",
            "Avg Time (s)": "mean",
            "Avg Path Length": "mean",
            "Avg Nodes Expanded": "mean"
        }).round(2)
        print("\nAverage performance by algorithm:")
        print(summary.to_string())
        
        print(f"\nTotal environments tested: {len(all_env_results)}")
    else:
        print("\n[ERROR] No benchmarks executed!")

    print("\n" + "=" * 95)
    print("BENCHMARK COMPLETED")
    print("=" * 95)

    # ============================================================
    # ANALIZA E SHKALLËZUESHMËRISË (opsionale, Reviewer A pika 2)
    # ============================================================
    if _args.scalability:
        print("\n" + "=" * 95)
        print("SCALABILITY & MEMORY ANALYSIS (grid size sweep)")
        print("=" * 95)
        grid_list = [int(g.strip()) for g in _args.scalability_grids.split(",") if g.strip()]
        run_scalability_analysis(model_loader_fn=load_model, grid_sizes=tuple(grid_list))
