"""
run_all_benchmarks.py
Skript i unifikuar për të ekzekutuar TË GJITHA proceset në rend të saktë:

1. Trajnimi i modeleve (train_all_models_backprop.py)
2. Benchmark-u i plote (benchmark_runner_environments_with_visualization.py)
3. Testimi i të gjitha mjediseve (test_all_environments_with_visualization.py)

PËRDORIMI:
    python run_all_benchmarks.py                   # Ekzekuton të gjitha me konfigurim default
    python run_all_benchmarks.py --grid 50         # Me grid=50
    python run_all_benchmarks.py --skip-train      # Pa trajnim
    python run_all_benchmarks.py --skip-test       # Pa test
    python run_all_benchmarks.py --env cube        # Vetëm për një mjedis specifik
    python run_all_benchmarks.py --help            # Ndihmë
"""

import os
import sys
import subprocess
import time
import argparse
import json
import io
from datetime import datetime

# ============================================================
# FORCO UTF-8 PËR WINDOWS
# ============================================================
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# ============================================================
# KONFIGURIMI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Vendos path për importe
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'env_model'))
sys.path.insert(0, os.path.join(BASE_DIR, 'env_model', 'envs'))
sys.path.insert(0, os.path.join(BASE_DIR, 'nacha_planner'))
sys.path.insert(0, os.path.join(BASE_DIR, 'planners'))
sys.path.insert(0, os.path.join(BASE_DIR, 'datasets'))
sys.path.insert(0, os.path.join(BASE_DIR, 'models'))

# ============================================================
# FUNKSIONET NDIHMËSE
# ============================================================
def print_header(title):
    """Printo një header të bukur."""
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)

def print_step(step_num, total_steps, description):
    """Printo hapin aktual."""
    print(f"\n{'─' * 90}")
    print(f"  [HAPI {step_num}/{total_steps}] {description}")
    print(f"{'─' * 90}")
    print(f"  Fillimi: {datetime.now().strftime('%H:%M:%S')}")
    print()

def run_command(cmd, description, timeout=None):
    """
    Ekzekuton një komandë dhe printon output-in në real kohë.
    
    Args:
        cmd (list): Komanda dhe argumentet
        description (str): Përshkrimi për printim
        timeout (int): Timeout në sekonda (None = pa timeout)
    
    Returns:
        bool: True nëse sukses, False nëse dështon
    """
    print(f"  Ekzekutimi: {' '.join(cmd)}")
    print()
    
    try:
        # Përdor python (jo pythonw) për të parë output-in
        if cmd[0].endswith('pythonw.exe'):
            cmd[0] = cmd[0].replace('pythonw.exe', 'python.exe')
        
        # Ekzekuto me pipe për të kapur output në real kohë
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Lexo dhe printo output-in në real kohë
        for line in process.stdout:
            print(f"  {line}", end='')
        
        # Prit për përfundim
        process.wait()
        
        if process.returncode == 0:
            print(f"\n  [OK] {description} - PËRFUNDOI ME SUKSES!")
            return True
        else:
            print(f"\n  [ERROR] {description} - DËSHTUAM (kodi: {process.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"\n  [TIMEOUT] {description} - TIMEOUT ({timeout}s)")
        return False
    except Exception as e:
        print(f"\n  [ERROR] {description} - GABIM: {e}")
        return False

def find_file(filename, search_dirs):
    """
    Kërkon një skedar në disa drejtori.
    
    Args:
        filename (str): Emri i skedarit
        search_dirs (list): Lista e drejtorive për të kërkuar
        
    Returns:
        str: Rruga e plotë e skedarit ose None
    """
    for dir_path in search_dirs:
        full_path = os.path.join(dir_path, filename)
        if os.path.exists(full_path):
            return full_path
    return None

def update_file_content(filepath, old_text, new_text):
    """Përditëson përmbajtjen e një skedari."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_text in content:
            content = content.replace(old_text, new_text)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
    except Exception as e:
        print(f"  [WARN] Nuk mund të përditësohet {filepath}: {e}")
    return False

# ============================================================
# KRYESORE
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run all benchmarks in sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SHEMBUJ:
  python run_all_benchmarks.py                    # Ekzekuton të gjitha
  python run_all_benchmarks.py --grid 50         # Me grid=50
  python run_all_benchmarks.py --skip-train      # Pa trajnim
  python run_all_benchmarks.py --skip-test       # Pa test
  python run_all_benchmarks.py --env cube        # Vetëm për mjedis specifik
  python run_all_benchmarks.py --runs 3          # 3 runs për benchmark
        """
    )
    
    parser.add_argument('--grid', type=int, default=20,
                        help='Grid size (20, 50, etc.) [default: 20]')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs per environment [default: 5]')
    parser.add_argument('--env', type=str, default=None,
                        help='Test only specific environment')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip model training')
    parser.add_argument('--skip-benchmark', action='store_true',
                        help='Skip benchmark')
    parser.add_argument('--skip-test', action='store_true',
                        help='Skip visualization test')
    parser.add_argument('--no-movingai', action='store_true',
                        help='Exclude MovingAI environments')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout per step in seconds [default: 3600]')
    parser.add_argument('--fix-grid', action='store_true',
                        help='Fix GRID_SIZE in benchmark script to match --grid')
    
    args = parser.parse_args()
    
    # ============================================================
    # HEADER
    # ============================================================
    print_header("RUNNING ALL BENCHMARKS - SEQUENTIAL EXECUTION")
    print(f"\nKONFIGURIMI:")
    print(f"  Grid Size: {args.grid}x{args.grid}x{args.grid}")
    print(f"  Runs per environment: {args.runs}")
    print(f"  Include MovingAI: {not args.no_movingai}")
    print(f"  Skip Training: {args.skip_train}")
    print(f"  Skip Benchmark: {args.skip_benchmark}")
    print(f"  Skip Test: {args.skip_test}")
    print(f"  Visualization: {'Disabled' if args.no_viz else 'Enabled'}")
    print(f"  Timeout per step: {args.timeout}s")
    print(f"\nDrejtoria: {BASE_DIR}")
    print(f"Fillimi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ============================================================
    # FIX GRID_SIZE NË benchmark_runner_environments_with_visualization.py
    # ============================================================
    if args.fix_grid:
        benchmark_script = os.path.join(BASE_DIR, 'benchmark_runner_environments_with_visualization.py')
        if os.path.exists(benchmark_script):
            print_header(f"FIXING GRID_SIZE TO {args.grid} IN benchmark_runner")
            old_text = f'GRID_SIZE = 50'
            new_text = f'GRID_SIZE = {args.grid}'
            if update_file_content(benchmark_script, old_text, new_text):
                print(f"  [OK] GRID_SIZE updated from 50 to {args.grid}")
            else:
                print(f"  [WARN] GRID_SIZE not found or already updated")
    
    # ============================================================
    # KONTROLLO SKEDARËT E NEVOJSHËM
    # ============================================================
    print_header("VERIFIKIMI I SKEDAREVE")
    
    # Përcakto drejtoritë ku mund të jenë skedarët
    search_dirs = [
        BASE_DIR,
        os.path.join(BASE_DIR, 'env_model'),
        os.path.join(BASE_DIR, 'env_model', 'envs'),
        os.path.join(BASE_DIR, 'models'),
        os.path.join(BASE_DIR, 'planners'),
        os.path.join(BASE_DIR, 'nacha_planner'),
        os.path.join(BASE_DIR, 'datasets'),
    ]
    
    required_files = [
        ('train_all_models_backprop.py', search_dirs),
        ('benchmark_runner_environments_with_visualization.py', search_dirs),
        ('test_all_environments_with_visualization.py', search_dirs),
        ('obstacle_manager_3d.py', [
            os.path.join(BASE_DIR, 'env_model', 'envs'),
            os.path.join(BASE_DIR, 'envs'),
            BASE_DIR,
        ]),
        ('movingai_loader.py', [
            os.path.join(BASE_DIR, 'datasets'),
            os.path.join(BASE_DIR, 'movingai'),
            BASE_DIR,
        ]),
        ('nacha_star.py', [
            os.path.join(BASE_DIR, 'nacha_planner'),
            BASE_DIR,
        ]),
        ('neural_cost_function.py', [
            os.path.join(BASE_DIR, 'nacha_planner'),
            BASE_DIR,
        ]),
        ('astar_baseline.py', [
            os.path.join(BASE_DIR, 'planners'),
            BASE_DIR,
        ]),
        ('astar_neural.py', [
            os.path.join(BASE_DIR, 'planners'),
            BASE_DIR,
        ]),
        ('rrt_star.py', [
            os.path.join(BASE_DIR, 'planners'),
            BASE_DIR,
        ]),
        ('bit_star.py', [
            os.path.join(BASE_DIR, 'planners'),
            BASE_DIR,
        ]),
        ('astar_all.py', [
            os.path.join(BASE_DIR, 'planners'),
            BASE_DIR,
        ]),
        ('astar_base.py', [
            os.path.join(BASE_DIR, 'planners'),
            BASE_DIR,
        ]),
        ('env_model_loader.py', [
            os.path.join(BASE_DIR, 'models'),
            os.path.join(BASE_DIR, 'env_model'),
            BASE_DIR,
        ]),
    ]
    
    missing = []
    found_files = {}
    
    for filename, dirs in required_files:
        found = find_file(filename, dirs)
        if found:
            print(f"  [OK] {found}")
            found_files[filename] = found
        else:
            print(f"  [MISSING] {filename} - NUK U GJET!")
            missing.append(filename)
    
    if missing:
        print(f"\n[WARN] {len(missing)} skedarë të munguar!")
        for f in missing:
            print(f"    - {f}")
        print("\n  Ju lutemi sigurohuni që të gjithë skedarët ekzistojnë.")
        
        # Pyet nëse duhet të vazhdojë gjithsesi
        response = input("\n  Vazhdo gjithsesi? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print(f"\n[OK] TE GJITHE skedarët e nevojshëm ekzistojnë!")
    
    # ============================================================
    # START
    # ============================================================
    total_steps = 0
    if not args.skip_train:
        total_steps += 1
    if not args.skip_benchmark:
        total_steps += 1
    if not args.skip_test:
        total_steps += 1
    
    current_step = 0
    success_count = 0
    
    results = {
        'config': {
            'grid_size': args.grid,
            'runs': args.runs,
            'include_movingai': not args.no_movingai,
            'timeout': args.timeout,
            'env': args.env,
            'start_time': datetime.now().isoformat()
        },
        'steps': []
    }
    
    # ============================================================
    # HAPI 1: TRAJNIMI
    # ============================================================
    if not args.skip_train:
        current_step += 1
        print_step(current_step, total_steps, "TRAJNIMI I MODELEVE")
        
        train_script = found_files.get('train_all_models_backprop.py', 'train_all_models_backprop.py')
        cmd = [
            sys.executable,
            train_script,
            f'--grid={args.grid}',
            f'--epochs=150',
            f'--samples=2000'
        ]
        
        if args.no_movingai:
            cmd.append('--no-movingai')
        
        if args.env:
            cmd.append(f'--env={args.env}')
        
        success = run_command(cmd, "TRAJNIMI", timeout=args.timeout)
        
        results['steps'].append({
            'name': 'training',
            'success': success,
            'cmd': ' '.join(cmd),
            'time': datetime.now().isoformat()
        })
        
        if success:
            success_count += 1
            print("\n  [OK] Model training completed successfully!")
        else:
            print("\n  [WARN] Model training failed or was interrupted.")
    
    # ============================================================
    # HAPI 2: BENCHMARK
    # ============================================================
    if not args.skip_benchmark:
        current_step += 1
        print_step(current_step, total_steps, "BENCHMARK RUNNER")
        
        benchmark_script = found_files.get('benchmark_runner_environments_with_visualization.py', 
                                          'benchmark_runner_environments_with_visualization.py')
        cmd = [
            sys.executable,
            benchmark_script
        ]
        
        success = run_command(cmd, "BENCHMARK RUNNER", timeout=args.timeout * 2)
        
        results['steps'].append({
            'name': 'benchmark',
            'success': success,
            'cmd': ' '.join(cmd),
            'time': datetime.now().isoformat()
        })
        
        if success:
            success_count += 1
            print("\n  [OK] Benchmark completed successfully!")
        else:
            print("\n  [WARN] Benchmark failed or was interrupted.")
    
    # ============================================================
    # HAPI 3: TEST I VIZUALIZIMIT
    # ============================================================
    if not args.skip_test:
        current_step += 1
        print_step(current_step, total_steps, "TEST I VIZUALIZIMIT (63 MJEDISE)")
        
        test_script = found_files.get('test_all_environments_with_visualization.py',
                                     'test_all_environments_with_visualization.py')
        cmd = [
            sys.executable,
            test_script,
            f'--grid={args.grid}',
            f'--runs={args.runs}'
        ]
        
        if args.no_movingai:
            cmd.append('--no-movingai')
        
        if args.no_viz:
            cmd.append('--no-viz')
        
        if args.env:
            cmd.append(f'--env={args.env}')
        
        success = run_command(cmd, "TEST I VIZUALIZIMIT", timeout=args.timeout)
        
        results['steps'].append({
            'name': 'visualization_test',
            'success': success,
            'cmd': ' '.join(cmd),
            'time': datetime.now().isoformat()
        })
        
        if success:
            success_count += 1
            print("\n  [OK] Visualization test completed successfully!")
        else:
            print("\n  [WARN] Visualization test failed or was interrupted.")
    
    # ============================================================
    # PËRMBLEDHJA
    # ============================================================
    print_header("PËRMBLEDHJA PËRFUNDIMTARE")
    
    print(f"\n  Rezultatet:")
    print(f"    Steps executed: {current_step}")
    print(f"    Success: {success_count}")
    print(f"    Failed: {current_step - success_count}")
    
    print("\n  Skedarët e rezultateve të krijuara:")
    
    result_files = []
    
    # Kontrollo skedarët CSV të benchmark
    csv_files = [f for f in os.listdir('.') if f.startswith('benchmark_all_environments_results') and f.endswith('.csv')]
    if csv_files:
        for f in csv_files:
            size = os.path.getsize(f) / 1024
            print(f"    [OK] {f} ({size:.1f} KB)")
            result_files.append(f)
    else:
        print(f"    [WARN] No CSV result files found")
    
    # Kontrollo skedarët e figurës
    fig_files = [f for f in os.listdir('.') if f.startswith('3d_paths_') and f.endswith('.png')]
    if fig_files:
        for f in fig_files[:5]:
            size = os.path.getsize(f) / 1024
            print(f"    [OK] {f} ({size:.1f} KB)")
        if len(fig_files) > 5:
            print(f"    ... dhe {len(fig_files) - 5} të tjera")
            result_files.extend(fig_files[:5])
    else:
        print(f"    [WARN] No 3D path figures found")
    
    # Kontrollo figurat e testit
    test_fig_dirs = [d for d in os.listdir('.') if d.startswith('results/figures_all_envs')]
    if test_fig_dirs:
        for d in test_fig_dirs:
            print(f"    [OK] {d}/")
    
    # Kontrollo skedarët e trajnimit
    train_files = [f for f in os.listdir('.') if f.startswith('train_config_') and f.endswith('.json')]
    if train_files:
        for f in train_files[:2]:
            size = os.path.getsize(f) / 1024
            print(f"    [OK] {f} ({size:.1f} KB)")
        if len(train_files) > 2:
            print(f"    ... dhe {len(train_files) - 2} të tjera")
    
    # Ruaj rezultatet në JSON
    results['summary'] = {
        'total_steps': current_step,
        'success_count': success_count,
        'failed_count': current_step - success_count,
        'end_time': datetime.now().isoformat(),
        'duration_seconds': (datetime.now() - datetime.fromisoformat(results['config']['start_time'])).total_seconds()
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"run_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n[SAVE] Rezultatet u ruajtën te: {results_file}")
    
    # ============================================================
    # STATISTIKAT
    # ============================================================
    print_header("STATISTIKAT PËRFUNDIMTARE")
    
    duration = results['summary']['duration_seconds']
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print(f"\n  Koha totale: {hours}h {minutes}m {seconds}s")
    print(f"  Steps: {success_count}/{current_step} sukses")
    
    if success_count == current_step:
        print("\n  [SUCCESS] TE GJITHA HAPAT PËRFUNDUAN ME SUKSES!")
    else:
        print("\n  [WARN] DISA HAPA DËSHTUAN. Kontrolloni log-et më lart.")
    
    print("\n" + "=" * 90)
    print("PËRFUNDOI!")
    print("=" * 90)
    
    # Kthe kod daljeje
    return 0 if success_count == current_step else 1


# ============================================================
# EKZEKUTIMI
# ============================================================
if __name__ == "__main__":
    sys.exit(main())
