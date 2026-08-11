"""
train_all_models_backprop.py
Train all environment models using backpropagation - PER TE GJITHA 63 MJEDISET
(Versioni i Përditësuar me ObstacleManager të ri dhe MovingAI)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time
import random
import json
from datetime import datetime
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
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'env_model'))
sys.path.insert(0, os.path.join(BASE_DIR, 'env_model', 'envs'))
sys.path.insert(0, os.path.join(BASE_DIR, 'datasets'))

# Përdor GPU nëse është e disponueshme
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ============================================================
# IMPORTET
# ============================================================
# Përdor obstacle_manager_3d.py të përditësuar
try:
    from obstacle_manager_3d import ObstacleManager
except ImportError:
    try:
        from env_model.envs.obstacle_manager_3d import ObstacleManager
    except ImportError:
        print("[ERROR] Cannot import ObstacleManager!")
        sys.exit(1)

# Import për MovingAI (44 mjedise)
try:
    from datasets.movingai_loader import load_all_movingai_maps
except ImportError:
    try:
        from movingai_loader import load_all_movingai_maps
    except ImportError:
        def load_all_movingai_maps(grid_size=20):
            return {}

# ============================================================
# FUNKSIONI PËR TË GJITHA MJEDISET
# ============================================================
def get_all_environments(seed=42, grid_size=20, include_movingai=True):
    """
    Merr të gjitha mjediset e disponueshme (63 total).
    """
    patterns = {}
    
    # 1. Mjediset nga ObstacleManager (8 + 7 + 4 = 19)
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
    
    # 2. Mjediset MovingAI (44)
    if include_movingai:
        try:
            movingai_maps = load_all_movingai_maps(grid_size=grid_size)
            patterns.update(movingai_maps)
            print(f"[INFO] Ngarkova {len(movingai_maps)} mjedise MovingAI")
        except Exception as e:
            print(f"[WARN] Gabim gjatë ngarkimit të MovingAI: {e}")
    
    return patterns

# ============================================================
# CONFIG
# ============================================================
GRID_SIZE = 20
START = (0, 0, 0)
END = (GRID_SIZE - 1, GRID_SIZE - 1, GRID_SIZE - 1)

SAMPLES_PER_CLASS = 2000
TRAIN_SPLIT = 0.8
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.001
INCLUDE_MOVINGAI = True

RESULTS_DIR = "training_results"
MODEL_DIR = "env_model/models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def get_config_filename():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"train_config_S{SAMPLES_PER_CLASS}_B{BATCH_SIZE}_E{EPOCHS}_{timestamp}.json"

CONFIG_FILE = os.path.join(RESULTS_DIR, get_config_filename())

# ============================================================
# DATA GENERATION
# ============================================================
def generate_dataset(obstacles, grid_size=GRID_SIZE, samples_per_class=SAMPLES_PER_CLASS):
    """Gjeneron dataset të balancuar për trajnim."""
    obstacle_set = set(obstacles)
    all_points = [(x, y, z) for x in range(grid_size)
                            for y in range(grid_size)
                            for z in range(grid_size)]

    obstacle_points = [p for p in all_points if p in obstacle_set]
    free_points = [p for p in all_points if p not in obstacle_set and p not in [START, END]]

    if not obstacle_points or not free_points:
        return [], []

    n_samples = min(samples_per_class, len(obstacle_points), len(free_points))
    
    if len(obstacle_points) < samples_per_class:
        sampled_obs = random.choices(obstacle_points, k=n_samples)
    else:
        sampled_obs = random.sample(obstacle_points, n_samples)
        
    sampled_free = random.sample(free_points, n_samples)

    X = sampled_obs + sampled_free
    Y = [1.0] * n_samples + [0.0] * n_samples

    data = list(zip(X, Y))
    random.shuffle(data)

    X, Y = zip(*data)
    return list(X), list(Y)

# ============================================================
# NEURAL NETWORK
# ============================================================
class ImprovedNN(nn.Module):
    def __init__(self, input_size=3, hidden_sizes=[128, 64, 32], output_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# ============================================================
# TRAIN FUNCTION
# ============================================================
def train_model_backprop(obstacles, pattern_name, grid_size=GRID_SIZE, epochs=EPOCHS, verbose=True):
    if verbose:
        print(f"\n[Train] Training model for: {pattern_name} (obstacles: {len(obstacles)}) | Device: {DEVICE}")
    
    X, Y = generate_dataset(obstacles, grid_size=grid_size)
    
    if len(X) == 0:
        print(f"  [WARN] No valid data for {pattern_name}, skipping...")
        return None, None
    
    X_arr = np.array(X, dtype=np.float32) / (grid_size - 1.0)
    Y_arr = np.array(Y, dtype=np.float32).reshape(-1, 1)
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_arr, Y_arr, test_size=(1 - TRAIN_SPLIT), random_state=42, stratify=Y_arr
    )
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(DEVICE)
    
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = ImprovedNN(input_size=3, hidden_sizes=[128, 64, 32], output_size=1).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)
    
    start_time = time.time()
    best_val_acc = 0.0
    best_model_state = None
    training_history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_preds_bin = (val_outputs > 0.5).float()
                val_acc = (val_preds_bin == Y_val_t).float().mean().item()
                
                scheduler.step(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                
                avg_loss = total_loss / len(train_loader)
                training_history.append({
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'val_accuracy': val_acc,
                    'best_val_accuracy': best_val_acc
                })
                
                if verbose and (epoch + 1) % 50 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Val Acc={val_acc:.3f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    elapsed = time.time() - start_time
    
    model_path = os.path.join(MODEL_DIR, f"env_model_{pattern_name}_backprop.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_sizes': [128, 64, 32],
        'input_size': 3,
        'output_size': 1,
        'grid_size': grid_size,
        'val_accuracy': best_val_acc,
        'training_history': training_history
    }, model_path)
    
    if verbose:
        print(f"  [SAVE] Saved: {model_path} | Val Acc: {best_val_acc:.3f} | Time: {elapsed:.2f}s")
    
    model_info = {
        'pattern_name': pattern_name,
        'obstacles': len(obstacles),
        'grid_size': grid_size,
        'best_val_accuracy': best_val_acc,
        'training_time': elapsed,
        'epochs': epochs,
        'samples_per_class': SAMPLES_PER_CLASS,
        'batch_size': BATCH_SIZE,
        'train_split': TRAIN_SPLIT,
        'learning_rate': LEARNING_RATE,
        'model_path': model_path,
        'training_history': training_history
    }
    
    return model, model_info

# ============================================================
# TRAIN ALL PATTERNS
# ============================================================
def train_all_patterns(patterns_to_train=None, grid_size=GRID_SIZE, include_movingai=INCLUDE_MOVINGAI):
    print("\n" + "="*70)
    print("TRAINING ALL MODELS WITH BACKPROPAGATION (63 ENVIRONMENTS)")
    print("="*70)
    print(f"  Grid Size: {grid_size}x{grid_size}x{grid_size}")
    print(f"  Device: {DEVICE}")
    print(f"  Include MovingAI: {include_movingai}")
    print("="*70)
    
    patterns = get_all_environments(seed=42, grid_size=grid_size, include_movingai=include_movingai)
    
    if patterns_to_train is not None:
        patterns = {k: v for k, v in patterns.items() if k in patterns_to_train}
    
    print(f"\n[INFO] Total environments to train: {len(patterns)}")
    
    base = [k for k in patterns.keys() if '+' not in k and not k.startswith('movingai_') and not k.startswith('stanford_')]
    mixed = [k for k in patterns.keys() if '+' in k]
    stanford = [k for k in patterns.keys() if k.startswith('stanford_')]
    movingai = [k for k in patterns.keys() if k.startswith('movingai_')]
    
    print(f"  - Base environments: {len(base)}")
    print(f"  - Mixed environments: {len(mixed)}")
    print(f"  - Stanford environments: {len(stanford)}")
    print(f"  - MovingAI environments: {len(movingai)}")
    
    all_models_info = []
    results = {}
    env_counter = 0
    total_envs = len(patterns)
    
    for pattern_name, obstacles in patterns.items():
        env_counter += 1
        print(f"\n[Train] [{env_counter}/{total_envs}] {pattern_name}")
        
        if len(obstacles) == 0:
            print(f"  [WARN] Skipping {pattern_name} (empty environment)")
            continue
        
        try:
            model, model_info = train_model_backprop(
                obstacles, 
                pattern_name, 
                grid_size=grid_size,
                verbose=True
            )
            if model_info:
                all_models_info.append(model_info)
                results[pattern_name] = model_info
        except Exception as e:
            print(f"  [ERROR] Error training {pattern_name}: {e}")
            continue
    
    save_training_info(all_models_info, grid_size)
    return results

def save_training_info(all_models_info, grid_size=GRID_SIZE):
    config_info = {
        'config': {
            'samples_per_class': SAMPLES_PER_CLASS,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'train_split': TRAIN_SPLIT,
            'learning_rate': LEARNING_RATE,
            'grid_size': grid_size,
            'device': str(DEVICE),
            'include_movingai': INCLUDE_MOVINGAI,
            'total_environments': len(all_models_info),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'models': all_models_info
    }
    
    if all_models_info:
        val_accuracies = [m['best_val_accuracy'] for m in all_models_info]
        training_times = [m['training_time'] for m in all_models_info]
        
        config_info['summary'] = {
            'total_models': len(all_models_info),
            'average_val_accuracy': float(np.mean(val_accuracies)),
            'min_val_accuracy': float(np.min(val_accuracies)),
            'max_val_accuracy': float(np.max(val_accuracies)),
            'total_training_time': float(np.sum(training_times)),
            'average_training_time': float(np.mean(training_times))
        }
    
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, default=str)
    
    print(f"\n[SAVE] Training info saved to: {CONFIG_FILE}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train all environment models")
    parser.add_argument('--grid', type=int, default=20, help="Grid size (20, 50, etc.)")
    parser.add_argument('--no-movingai', action='store_true', help="Exclude MovingAI environments")
    parser.add_argument('--env', type=str, default=None, help="Train only specific environment")
    parser.add_argument('--epochs', type=int, default=150, help="Number of epochs")
    parser.add_argument('--samples', type=int, default=2000, help="Samples per class")
    
    args = parser.parse_args()
    
    GRID_SIZE = args.grid
    EPOCHS = args.epochs
    SAMPLES_PER_CLASS = args.samples
    INCLUDE_MOVINGAI = not args.no_movingai
    
    if args.env:
        patterns_to_train = [args.env]
    else:
        patterns_to_train = None
    
    train_all_patterns(
        patterns_to_train=patterns_to_train,
        grid_size=GRID_SIZE,
        include_movingai=INCLUDE_MOVINGAI
    )
    
    print("\n" + "="*70)
    print("[OK] TRAINING COMPLETED")
    print("="*70)