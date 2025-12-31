#!/usr/bin/env python3
"""
Multi-seed training script for Gaussian initial condition.

Usage:
    python train_gaussian.py                     # Run all seeds
    python train_gaussian.py --seeds 42 43      # Run specific seeds
    python train_gaussian.py --dry-run          # Show configuration only
"""

import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'

import argparse
import time
import pickle
import numpy as np
import deepxde as dde
import tensorflow as tf
from scipy.special import erf
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

SEEDS = [42, 43, 44, 45, 46]  # 5 seeds

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 11 levels

GRID_CONFIGS = [
    (21, 21),  
    (21, 51),
    (51, 21),
    (51, 51),
    (51, 101)     
]

MODELS = ['P', 'PD']  # Poisson-only, Poisson + Diffusion-Decay

# Training settings
EPOCHS = 5000
DISPLAY_EVERY = 5000
NUM_DOMAIN = 2000

# Physical parameters
EPS = 1.0
D_COEFF = 0.01
K_COEFF = 0.5

# Gaussian parameters
Q0 = 1.0
SIGMA0 = 0.1

# Domain
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

# Output directory
SAVE_DIR = './results/gaussian'


# =============================================================================
# Helper Functions
# =============================================================================

def generate_gaussian_data(x_grid, t_grid):
    """
    Generate analytical solution for Gaussian initial condition.

    ρ(x,t) = Q(t)/(√(2π)σ(t)) * exp(-x²/(2σ(t)²))
    where:
        σ(t) = √(σ0² + 2Dt)  (diffusion spreading)
        Q(t) = Q0 * exp(-kt)  (decay)
    """
    X, T = np.meshgrid(x_grid, t_grid, indexing='ij')

    # σ(t) and Q(t)
    sigma_t = np.sqrt(SIGMA0**2 + 2*D_COEFF*T)
    Q_t = Q0 * np.exp(-K_COEFF*T)

    # ρ analytical
    rho_mesh = Q_t / (np.sqrt(2*np.pi) * sigma_t) * np.exp(-X**2 / (2*sigma_t**2))

    # V analytical (Poisson solution)
    z = X / (np.sqrt(2) * sigma_t)
    V_raw = -Q_t / (2*EPS) * (
        X * erf(z) +
        sigma_t * np.sqrt(2/np.pi) * np.exp(-z**2)
    )
    # Mean-zero gauge
    V_mesh = V_raw - np.mean(V_raw, axis=0, keepdims=True)

    return rho_mesh, V_mesh


def add_noise(phi_clean, noise_level, seed):
    """Add relative Gaussian noise."""
    np.random.seed(seed)
    if noise_level == 0:
        return phi_clean.copy()
    phi_scale = np.max(np.abs(phi_clean))
    return phi_clean + noise_level * phi_scale * np.random.randn(*phi_clean.shape)


def get_model_name(model_type, noise, num_x, num_t):
    """Generate model filename."""
    return f"{model_type}_noise{int(noise*100):02d}_nx{num_x}_nt{num_t}"


def relative_L2_error(pred, exact):
    """Compute relative L2 error."""
    return np.sqrt(np.mean((pred - exact)**2)) / np.sqrt(np.mean(exact**2))


# =============================================================================
# PDE Definitions
# =============================================================================

def pde_poisson(X, y):
    """Poisson-only: -∂²φ/∂x² - ρ/ε = 0"""
    phi_xx = dde.grad.hessian(y, X, component=0, i=0, j=0)
    rho = y[:, 1:2]
    return -phi_xx - rho / EPS


def pde_coupled(X, y):
    """Coupled: Poisson + Diffusion-Decay"""
    phi_xx = dde.grad.hessian(y, X, component=0, i=0, j=0)
    rho = y[:, 1:2]
    rho_t = dde.grad.jacobian(y, X, i=1, j=1)
    rho_xx = dde.grad.hessian(y, X, component=1, i=0, j=0)
    return [-phi_xx - rho / EPS, rho_t - D_COEFF * rho_xx + K_COEFF * rho]


# =============================================================================
# Training Function
# =============================================================================

def train_single_model(model_type, noise_level, num_x, num_t, seed, seed_dir):
    """Train a single model and save results."""
    model_name = get_model_name(model_type, noise_level, num_x, num_t)
    save_path = os.path.join(seed_dir, f"{model_name}.pkl")

    # Skip if exists
    if os.path.exists(save_path):
        print(f"  [SKIP] {model_name}")
        return None

    tf.compat.v1.reset_default_graph()
    dde.config.set_random_seed(seed)
    np.random.seed(seed)

    # Generate data
    x_grid = np.linspace(X_MIN, X_MAX, num_x)
    t_grid = np.linspace(T_MIN, T_MAX, num_t)
    X_mesh, T_mesh = np.meshgrid(x_grid, t_grid, indexing='ij')
    XT_data = np.hstack([X_mesh.flatten()[:, None], T_mesh.flatten()[:, None]])

    # Gaussian analytical solution
    rho_ex_mesh, phi_ex_mesh = generate_gaussian_data(x_grid, t_grid)
    phi_ex = phi_ex_mesh.flatten()[:, None]
    rho_ex = rho_ex_mesh.flatten()[:, None]
    phi_obs = add_noise(phi_ex, noise_level, seed)

    # Geometry
    geom = dde.geometry.Interval(X_MIN, X_MAX)
    timedomain = dde.geometry.TimeDomain(T_MIN, T_MAX)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Observation
    observe_phi = dde.icbc.PointSetBC(XT_data, phi_obs, component=0)

    # Model-specific settings
    if model_type == 'P':
        pde_fn = pde_poisson
        loss_weights = [1, 1000]
    else:  # PD
        pde_fn = pde_coupled
        loss_weights = [1, 1, 1000]

    # Data
    data = dde.data.TimePDE(
        geomtime, pde_fn, [observe_phi],
        num_domain=NUM_DOMAIN, num_boundary=0, num_initial=0,
        anchors=XT_data, num_test=5000
    )

    # Network
    net = dde.nn.PFNN(
        [2, [64, 64], [64, 64], [64, 64], [64, 64], 2],
        "tanh", "Glorot uniform"
    )

    # Compile and train
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss_weights=loss_weights)

    start_time = time.time()
    losshistory, _ = model.train(epochs=EPOCHS, display_every=DISPLAY_EVERY)
    train_time = time.time() - start_time

    # Predict
    output = model.predict(XT_data)
    phi_pred = output[:, 0:1]
    rho_pred = output[:, 1:2]

    # Metrics
    phi_L2 = relative_L2_error(phi_pred, phi_ex)
    rho_L2 = relative_L2_error(rho_pred, rho_ex)

    # Save
    result = {
        'model_name': model_name,
        'model_type': model_type,
        'noise_level': noise_level,
        'num_x': num_x,
        'num_t': num_t,
        'seed': seed,
        'train_time': train_time,
        'x_grid': x_grid,
        't_grid': t_grid,
        'XT_data': XT_data,
        'phi_ex': phi_ex,
        'rho_ex': rho_ex,
        'phi_obs': phi_obs,
        'phi_pred': phi_pred,
        'rho_pred': rho_pred,
        'phi_L2': phi_L2,
        'rho_L2': rho_L2,
        'loss_history': np.array(losshistory.loss_train),
    }

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)

    tf.keras.backend.clear_session()

    print(f"  [DONE] {model_name}: rho_L2={rho_L2:.4f}, time={train_time:.1f}s")
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-seed training for Gaussian IC')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help='Seeds to run (default: 42-46)')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR,
                        help='Output directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show configuration only')
    args = parser.parse_args()

    seeds = args.seeds
    save_dir = args.save_dir

    total_models = len(seeds) * len(NOISE_LEVELS) * len(GRID_CONFIGS) * len(MODELS)

    print("=" * 60)
    print("PD-iPINN Multi-Seed Training (Gaussian)")
    print("=" * 60)
    print(f"Seeds: {seeds}")
    print(f"Noise levels: {len(NOISE_LEVELS)} (0-100%)")
    print(f"Grid configs: {len(GRID_CONFIGS)}")
    print(f"Models: {MODELS}")
    print(f"Gaussian: Q0={Q0}, σ0={SIGMA0}")
    print(f"Total: {total_models} models")
    print(f"Output: {save_dir}")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry run] Exiting without training.")
        return

    os.makedirs(save_dir, exist_ok=True)

    global_start = time.time()
    global_done = 0
    global_skip = 0

    for seed_idx, seed in enumerate(seeds):
        seed_dir = os.path.join(save_dir, f'seed{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"SEED {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'='*60}")

        seed_start = time.time()
        seed_done = 0
        seed_skip = 0

        for noise in NOISE_LEVELS:
            print(f"\n--- Noise: {int(noise*100)}% ---")

            for (num_x, num_t) in GRID_CONFIGS:
                for model_type in MODELS:
                    result = train_single_model(
                        model_type, noise, num_x, num_t, seed, seed_dir
                    )
                    if result is None:
                        seed_skip += 1
                        global_skip += 1
                    else:
                        seed_done += 1
                        global_done += 1

        seed_time = time.time() - seed_start
        print(f"\nSeed {seed}: {seed_done} trained, {seed_skip} skipped, {seed_time/60:.1f} min")

    total_time = time.time() - global_start
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"Trained: {global_done}")
    print(f"Skipped: {global_skip}")
    print(f"Time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"Saved to: {save_dir}")


if __name__ == '__main__':
    main()
