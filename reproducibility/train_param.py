#!/usr/bin/env python3
"""
Multi-seed training script for parameter estimation with varying initial guesses.

Tests robustness to initial parameter guesses from 0.1× to 10× true value.

Usage:
    python train_param.py                     # Run all
    python train_param.py --seeds 42 43       # Run specific seeds
    python train_param.py --dry-run           # Show configuration only
"""

import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'

import argparse
import time
import pickle
import numpy as np
import deepxde as dde
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

SEEDS = [42, 43, 44, 45, 46]  # 5 seeds

# Noise levels: 0-100% in 5% steps (21 levels)
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]  # 21 levels

GRID_CONFIGS = [(51, 101)]

# Which parameter to learn: 'D' or 'K'
PARAM_MODES = ['D', 'K']

# Initial guess multipliers (relative to true value)
# Symmetric: underestimation (0.01-0.5) and overestimation (2-100)
# 0.1=1/10, 0.2=1/5, 0.5=1/2, 1=exact, 2, 5, 10
INIT_MULTIPLIERS = [0.1, 0.2, 0.5, 1.0, 2, 5, 10] 

# Training settings
EPOCHS = 5000
DISPLAY_EVERY = 1000
NUM_DOMAIN = 2000

# Physics parameters (Ground Truth)
EPS = 1.0
D_TRUE = 0.01
K_TRUE = 0.5
DECAY_RATE = D_TRUE * (np.pi ** 2) + K_TRUE

# Domain
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

# Output directory
SAVE_DIR = './results/param'


# =============================================================================
# Helper Functions
# =============================================================================

def phi_ex_func(X):
    """Exact potential."""
    x, t = X[:, 0:1], X[:, 1:2]
    return np.sin(np.pi * x) * np.exp(-DECAY_RATE * t)


def rho_ex_func(X):
    """Exact charge density."""
    x, t = X[:, 0:1], X[:, 1:2]
    return EPS * (np.pi ** 2) * np.sin(np.pi * x) * np.exp(-DECAY_RATE * t)


def add_noise(phi_clean, noise_level, seed):
    """Add relative Gaussian noise."""
    np.random.seed(seed)
    if noise_level == 0:
        return phi_clean.copy()
    phi_scale = np.max(np.abs(phi_clean))
    return phi_clean + noise_level * phi_scale * np.random.randn(*phi_clean.shape)


def multiplier_to_str(mult):
    """Convert multiplier to filename-safe string."""
    if mult < 1:
        # 0.01 -> "0p01", 0.1 -> "0p1", 0.5 -> "0p5"
        return f"{mult:.2f}".replace('.', 'p').rstrip('0').rstrip('p') or "0"
    else:
        # 1 -> "1", 10 -> "10", 100 -> "100"
        return f"{int(mult)}" if mult == int(mult) else f"{mult}".replace('.', 'p')


def get_model_name(param_mode, noise, multiplier, num_x, num_t):
    """Generate model filename."""
    mult_str = multiplier_to_str(multiplier)
    return f"learn{param_mode}_mult{mult_str}_noise{int(noise*100):02d}_nx{num_x}_nt{num_t}"


def relative_L2_error(pred, exact):
    """Compute relative L2 error."""
    return np.sqrt(np.mean((pred - exact)**2)) / np.sqrt(np.mean(exact**2))


# =============================================================================
# Training Function
# =============================================================================

def train_single_model(param_mode, noise_level, multiplier, num_x, num_t, seed, seed_dir):
    """Train a single model with parameter estimation."""
    model_name = get_model_name(param_mode, noise_level, multiplier, num_x, num_t)
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

    phi_ex = phi_ex_func(XT_data)
    rho_ex = rho_ex_func(XT_data)
    phi_obs = add_noise(phi_ex, noise_level, seed)

    # Compute initial guesses based on multiplier
    D_init = D_TRUE * multiplier
    K_init = K_TRUE * multiplier

    # Create Variables based on param_mode
    if param_mode == 'D':
        D_var = dde.Variable(D_init)
        k_var = K_TRUE
    else:  # 'K'
        D_var = D_TRUE
        k_var = dde.Variable(K_init)

    # PDE Definition
    def pde_coupled(X, y):
        phi_xx = dde.grad.hessian(y, X, component=0, i=0, j=0)
        rho = y[:, 1:2]
        rho_t = dde.grad.jacobian(y, X, i=1, j=1)
        rho_xx = dde.grad.hessian(y, X, component=1, i=0, j=0)
        res_poisson = -phi_xx - rho / EPS
        res_diffusion = rho_t - D_var * rho_xx + k_var * rho
        return [res_poisson, res_diffusion]

    # Geometry
    geom = dde.geometry.Interval(X_MIN, X_MAX)
    timedomain = dde.geometry.TimeDomain(T_MIN, T_MAX)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Observation
    observe_phi = dde.icbc.PointSetBC(XT_data, phi_obs, component=0)

    # Data
    data = dde.data.TimePDE(
        geomtime, pde_coupled, [observe_phi],
        num_domain=NUM_DOMAIN, num_boundary=0, num_initial=0,
        anchors=XT_data, num_test=5000
    )

    # Network
    net = dde.nn.PFNN(
        [2, [64, 64], [64, 64], [64, 64], [64, 64], 2],
        "tanh", "Glorot uniform"
    )

    model = dde.Model(data, net)

    # Setup callback
    if param_mode == 'D':
        external_vars = [D_var]
    else:
        external_vars = [k_var]

    variable_callback = dde.callbacks.VariableValue(
        external_vars,
        period=DISPLAY_EVERY,
        filename="variables.dat"
    )

    # Compile and train
    model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1000],
                  external_trainable_variables=external_vars)

    start_time = time.time()
    losshistory, _ = model.train(
        epochs=EPOCHS,
        display_every=DISPLAY_EVERY,
        callbacks=[variable_callback]
    )
    train_time = time.time() - start_time

    # Get final parameter values
    if param_mode == 'D':
        D_est = float(model.sess.run(D_var))
        k_est = K_TRUE
    else:
        D_est = D_TRUE
        k_est = float(model.sess.run(k_var))

    # Parse parameter history
    param_history = []
    if os.path.exists("variables.dat"):
        with open("variables.dat", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    epoch = float(parts[0])
                    val = float(parts[1].strip('[]'))
                    param_history.append((epoch, val))

    # Predict
    output = model.predict(XT_data)
    phi_pred = output[:, 0:1]
    rho_pred = output[:, 1:2]

    # Metrics
    phi_L2 = relative_L2_error(phi_pred, phi_ex)
    rho_L2 = relative_L2_error(rho_pred, rho_ex)

    if param_mode == 'D':
        param_error = abs(D_est - D_TRUE) / D_TRUE
    else:
        param_error = abs(k_est - K_TRUE) / K_TRUE

    # Save
    result = {
        'model_name': model_name,
        'param_mode': param_mode,
        'noise_level': noise_level,
        'multiplier': multiplier,
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
        'D_true': D_TRUE,
        'K_true': K_TRUE,
        'D_init': D_init if param_mode == 'D' else None,
        'K_init': K_init if param_mode == 'K' else None,
        'D_est': D_est,
        'k_est': k_est,
        'param_error': param_error,
        'param_history': param_history,
        'loss_history': np.array(losshistory.loss_train),
    }

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)

    # Cleanup
    if os.path.exists("variables.dat"):
        os.remove("variables.dat")
    tf.keras.backend.clear_session()

    print(f"  [DONE] {model_name}: err={param_error:.4f}, rho_L2={rho_L2:.4f}, time={train_time:.1f}s")
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Parameter estimation with varying initial guesses')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help='Seeds to run (default: 42-46)')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR,
                        help='Output directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show configuration only')
    args = parser.parse_args()

    seeds = args.seeds
    save_dir = args.save_dir

    total_models = len(seeds) * len(NOISE_LEVELS) * len(GRID_CONFIGS) * len(PARAM_MODES) * len(INIT_MULTIPLIERS)

    print("=" * 70)
    print("PD-iPINN Parameter Estimation - Initial Guess Sensitivity Study")
    print("=" * 70)
    print(f"Seeds: {seeds} ({len(seeds)} seeds)")
    print(f"Noise levels: {[f'{int(n*100)}%' for n in NOISE_LEVELS]} ({len(NOISE_LEVELS)} levels)")
    print(f"Grid configs: {GRID_CONFIGS}")
    print(f"Param modes: {PARAM_MODES}")
    print(f"Multipliers: {INIT_MULTIPLIERS} ({len(INIT_MULTIPLIERS)} levels)")
    print()
    print("Initial guess ranges:")
    print(f"  D: {D_TRUE} × {INIT_MULTIPLIERS}")
    print(f"     = {[D_TRUE * m for m in INIT_MULTIPLIERS]}")
    print(f"  k: {K_TRUE} × {INIT_MULTIPLIERS}")
    print(f"     = {[K_TRUE * m for m in INIT_MULTIPLIERS]}")
    print()
    print(f"Total models: {total_models}")
    print(f"  = {len(seeds)} seeds × {len(NOISE_LEVELS)} noise × {len(PARAM_MODES)} params × {len(INIT_MULTIPLIERS)} multipliers")
    print(f"Output: {save_dir}")
    print("=" * 70)

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

        print(f"\n{'='*70}")
        print(f"SEED {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'='*70}")

        seed_start = time.time()
        seed_done = 0
        seed_skip = 0

        for noise in NOISE_LEVELS:
            print(f"\n--- Noise: {int(noise*100)}% ---")

            for (num_x, num_t) in GRID_CONFIGS:
                for param_mode in PARAM_MODES:
                    for multiplier in INIT_MULTIPLIERS:
                        result = train_single_model(
                            param_mode, noise, multiplier, num_x, num_t, seed, seed_dir
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
    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)
    print(f"Trained: {global_done}")
    print(f"Skipped: {global_skip}")
    print(f"Time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"Saved to: {save_dir}")


if __name__ == '__main__':
    main()
