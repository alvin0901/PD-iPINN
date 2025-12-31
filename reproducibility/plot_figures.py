#!/usr/bin/env python3
"""
Figure generation for PD-iPINN paper.

Usage:
    python plot_figures.py --data-dir ./results --fig-dir ./figures
    python plot_figures.py --main-only
    python plot_figures.py --supp-only
    python plot_figures.py --fig 2 3 5
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from pathlib import Path
from scipy.special import erf

# =============================================================================
# AIP Style Configuration
# =============================================================================

SINGLE_COL = 3.37
DOUBLE_COL = 6.69
MAX_HEIGHT = 8.25

FIG_SIZES = {
    '1x2': (DOUBLE_COL, 3.0),
    '1x3': (DOUBLE_COL, DOUBLE_COL * 0.32),
    '2x2': (DOUBLE_COL, DOUBLE_COL * 0.70),
    '2x3': (DOUBLE_COL, DOUBLE_COL * 0.55),
    '2x5': (DOUBLE_COL, DOUBLE_COL * 0.45),
    '3x3': (DOUBLE_COL, DOUBLE_COL * 0.95),
    '3x5': (DOUBLE_COL, DOUBLE_COL * 0.70),
}

COLORS = {
    'true': '#000000',
    'P': '#8B0000',
    'PD': '#00008B',
    'noisy': '#0072B2',
}

AIP_RC_PARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'],
    'font.size': 8,
    'mathtext.fontset': 'stix',
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'axes.linewidth': 0.6,
    'axes.labelpad': 2,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'legend.fontsize': 6,
    'legend.frameon': False,
    'legend.handlelength': 1.5,
    'legend.handletextpad': 0.4,
    'legend.labelspacing': 0.3,
    'lines.linewidth': 0.8,
    'lines.markersize': 3,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.pad_inches': 0.02,
}

# =============================================================================
# Global Configuration
# =============================================================================

DATA_ROOT = None
DATA_ROOT_SIN = None
DATA_ROOT_GAUSSIAN = None
DATA_ROOT_2D = None
DATA_ROOT_PARAM = None
FIG_DIR = None
SEEDS = [42, 43, 44, 45, 46]

D_1D = 0.01
k_1D = 0.5
D_2D = 0.01
k_2D = 0.5
EPS = 1.0
L = 1.0
T = 1.0
lam_1D = D_1D * np.pi**2 + k_1D
lam_2D = D_2D * 2 * np.pi**2 + k_2D

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

D_TRUE = 0.01
K_TRUE = 0.5
NOISE_LEVELS_PARAM = [i * 0.05 for i in range(21)]
INIT_MULTIPLIERS = [0.01, 0.02, 0.1, 0.2, 0.5, 1.0, 2, 5, 10, 50, 100]

SIGMA0 = 0.1
Q0 = 1.0

DISPLAY_EVERY = 100


def init_paths(data_dir, fig_dir):
    global DATA_ROOT, DATA_ROOT_SIN, DATA_ROOT_GAUSSIAN, DATA_ROOT_2D, DATA_ROOT_PARAM, FIG_DIR
    DATA_ROOT = Path(data_dir)
    DATA_ROOT_SIN = DATA_ROOT / 'sin'
    DATA_ROOT_GAUSSIAN = DATA_ROOT / 'gaussian'
    DATA_ROOT_2D = DATA_ROOT / '2d'
    DATA_ROOT_PARAM = DATA_ROOT / 'param'
    FIG_DIR = fig_dir
    os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================================
# Helper Functions
# =============================================================================

def add_panel_label(ax, label, loc='upper left', offset=(0.02, 0.98), color='black'):
    loc_params = {
        'upper left': {'x': offset[0], 'y': offset[1], 'ha': 'left', 'va': 'top'},
        'upper right': {'x': 1 - offset[0], 'y': offset[1], 'ha': 'right', 'va': 'top'},
        'lower left': {'x': offset[0], 'y': offset[0], 'ha': 'left', 'va': 'bottom'},
        'lower right': {'x': 1 - offset[0], 'y': offset[0], 'ha': 'right', 'va': 'bottom'},
    }
    p = loc_params.get(loc, loc_params['upper left'])
    ax.text(p['x'], p['y'], f'({label})', transform=ax.transAxes,
            fontsize=9, ha=p['ha'], va=p['va'], color=color)


def setup_axis(ax, nbins=None, square=True):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if nbins is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    if square:
        ax.set_box_aspect(1)


def setup_grid_axes(axes, nbins=4, show_ylabel_cols=None, square=True):
    if show_ylabel_cols is None:
        show_ylabel_cols = [0]
    nrows, ncols = axes.shape
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            setup_axis(ax, nbins=nbins, square=square)
            if j not in show_ylabel_cols:
                ax.set_yticklabels([])


def save_figure(fig, name, formats=['png', 'pdf']):
    os.makedirs(FIG_DIR, exist_ok=True)
    for fmt in formats:
        filepath = os.path.join(FIG_DIR, f'{name}.{fmt}')
        fig.savefig(filepath, format=fmt)
        print(f"  Saved: {filepath}")


def print_fig_size(fig, name=""):
    w, h = fig.get_size_inches()
    print(f"  [{name}] Size: {w:.3f} x {h:.3f} inches")


# =============================================================================
# Analytic Solutions
# =============================================================================

def phi_sin(x, t):
    return np.sin(np.pi * x) * np.exp(-lam_1D * t)


def rho_sin(x, t):
    return EPS * np.pi**2 * np.sin(np.pi * x) * np.exp(-lam_1D * t)


def sigma_t(t):
    return np.sqrt(SIGMA0**2 + 2 * D_1D * t)


def Q_t(t):
    return Q0 * np.exp(-k_1D * t)


def rho_gaussian(x, t):
    sig = sigma_t(t)
    return Q_t(t) / (np.sqrt(2 * np.pi) * sig) * np.exp(-x**2 / (2 * sig**2))


def phi_gaussian(x, t):
    sig = sigma_t(t)
    z = x / (np.sqrt(2) * sig)
    V_raw = -Q_t(t) / (2 * EPS) * (
        x * erf(z) + sig * np.sqrt(2 / np.pi) * np.exp(-z**2)
    )
    V_mean = np.mean(V_raw) if isinstance(x, np.ndarray) else 0
    return V_raw - V_mean


def phi_2d(x, y, t):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-lam_2D * t)


def rho_2d(x, y, t):
    return EPS * 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-lam_2D * t)


# =============================================================================
# Data Loading
# =============================================================================

def load_model_1d(data_root, model_type, noise, num_x, num_t, seed):
    filename = f"{model_type}_noise{int(noise*100):02d}_nx{num_x}_nt{num_t}.pkl"
    path = data_root / f'seed{seed}' / filename
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model_2d(model_type, noise, num_x, num_y, num_t, seed):
    filename = f"{model_type}_noise{int(noise*100):02d}_nx{num_x}_ny{num_y}_nt{num_t}.pkl"
    path = DATA_ROOT_2D / f'seed{seed}' / filename
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def multiplier_to_str(mult):
    if mult < 1:
        return f"{mult:.2f}".replace('.', 'p').rstrip('0').rstrip('p') or "0"
    else:
        return f"{int(mult)}" if mult == int(mult) else f"{mult}".replace('.', 'p')


def load_param_model(param_mode, noise, multiplier, num_x, num_t, seed):
    mult_str = multiplier_to_str(multiplier)
    filename = f"learn{param_mode}_mult{mult_str}_noise{int(noise*100):02d}_nx{num_x}_nt{num_t}.pkl"
    path = DATA_ROOT_PARAM / f'seed{seed}' / filename
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_metric_stats_1d(data_root, model_type, noise, num_x, num_t, metric, seeds=None):
    if seeds is None:
        seeds = SEEDS
    values = []
    for seed in seeds:
        r = load_model_1d(data_root, model_type, noise, num_x, num_t, seed)
        if r is not None and metric in r:
            values.append(r[metric])
    if len(values) == 0:
        return np.nan, np.nan
    return np.mean(values), np.std(values)


def get_metric_stats_2d(model_type, noise, num_x, num_y, num_t, metric, seeds=None):
    if seeds is None:
        seeds = SEEDS
    values = []
    for seed in seeds:
        r = load_model_2d(model_type, noise, num_x, num_y, num_t, seed)
        if r is not None and metric in r:
            values.append(r[metric])
    if len(values) == 0:
        return np.nan, np.nan
    return np.mean(values), np.std(values)


def get_param_error_stats(param_mode, noise, multiplier, num_x=51, num_t=101, seeds=None):
    if seeds is None:
        seeds = SEEDS
    values = []
    for seed in seeds:
        r = load_param_model(param_mode, noise, multiplier, num_x, num_t, seed)
        if r is not None and 'param_error' in r:
            values.append(r['param_error'])
    if len(values) == 0:
        return np.nan, np.nan
    return np.mean(values), np.std(values)


# =============================================================================
# Figure 2: Synthetic Datasets
# =============================================================================

def plot_fig2(noise_level=0.50, num_data_1d=51, num_data_2d=21, save=False):
    print("Generating Fig 2: Synthetic datasets")
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.95))

    x_fine = np.linspace(-L, L, 500)
    x_data = np.linspace(-L, L, num_data_1d)
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    cmap = plt.cm.viridis
    t_colors = [cmap(i / (len(t_values) - 1)) for i in range(len(t_values))]

    np.random.seed(44)
    ticks_1d = [-1, -0.5, 0, 0.5, 1]
    ylabel_x = -0.14

    # Row (a): 1D Sinusoidal
    ax = axes[0, 0]
    for t_val, color in zip(t_values, t_colors):
        ax.plot(x_fine, rho_sin(x_fine, t_val), '-', color=color, label=f'$t={t_val}$')
    ax.set_ylabel(r'$\rho$')
    ax.yaxis.set_label_coords(ylabel_x, 0.5)
    ax.set_title(r'$\rho_{\mathrm{ex}}$')
    ax.set_xlim(-L, L)
    ax.set_xticks(ticks_1d)
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.legend(loc='lower right', fontsize=5)
    setup_axis(ax)

    ax = axes[0, 1]
    for t_val, color in zip(t_values, t_colors):
        ax.plot(x_fine, phi_sin(x_fine, t_val), '-', color=color)
    ax.set_ylabel(r'$\phi$')
    ax.yaxis.set_label_coords(ylabel_x, 0.5)
    ax.set_title(r'$\phi_{\mathrm{ex}}$')
    ax.set_xlim(-L, L)
    ax.set_xticks(ticks_1d)
    phi_sin_ylim = ax.get_ylim()
    setup_axis(ax)

    ax = axes[0, 2]
    ax.plot(x_fine, phi_sin(x_fine, 0), '-', color=COLORS['true'], label='True')
    phi_data = phi_sin(x_data, 0)
    phi_scale = np.max(np.abs(phi_data))
    phi_noisy = phi_data + noise_level * phi_scale * np.random.randn(len(phi_data))
    ax.plot(x_data, phi_noisy, 'o', color=COLORS['noisy'], markersize=1.5, label='Noisy')
    ax.set_ylabel(r'$\phi$')
    ax.yaxis.set_label_coords(ylabel_x, 0.5)
    ax.set_title(rf'$\phi_{{\mathrm{{obs}}}}$ ($\alpha={int(noise_level*100)}\%$)')
    ax.set_xlim(-L, L)
    ax.set_xticks(ticks_1d)
    ax.set_ylim(phi_sin_ylim)
    ax.legend(loc='lower right', fontsize=5)
    setup_axis(ax)
    add_panel_label(axes[0, 0], 'a')

    # Row (b): 1D Gaussian
    ax = axes[1, 0]
    for t_val, color in zip(t_values, t_colors):
        ax.plot(x_fine, rho_gaussian(x_fine, t_val), '-', color=color)
    ax.set_ylabel(r'$\rho$')
    ax.yaxis.set_label_coords(ylabel_x, 0.5)
    ax.set_xlim(-L, L)
    ax.set_xticks(ticks_1d)
    setup_axis(ax)

    ax = axes[1, 1]
    for t_val, color in zip(t_values, t_colors):
        ax.plot(x_fine, phi_gaussian(x_fine, t_val), '-', color=color)
    ax.set_ylabel(r'$\phi$')
    ax.yaxis.set_label_coords(ylabel_x, 0.5)
    ax.set_xlim(-L, L)
    ax.set_xticks(ticks_1d)
    ax.set_ylim(-0.24, 0.24)
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    setup_axis(ax)

    ax = axes[1, 2]
    ax.plot(x_fine, phi_gaussian(x_fine, 0), '-', color=COLORS['true'], label='True')
    phi_data_g = phi_gaussian(x_data, 0)
    phi_scale_g = np.max(np.abs(phi_data_g))
    phi_noisy_g = phi_data_g + noise_level * phi_scale_g * np.random.randn(len(phi_data_g))
    ax.plot(x_data, phi_noisy_g, 'o', color=COLORS['noisy'], markersize=1.5)
    ax.set_ylabel(r'$\phi$')
    ax.yaxis.set_label_coords(ylabel_x, 0.5)
    ax.set_xlim(-L, L)
    ax.set_xticks(ticks_1d)
    ax.set_ylim(-0.24, 0.24)
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    setup_axis(ax)
    add_panel_label(axes[1, 0], 'b')

    # Row (c): 2D Sinusoidal
    x_2d = np.linspace(X_MIN, X_MAX, num_data_2d)
    y_2d = np.linspace(Y_MIN, Y_MAX, num_data_2d)
    X_2d, Y_2d = np.meshgrid(x_2d, y_2d)

    phi_ex_2d = phi_2d(X_2d, Y_2d, 0)
    rho_ex_2d = rho_2d(X_2d, Y_2d, 0)
    phi_scale_2d = np.max(np.abs(phi_ex_2d))
    phi_noisy_2d = phi_ex_2d + noise_level * phi_scale_2d * np.random.randn(*phi_ex_2d.shape)

    ticks_2d = [-1, -0.5, 0, 0.5, 1]

    ax = axes[2, 0]
    im0 = ax.imshow(rho_ex_2d, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower',
                    cmap='viridis', vmin=-20, vmax=20)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.yaxis.set_label_coords(ylabel_x, 0.5)
    ax.set_xticks(ticks_2d)
    ax.set_yticks(ticks_2d)
    setup_axis(ax)

    ax = axes[2, 1]
    im1 = ax.imshow(phi_ex_2d, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower', cmap='viridis')
    ax.set_xlabel(r'$x$')
    ax.set_xticks(ticks_2d)
    ax.set_yticks(ticks_2d)
    ax.set_yticklabels([])
    setup_axis(ax)

    ax = axes[2, 2]
    im2 = ax.imshow(phi_noisy_2d, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin='lower',
                    cmap='viridis', vmin=-3.0, vmax=3.0)
    ax.set_xlabel(r'$x$')
    ax.set_xticks(ticks_2d)
    ax.set_yticks(ticks_2d)
    ax.set_yticklabels([])
    setup_axis(ax)
    add_panel_label(axes[2, 0], 'c')

    fig.tight_layout()

    cbar_width = 0.012
    cbar_pad = 0.008
    for ax, im, label in [(axes[2, 0], im0, r'$\rho$'),
                          (axes[2, 1], im1, r'$\phi$'),
                          (axes[2, 2], im2, r'$\phi$')]:
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + cbar_pad, pos.y0, cbar_width, pos.height])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.set_title(label, fontsize=7, pad=3)
        cbar.ax.tick_params(labelsize=5)
        cbar.ax.yaxis.set_tick_params(pad=8)
        for tick_label in cbar.ax.get_yticklabels():
            tick_label.set_ha('center')

    if save:
        save_figure(fig, 'fig2_synthetic_datasets')
    print_fig_size(fig, "Fig 2")
    plt.close(fig)
    return fig


# =============================================================================
# Figure 3: Sin Time Evolution
# =============================================================================

def plot_fig3(noise=0.50, num_x=51, num_t=101, seed=44, save=False):
    print(f"Generating Fig 3: Sin evolution, noise={int(noise*100)}%")
    r_P = load_model_1d(DATA_ROOT_SIN, 'P', noise, num_x, num_t, seed)
    r_PD = load_model_1d(DATA_ROOT_SIN, 'PD', noise, num_x, num_t, seed)

    if r_P is None or r_PD is None:
        print("  [ERROR] Data not found")
        return None

    x_grid = r_P['x_grid']
    t_grid = r_P['t_grid']

    phi_gt = r_P['phi_ex'].reshape(num_x, num_t)
    rho_gt = r_P['rho_ex'].reshape(num_x, num_t)
    phi_P = r_P['phi_pred'].reshape(num_x, num_t)
    phi_PD = r_PD['phi_pred'].reshape(num_x, num_t)
    rho_P = r_P['rho_pred'].reshape(num_x, num_t)
    rho_PD = r_PD['rho_pred'].reshape(num_x, num_t)

    t_targets = [0, 0.25, 0.5, 0.75, 1.0]
    t_indices = [np.argmin(np.abs(t_grid - t)) for t in t_targets]

    phi_ylim = (-1.2, 1.2)
    rho_ylim = (-12, 12)

    fig, axes = plt.subplots(2, 5, figsize=FIG_SIZES['2x5'])
    ylabel_x = -0.22

    for i, (t_idx, t_val) in enumerate(zip(t_indices, t_targets)):
        ax = axes[0, i]
        ax.plot(x_grid, phi_gt[:, t_idx], '-', color=COLORS['true'], label='True')
        ax.plot(x_grid, phi_P[:, t_idx], '-', color=COLORS['P'], label='P-iPINN')
        ax.plot(x_grid, phi_PD[:, t_idx], '-', color=COLORS['PD'], label='PD-iPINN')
        ax.set_xlabel(r'$x$')
        if i == 0:
            ax.set_ylabel(r'$\phi$')
            ax.yaxis.set_label_coords(ylabel_x, 0.5)
            ax.legend(loc='lower right', fontsize=4)
        ax.set_title(f'$t = {t_val}$')
        ax.set_ylim(phi_ylim)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])

        ax = axes[1, i]
        ax.plot(x_grid, rho_gt[:, t_idx], '-', color=COLORS['true'])
        ax.plot(x_grid, rho_P[:, t_idx], '-', color=COLORS['P'])
        ax.plot(x_grid, rho_PD[:, t_idx], '-', color=COLORS['PD'])
        ax.set_xlabel(r'$x$')
        if i == 0:
            ax.set_ylabel(r'$\rho$')
            ax.yaxis.set_label_coords(ylabel_x, 0.5)
        ax.set_ylim(rho_ylim)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-10, 0, 10])

    setup_grid_axes(axes, nbins=None, show_ylabel_cols=[0])

    fig.tight_layout()
    fig.text(0.5, -0.02, rf'Noise level: $\alpha = {int(noise*100)}\%$',
             ha='center', va='top', fontsize=8)

    if save:
        save_figure(fig, f'fig3_sin_evolution_noise{int(noise*100):02d}')
    print_fig_size(fig, "Fig 3")
    plt.close(fig)
    return fig


# =============================================================================
# Figure 4: Sin L2 Error vs Noise
# =============================================================================

def plot_fig4(num_x=51, num_t=101, save=False):
    print("Generating Fig 4: Sin L2 error vs noise")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZES['1x2'])

    for idx, metric in enumerate(['phi_L2', 'rho_L2']):
        ax = axes[idx]

        for model_type in ['P', 'PD']:
            means, stds = [], []
            for noise in noise_levels:
                m, s = get_metric_stats_1d(DATA_ROOT_SIN, model_type, noise, num_x, num_t, metric)
                means.append(m)
                stds.append(s)

            means, stds = np.array(means), np.array(stds)
            noise_pct = np.array(noise_levels) * 100

            label = 'P-iPINN' if model_type == 'P' else 'PD-iPINN'
            color = COLORS[model_type]

            ax.plot(noise_pct, means, 'o-', color=color, label=label, markersize=3)
            ax.fill_between(noise_pct, means - stds, means + stds, color=color, alpha=0.2)

        ylabel = r'$\phi$ $L_2$ error' if metric == 'phi_L2' else r'$\rho$ $L_2$ error'
        ax.set_xlabel(r'Noise level $\alpha$ (%)')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 2.5)
        ax.legend(loc='upper left', bbox_to_anchor=(0.00, 0.95))
        setup_axis(ax)
        add_panel_label(ax, 'a' if idx == 0 else 'b')

    fig.tight_layout()
    if save:
        save_figure(fig, 'fig4_sin_l2_error')
    print_fig_size(fig, "Fig 4")
    plt.close(fig)
    return fig


# =============================================================================
# Figure 5: Sin Grid Robustness Heatmap
# =============================================================================

def plot_fig5(save=False):
    print("Generating Fig 5: Grid robustness heatmap")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    x_grids = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
    t_grids = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))

    cmap = plt.cm.inferno
    vmin, vmax = 0, 2.0

    for col, model_type in enumerate(['P', 'PD']):
        data_mean = np.zeros((len(x_grids), len(noise_levels)))
        for i, nx in enumerate(x_grids):
            for j, noise in enumerate(noise_levels):
                m, s = get_metric_stats_1d(DATA_ROOT_SIN, model_type, noise, nx, 101, 'rho_L2')
                data_mean[i, j] = m

        ax = axes[0, col]
        ax.set_box_aspect(1)
        im = ax.imshow(data_mean, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f'{int(n*100)}' for n in noise_levels])
        ax.set_yticks(range(len(x_grids)))
        ax.set_yticklabels(x_grids)
        ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel(r'$N_x$')
            ax.set_title('P-iPINN')
        else:
            ax.set_yticklabels([])
            ax.set_title('PD-iPINN')

        add_panel_label(ax, 'a' if col == 0 else 'b', color='white')

        data_mean = np.zeros((len(t_grids), len(noise_levels)))
        for i, nt in enumerate(t_grids):
            for j, noise in enumerate(noise_levels):
                m, s = get_metric_stats_1d(DATA_ROOT_SIN, model_type, noise, 51, nt, 'rho_L2')
                data_mean[i, j] = m

        ax = axes[1, col]
        ax.set_box_aspect(1)
        im = ax.imshow(data_mean, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f'{int(n*100)}' for n in noise_levels])
        ax.set_yticks(range(len(t_grids)))
        ax.set_yticklabels(t_grids)
        ax.set_xlabel(r'Noise $\alpha$ (%)')

        if col == 0:
            ax.set_ylabel(r'$N_t$')
        else:
            ax.set_yticklabels([])

        add_panel_label(ax, 'c' if col == 0 else 'd', color='white')

    axes[0, 1].text(1.05, 0.5, r'$N_x$ sweep', transform=axes[0, 1].transAxes,
                    rotation=270, va='center', ha='left', fontsize=8)
    axes[1, 1].text(1.05, 0.5, r'$N_t$ sweep', transform=axes[1, 1].transAxes,
                    rotation=270, va='center', ha='left', fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.85, 1])

    cbar_ax = fig.add_axes([0.90, 0.085, 0.025, 0.86])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$\rho$ $L_2$ error')

    if save:
        save_figure(fig, 'fig5_sin_grid_robustness')
    print_fig_size(fig, "Fig 5")
    plt.close(fig)
    return fig


# =============================================================================
# Figure 6: Gaussian Time Evolution
# =============================================================================

def plot_fig6(noise=0.50, num_x=51, num_t=101, seed=44, save=False):
    print(f"Generating Fig 6: Gaussian evolution, noise={int(noise*100)}%")
    r_P = load_model_1d(DATA_ROOT_GAUSSIAN, 'P', noise, num_x, num_t, seed)
    r_PD = load_model_1d(DATA_ROOT_GAUSSIAN, 'PD', noise, num_x, num_t, seed)

    if r_P is None or r_PD is None:
        print("  [ERROR] Data not found")
        return None

    x_grid = r_P['x_grid']
    t_grid = r_P['t_grid']
    x_fine = np.linspace(-1, 1, 500)

    phi_gt = r_P['phi_ex'].reshape(num_x, num_t)
    rho_gt = r_P['rho_ex'].reshape(num_x, num_t)
    phi_P = r_P['phi_pred'].reshape(num_x, num_t)
    phi_PD = r_PD['phi_pred'].reshape(num_x, num_t)
    rho_P = r_P['rho_pred'].reshape(num_x, num_t)
    rho_PD = r_PD['rho_pred'].reshape(num_x, num_t)

    t_targets = [0, 0.25, 0.5, 0.75, 1.0]
    t_indices = [np.argmin(np.abs(t_grid - t)) for t in t_targets]

    phi_ylim = (-0.3, 0.3)
    rho_ylim = (-0.2, 4.2)

    fig, axes = plt.subplots(2, 5, figsize=FIG_SIZES['2x5'])
    ylabel_x = -0.22

    for i, (t_idx, t_val) in enumerate(zip(t_indices, t_targets)):
        ax = axes[0, i]
        ax.plot(x_fine, phi_gaussian(x_fine, t_val), '-', color=COLORS['true'], label='True')
        ax.plot(x_grid, phi_P[:, t_idx], '-', color=COLORS['P'], label='P-iPINN')
        ax.plot(x_grid, phi_PD[:, t_idx], '-', color=COLORS['PD'], label='PD-iPINN')
        ax.set_xlabel(r'$x$')
        if i == 0:
            ax.set_ylabel(r'$\phi$')
            ax.yaxis.set_label_coords(ylabel_x, 0.5)
            ax.legend(loc='upper right', fontsize=4)
        ax.set_title(f'$t = {t_val}$')
        ax.set_ylim(phi_ylim)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-0.2, 0, 0.2])

        ax = axes[1, i]
        ax.plot(x_fine, rho_gaussian(x_fine, t_val), '-', color=COLORS['true'])
        ax.plot(x_grid, rho_P[:, t_idx], '-', color=COLORS['P'])
        ax.plot(x_grid, rho_PD[:, t_idx], '-', color=COLORS['PD'])
        ax.set_xlabel(r'$x$')
        if i == 0:
            ax.set_ylabel(r'$\rho$')
            ax.yaxis.set_label_coords(ylabel_x, 0.5)
        ax.set_ylim(rho_ylim)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([0, 2, 4])

    setup_grid_axes(axes, nbins=None, show_ylabel_cols=[0])

    fig.tight_layout()
    fig.text(0.5, -0.02, rf'Noise level: $\alpha = {int(noise*100)}\%$',
             ha='center', va='top', fontsize=8)

    if save:
        save_figure(fig, f'fig6_gaussian_evolution_noise{int(noise*100):02d}')
    print_fig_size(fig, "Fig 6")
    plt.close(fig)
    return fig


# =============================================================================
# Figure 7: Gaussian L2 Error vs Noise
# =============================================================================

def plot_fig7(num_x=51, num_t=101, save=False):
    print("Generating Fig 7: Gaussian L2 error vs noise")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZES['1x2'])

    for idx, metric in enumerate(['phi_L2', 'rho_L2']):
        ax = axes[idx]

        for model_type in ['P', 'PD']:
            means, stds = [], []
            for noise in noise_levels:
                m, s = get_metric_stats_1d(DATA_ROOT_GAUSSIAN, model_type, noise, num_x, num_t, metric)
                means.append(m)
                stds.append(s)

            means, stds = np.array(means), np.array(stds)
            noise_pct = np.array(noise_levels) * 100

            label = 'P-iPINN' if model_type == 'P' else 'PD-iPINN'
            color = COLORS[model_type]

            ax.plot(noise_pct, means, 'o-', color=color, label=label, markersize=3)
            ax.fill_between(noise_pct, means - stds, means + stds, color=color, alpha=0.2)

        ylabel = r'$\phi$ $L_2$ error' if metric == 'phi_L2' else r'$\rho$ $L_2$ error'
        ax.set_xlabel(r'Noise level $\alpha$ (%)')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 2.5)
        ax.legend(loc='upper left', bbox_to_anchor=(0.00, 0.95))
        setup_axis(ax)
        add_panel_label(ax, 'a' if idx == 0 else 'b')

    fig.tight_layout()
    if save:
        save_figure(fig, 'fig7_gaussian_l2_error')
    print_fig_size(fig, "Fig 7")
    plt.close(fig)
    return fig


# =============================================================================
# Figure 8: 2D Time Evolution
# =============================================================================

def plot_fig8(noise=0.50, num_x=21, num_y=21, num_t=51, seed=44, save=False):
    print(f"Generating Fig 8: 2D evolution, noise={int(noise*100)}%")
    r_P = load_model_2d('P', noise, num_x, num_y, num_t, seed)
    r_PD = load_model_2d('PD', noise, num_x, num_y, num_t, seed)

    if r_P is None or r_PD is None:
        print("  [ERROR] Data not found")
        return None

    t_grid = r_P['t_grid']
    t_targets = [0, 0.25, 0.5, 0.75, 1.0]
    t_indices = [np.argmin(np.abs(t_grid - t)) for t in t_targets]

    rho_gt = r_P['rho_ex'].reshape(num_x, num_y, num_t)
    rho_P = r_P['rho_pred'].reshape(num_x, num_y, num_t)
    rho_PD = r_PD['rho_pred'].reshape(num_x, num_y, num_t)

    fig, axes = plt.subplots(3, 5, figsize=FIG_SIZES['3x5'])

    vmin = rho_gt.min()
    vmax = rho_gt.max()

    for i, (t_idx, t_val) in enumerate(zip(t_indices, t_targets)):
        ax = axes[0, i]
        im = ax.imshow(rho_gt[:, :, t_idx], extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                       origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'$t = {t_val}$')
        if i == 0:
            ax.set_ylabel('Exact\n$y$')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(r'$x$')

        ax = axes[1, i]
        im = ax.imshow(rho_P[:, :, t_idx], extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                       origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_ylabel('P-iPINN\n$y$')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(r'$x$')

        ax = axes[2, i]
        im = ax.imshow(rho_PD[:, :, t_idx], extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                       origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_ylabel('PD-iPINN\n$y$')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(r'$x$')

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.131, 0.02, 0.73])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$\rho$')
    cbar.ax.yaxis.set_tick_params(pad=10)
    for label in cbar.ax.get_yticklabels():
        label.set_ha('center')

    fig.text(0.5, 0.05, rf'Noise level: $\alpha = {int(noise*100)}\%$',
             ha='center', va='top', fontsize=8)

    if save:
        save_figure(fig, f'fig8_2d_evolution_noise{int(noise*100):02d}')
    print_fig_size(fig, "Fig 8")
    plt.close(fig)
    return fig


# =============================================================================
# Figure 9: 2D L2 Error vs Noise
# =============================================================================

def plot_fig9(num_x=21, num_y=21, num_t=51, save=False):
    print("Generating Fig 9: 2D L2 error vs noise")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZES['1x2'])

    for idx, metric in enumerate(['phi_L2', 'rho_L2']):
        ax = axes[idx]

        for model_type in ['P', 'PD']:
            means, stds = [], []
            for noise in noise_levels:
                m, s = get_metric_stats_2d(model_type, noise, num_x, num_y, num_t, metric)
                means.append(m)
                stds.append(s)

            means, stds = np.array(means), np.array(stds)
            noise_pct = np.array(noise_levels) * 100

            label = 'P-iPINN' if model_type == 'P' else 'PD-iPINN'
            color = COLORS[model_type]

            ax.plot(noise_pct, means, 'o-', color=color, label=label, markersize=3)
            ax.fill_between(noise_pct, means - stds, means + stds, color=color, alpha=0.2)

        ylabel = r'$\phi$ $L_2$ error' if metric == 'phi_L2' else r'$\rho$ $L_2$ error'
        ax.set_xlabel(r'Noise level $\alpha$ (%)')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 2.5)
        ax.legend(loc='upper left', bbox_to_anchor=(0.00, 0.95))
        setup_axis(ax)
        add_panel_label(ax, 'a' if idx == 0 else 'b')

    fig.tight_layout()
    if save:
        save_figure(fig, 'fig9_2d_l2_error')
    print_fig_size(fig, "Fig 9")
    plt.close(fig)
    return fig


# =============================================================================
# Figure 10: Parameter Estimation Heatmaps
# =============================================================================

def plot_fig10(save=False):
    print("Generating Fig 10: Parameter estimation heatmaps")
    noise_levels = NOISE_LEVELS_PARAM
    multipliers = [0.1, 0.2, 0.5, 1.0, 2, 5, 10]
    mult_labels = ['1/10', '1/5', '1/2', '1', '2', '5', '10']

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.40))

    for col, param_mode in enumerate(['D', 'K']):
        ax = axes[col]

        data_mean = np.zeros((len(multipliers), len(noise_levels)))
        for i, mult in enumerate(multipliers):
            for j, noise in enumerate(noise_levels):
                m, _ = get_param_error_stats(param_mode, noise, mult)
                data_mean[i, j] = m

        im = ax.imshow(data_mean, aspect='auto', cmap='inferno',
                       vmin=0, vmax=1, origin='lower')

        noise_pct = [int(n * 100) for n in noise_levels]
        ax.set_xticks(range(0, len(noise_levels), 4))
        ax.set_xticklabels([f'{noise_pct[i]}' for i in range(0, len(noise_levels), 4)])
        ax.set_xlabel(r'Noise level $\alpha$ (%)')

        ax.set_yticks(range(len(multipliers)))
        ax.set_yticklabels(mult_labels)
        if col == 0:
            ax.set_ylabel('Initial guess multiplier')

        ax.set_xticks(np.arange(-0.5, len(noise_levels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(multipliers), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.5, alpha=0.5)

        param_name = 'D' if param_mode == 'D' else 'k'
        true_val = D_TRUE if param_mode == 'D' else K_TRUE
        ax.set_title(f'{param_name} estimation (true = {true_val})')

        add_panel_label(ax, 'a' if col == 0 else 'b', color='white')

    fig.tight_layout(rect=[0, 0, 0.88, 1])

    cbar_ax = fig.add_axes([0.90, 0.16, 0.025, 0.73])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Relative error')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    if save:
        save_figure(fig, 'fig10_param_heatmap')
    print_fig_size(fig, "Fig 10")
    plt.close(fig)
    return fig


# =============================================================================
# Figure 11: Total Charge Error Heatmap
# =============================================================================

def plot_fig11(num_x=51, num_t=101, seeds=None, save=False, print_stats=True):
    if seeds is None:
        seeds = SEEDS
    print("Generating Fig 11: Total charge error heatmap")

    noise_levels = np.round(np.linspace(0.0, 1.0, 11), 2)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))

    data_mean = {}
    data_std = {}

    for case, data_root in [('Sin', DATA_ROOT_SIN), ('Gaussian', DATA_ROOT_GAUSSIAN)]:
        for model_type in ['P', 'PD']:
            mats = []
            for seed in seeds:
                per_noise = []
                for alpha in noise_levels:
                    r = load_model_1d(data_root, model_type, alpha, num_x, num_t, seed)
                    if r is None:
                        per_noise.append(np.full(num_t, np.nan))
                        continue

                    x_grid = r['x_grid']
                    rho_pred = r['rho_pred'].reshape(num_x, num_t)
                    rho_ex = r['rho_ex'].reshape(num_x, num_t)
                    Q_pred = np.trapz(rho_pred, x_grid, axis=0)

                    if case == 'Sin':
                        Q_scale = np.trapz(np.abs(rho_ex[:, 0]), x_grid)
                        Q_scale = max(Q_scale, 1e-12)
                        err = np.abs(Q_pred) / Q_scale
                    else:
                        Q_true = np.trapz(rho_ex, x_grid, axis=0)
                        Q0_scale = max(np.abs(Q_true[0]), 1e-12)
                        err = np.abs(Q_pred - Q_true) / Q0_scale
                    per_noise.append(err)
                mats.append(np.array(per_noise))

            mats = np.array(mats)
            data_mean[(case, model_type)] = np.nanmean(mats, axis=0)
            data_std[(case, model_type)] = np.nanstd(mats, axis=0)

    if print_stats:
        print("\n" + "="*70)
        print("Total Charge Error Statistics (time-averaged)")
        print("="*70)
        for case in ['Sin', 'Gaussian']:
            print(f"\n[{case}]")
            print(f"{'Noise':<8} {'P-iPINN':<20} {'PD-iPINN':<20}")
            print("-"*50)
            for j, alpha in enumerate(noise_levels):
                p_mean = np.nanmean(data_mean[(case, 'P')][j, :])
                p_std = np.nanmean(data_std[(case, 'P')][j, :])
                pd_mean = np.nanmean(data_mean[(case, 'PD')][j, :])
                pd_std = np.nanmean(data_std[(case, 'PD')][j, :])
                print(f"{int(alpha*100):>3}%     {p_mean:.3f}±{p_std:.3f}          {pd_mean:.3f}±{pd_std:.3f}")
        print("="*70 + "\n")

    vmax = 0.5
    plot_config = [(0,0,'Sin','P','a'), (0,1,'Sin','PD','b'),
                   (1,0,'Gaussian','P','c'), (1,1,'Gaussian','PD','d')]

    for row, col, case, model_type, label in plot_config:
        ax = axes[row, col]
        mat = data_mean[(case, model_type)]

        im = ax.imshow(mat.T, aspect='auto', origin='lower',
                       extent=[0, 100, 0, 1],
                       vmin=0, vmax=vmax, cmap='inferno')

        ax.set_box_aspect(1)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax.set_xticklabels(['0', '', '20', '', '40', '', '60', '', '80', '', '100'])
        ax.set_yticks([0, 0.5, 1])

        if row == 1:
            ax.set_xlabel(r'Noise $\alpha$ (%)')
        else:
            ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel(r'Time $t$')
        else:
            ax.set_yticklabels([])

        if row == 0:
            ax.set_title('P-iPINN' if model_type == 'P' else 'PD-iPINN')

        add_panel_label(ax, label, color='white')

    axes[0, 1].text(1.05, 0.5, 'Sin', transform=axes[0, 1].transAxes,
                    rotation=270, va='center', ha='left', fontsize=8)
    axes[1, 1].text(1.05, 0.5, 'Gaussian', transform=axes[1, 1].transAxes,
                    rotation=270, va='center', ha='left', fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.85, 1])

    cbar_ax = fig.add_axes([0.90, 0.085, 0.025, 0.86])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'Normalized error in $Q$')
    cbar.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    if save:
        save_figure(fig, 'fig11_total_charge_heatmap')
    print_fig_size(fig, "Fig 11")
    plt.close(fig)
    return fig


# =============================================================================
# Figure S1: Gaussian Grid Comparison
# =============================================================================

def plot_fig_S1(save=False):
    print("Generating Fig S1: Gaussian L2 error - Grid comparison")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    GRID_CONFIGS = [(21, 21), (21, 51), (51, 21), (51, 51), (51, 101)]
    panel_labels = ['a', 'b', 'c', 'd', 'e']

    fig, axes = plt.subplots(5, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 1.4))

    for row, (num_x, num_t) in enumerate(GRID_CONFIGS):
        for col, metric in enumerate(['phi_L2', 'rho_L2']):
            ax = axes[row, col]

            for model_type in ['P', 'PD']:
                means, stds = [], []
                for noise in noise_levels:
                    m, s = get_metric_stats_1d(DATA_ROOT_GAUSSIAN, model_type, noise, num_x, num_t, metric)
                    means.append(m)
                    stds.append(s)

                means, stds = np.array(means), np.array(stds)
                noise_pct = np.array(noise_levels) * 100

                label = 'P-iPINN' if model_type == 'P' else 'PD-iPINN'
                color = COLORS[model_type]

                ax.plot(noise_pct, means, 'o-', color=color, label=label, markersize=3)
                ax.fill_between(noise_pct, means - stds, means + stds, color=color, alpha=0.2)

            if col == 0:
                ax.set_ylabel(r'$\phi$ $L_2$ error')
            else:
                ax.set_ylabel(r'$\rho$ $L_2$ error')

            if row == 4:
                ax.set_xlabel(r'Noise level $\alpha$ (%)')
            else:
                ax.set_xticklabels([])

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 2.5)

            if row == 0:
                ax.legend(loc='upper left', bbox_to_anchor=(0.00, 0.95), fontsize=6)

            setup_axis(ax)

            if col == 0:
                add_panel_label(ax, panel_labels[row])

        axes[row, 1].text(1.05, 0.5, f'$N_x={num_x}, N_t={num_t}$',
                          transform=axes[row, 1].transAxes,
                          rotation=270, va='center', ha='left', fontsize=8)

    axes[0, 0].set_title(r'Potential $\phi$')
    axes[0, 1].set_title(r'Charge density $\rho$')

    fig.tight_layout(rect=[0, 0, 0.92, 1])

    if save:
        save_figure(fig, 'fig_S1_gaussian_grid_comparison')
    print_fig_size(fig, "Fig S1")
    plt.close(fig)
    return fig


# =============================================================================
# Figure S2: 2D Grid Comparison
# =============================================================================

def plot_fig_S2(save=False):
    print("Generating Fig S2: 2D L2 error - Grid comparison")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    GRID_CONFIGS = [(21, 21, 21), (21, 21, 51), (51, 51, 101)]
    panel_labels = ['a', 'b', 'c']

    fig, axes = plt.subplots(3, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))

    for row, (num_x, num_y, num_t) in enumerate(GRID_CONFIGS):
        for col, metric in enumerate(['phi_L2', 'rho_L2']):
            ax = axes[row, col]

            for model_type in ['P', 'PD']:
                means, stds = [], []
                for noise in noise_levels:
                    m, s = get_metric_stats_2d(model_type, noise, num_x, num_y, num_t, metric)
                    means.append(m)
                    stds.append(s)

                means, stds = np.array(means), np.array(stds)
                noise_pct = np.array(noise_levels) * 100

                label = 'P-iPINN' if model_type == 'P' else 'PD-iPINN'
                color = COLORS[model_type]

                ax.plot(noise_pct, means, 'o-', color=color, label=label, markersize=3)
                ax.fill_between(noise_pct, means - stds, means + stds, color=color, alpha=0.2)

            if col == 0:
                ax.set_ylabel(r'$\phi$ $L_2$ error')
            else:
                ax.set_ylabel(r'$\rho$ $L_2$ error')

            if row == 2:
                ax.set_xlabel(r'Noise level $\alpha$ (%)')
            else:
                ax.set_xticklabels([])

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 2.5)

            if row == 0:
                ax.legend(loc='upper left', bbox_to_anchor=(0.00, 0.95), fontsize=6)

            setup_axis(ax)

            if col == 0:
                add_panel_label(ax, panel_labels[row])

        axes[row, 1].text(1.05, 0.5, f'$N_x=N_y={num_x}, N_t={num_t}$',
                          transform=axes[row, 1].transAxes,
                          rotation=270, va='center', ha='left', fontsize=8)

    axes[0, 0].set_title(r'Potential $\phi$')
    axes[0, 1].set_title(r'Charge density $\rho$')

    fig.tight_layout(rect=[0, 0, 0.88, 1])

    if save:
        save_figure(fig, 'fig_S2_2d_grid_comparison')
    print_fig_size(fig, "Fig S2")
    plt.close(fig)
    return fig


# =============================================================================
# Figure S3: Total Charge Error Heatmap
# =============================================================================

def plot_fig_S3(num_x=51, num_t=101, seeds=None, save=False, print_stats=True):
    if seeds is None:
        seeds = SEEDS
    print("Generating Fig S3: Total charge error heatmap")

    noise_levels = np.round(np.linspace(0.0, 1.0, 11), 2)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))

    data_mean = {}
    data_std = {}

    for case, data_root in [('Sin', DATA_ROOT_SIN), ('Gaussian', DATA_ROOT_GAUSSIAN)]:
        for model_type in ['P', 'PD']:
            mats = []
            for seed in seeds:
                per_noise = []
                for alpha in noise_levels:
                    r = load_model_1d(data_root, model_type, alpha, num_x, num_t, seed)
                    if r is None:
                        per_noise.append(np.full(num_t, np.nan))
                        continue

                    x_grid = r['x_grid']
                    rho_pred = r['rho_pred'].reshape(num_x, num_t)
                    rho_ex = r['rho_ex'].reshape(num_x, num_t)
                    Q_pred = np.trapz(rho_pred, x_grid, axis=0)

                    if case == 'Sin':
                        Q_scale = np.trapz(np.abs(rho_ex[:, 0]), x_grid)
                        Q_scale = max(Q_scale, 1e-12)
                        err = np.abs(Q_pred) / Q_scale
                    else:
                        Q_true = np.trapz(rho_ex, x_grid, axis=0)
                        Q0_scale = max(np.abs(Q_true[0]), 1e-12)
                        err = np.abs(Q_pred - Q_true) / Q0_scale
                    per_noise.append(err)
                mats.append(np.array(per_noise))

            mats = np.array(mats)
            data_mean[(case, model_type)] = np.nanmean(mats, axis=0)
            data_std[(case, model_type)] = np.nanstd(mats, axis=0)

    if print_stats:
        print("\n" + "="*70)
        print("Total Charge Error Statistics (time-averaged)")
        print("="*70)
        for case in ['Sin', 'Gaussian']:
            print(f"\n[{case}]")
            print(f"{'Noise':<8} {'P-iPINN':<20} {'PD-iPINN':<20}")
            print("-"*50)
            for j, alpha in enumerate(noise_levels):
                p_mean = np.nanmean(data_mean[(case, 'P')][j, :])
                p_std = np.nanmean(data_std[(case, 'P')][j, :])
                pd_mean = np.nanmean(data_mean[(case, 'PD')][j, :])
                pd_std = np.nanmean(data_std[(case, 'PD')][j, :])
                print(f"{int(alpha*100):>3}%     {p_mean:.3f}±{p_std:.3f}          {pd_mean:.3f}±{pd_std:.3f}")
        print("="*70 + "\n")

    vmax = 0.5
    plot_config = [(0,0,'Sin','P','a'), (0,1,'Sin','PD','b'),
                   (1,0,'Gaussian','P','c'), (1,1,'Gaussian','PD','d')]

    for row, col, case, model_type, label in plot_config:
        ax = axes[row, col]
        mat = data_mean[(case, model_type)]

        im = ax.imshow(mat.T, aspect='auto', origin='lower',
                       extent=[0, 100, 0, 1],
                       vmin=0, vmax=vmax, cmap='inferno')

        ax.set_box_aspect(1)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax.set_xticklabels(['0', '', '20', '', '40', '', '60', '', '80', '', '100'])
        ax.set_yticks([0, 0.5, 1])

        if row == 1:
            ax.set_xlabel(r'Noise $\alpha$ (%)')
        else:
            ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel(r'Time $t$')
        else:
            ax.set_yticklabels([])

        if row == 0:
            ax.set_title('P-iPINN' if model_type == 'P' else 'PD-iPINN')

        add_panel_label(ax, label, color='white')

    axes[0, 1].text(1.05, 0.5, 'Sin', transform=axes[0, 1].transAxes,
                    rotation=270, va='center', ha='left', fontsize=8)
    axes[1, 1].text(1.05, 0.5, 'Gaussian', transform=axes[1, 1].transAxes,
                    rotation=270, va='center', ha='left', fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.85, 1])

    cbar_ax = fig.add_axes([0.90, 0.085, 0.025, 0.86])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'Normalized error in $Q$')
    cbar.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    if save:
        save_figure(fig, 'fig_S3_total_charge_heatmap')
    print_fig_size(fig, "Fig S3")
    plt.close(fig)
    return fig


# =============================================================================
# Figure S4: Grid Robustness Heatmap with Values
# =============================================================================

def plot_fig_S4(save=False, show_values=True):
    print("Generating Fig S4: Grid robustness heatmap with uncertainty")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    x_grids = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
    t_grids = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))

    cmap = plt.cm.inferno
    vmin, vmax = 0, 2.0

    for col, model_type in enumerate(['P', 'PD']):
        data_mean = np.zeros((len(x_grids), len(noise_levels)))
        data_std = np.zeros((len(x_grids), len(noise_levels)))
        for i, nx in enumerate(x_grids):
            for j, noise in enumerate(noise_levels):
                m, s = get_metric_stats_1d(DATA_ROOT_SIN, model_type, noise, nx, 101, 'rho_L2')
                data_mean[i, j] = m
                data_std[i, j] = s

        ax = axes[0, col]
        ax.set_box_aspect(1)
        im = ax.imshow(data_mean, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        if show_values:
            for i in range(len(x_grids)):
                for j in range(len(noise_levels)):
                    val = data_mean[i, j]
                    std = data_std[i, j]
                    text_color = 'black' if val > (vmax - vmin) / 2 else 'white'
                    if np.isnan(val):
                        ax.text(j, i, 'N/A', ha='center', va='center', fontsize=4, color='gray')
                    else:
                        ax.text(j, i, f'{val:.2f}\n±{std:.2f}', ha='center', va='center',
                                fontsize=4, color=text_color)

        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f'{int(n*100)}' for n in noise_levels])
        ax.set_yticks(range(len(x_grids)))
        ax.set_yticklabels(x_grids)
        ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel(r'$N_x$')
            ax.set_title('P-iPINN')
        else:
            ax.set_yticklabels([])
            ax.set_title('PD-iPINN')

        add_panel_label(ax, 'a' if col == 0 else 'b', color='white')

        data_mean = np.zeros((len(t_grids), len(noise_levels)))
        data_std = np.zeros((len(t_grids), len(noise_levels)))
        for i, nt in enumerate(t_grids):
            for j, noise in enumerate(noise_levels):
                m, s = get_metric_stats_1d(DATA_ROOT_SIN, model_type, noise, 51, nt, 'rho_L2')
                data_mean[i, j] = m
                data_std[i, j] = s

        ax = axes[1, col]
        ax.set_box_aspect(1)
        im = ax.imshow(data_mean, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        if show_values:
            for i in range(len(t_grids)):
                for j in range(len(noise_levels)):
                    val = data_mean[i, j]
                    std = data_std[i, j]
                    text_color = 'black' if val > (vmax - vmin) / 2 else 'white'
                    if np.isnan(val):
                        ax.text(j, i, 'N/A', ha='center', va='center', fontsize=4, color='gray')
                    else:
                        ax.text(j, i, f'{val:.2f}\n±{std:.2f}', ha='center', va='center',
                                fontsize=4, color=text_color)

        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f'{int(n*100)}' for n in noise_levels])
        ax.set_yticks(range(len(t_grids)))
        ax.set_yticklabels(t_grids)
        ax.set_xlabel(r'Noise $\alpha$ (%)')

        if col == 0:
            ax.set_ylabel(r'$N_t$')
        else:
            ax.set_yticklabels([])

        add_panel_label(ax, 'c' if col == 0 else 'd', color='white')

    axes[0, 1].text(1.05, 0.5, r'$N_x$ sweep', transform=axes[0, 1].transAxes,
                    rotation=270, va='center', ha='left', fontsize=8)
    axes[1, 1].text(1.05, 0.5, r'$N_t$ sweep', transform=axes[1, 1].transAxes,
                    rotation=270, va='center', ha='left', fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.85, 1])

    cbar_ax = fig.add_axes([0.90, 0.085, 0.025, 0.86])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$\rho$ $L_2$ error')

    if save:
        save_figure(fig, 'fig_S4_grid_robustness_with_std')
    print_fig_size(fig, "Fig S4")
    plt.close(fig)
    return fig


# =============================================================================
# Figure S5: Parameter Estimation Heatmap with Values
# =============================================================================

def plot_fig_S5(save=False, show_values=True):
    print("Generating Fig S5: Parameter estimation heatmaps with uncertainty")
    noise_levels = NOISE_LEVELS_PARAM
    multipliers = [0.1, 0.2, 0.5, 1.0, 2, 5, 10]
    mult_labels = ['1/10', '1/5', '1/2', '1', '2', '5', '10']

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.45))
    vmin, vmax = 0, 1.0
    cmap = plt.cm.inferno

    for col, param_mode in enumerate(['D', 'K']):
        ax = axes[col]

        data_mean = np.zeros((len(multipliers), len(noise_levels)))
        data_std = np.zeros((len(multipliers), len(noise_levels)))

        for i, mult in enumerate(multipliers):
            for j, noise in enumerate(noise_levels):
                m, s = get_param_error_stats(param_mode, noise, mult)
                data_mean[i, j] = m
                data_std[i, j] = s

        im = ax.imshow(data_mean, aspect='auto', cmap=cmap,
                       vmin=vmin, vmax=vmax, origin='lower')

        if show_values:
            for i in range(len(multipliers)):
                for j in range(len(noise_levels)):
                    val = data_mean[i, j]
                    std = data_std[i, j]
                    text_color = 'black' if val > (vmax - vmin) / 2 else 'white'
                    if np.isnan(val):
                        ax.text(j, i, 'N/A', ha='center', va='center', fontsize=2, color='gray')
                    else:
                        ax.text(j, i, f'{val:.2f}\n±{std:.2f}', ha='center', va='center',
                                fontsize=3, color=text_color)

        noise_pct = [int(n * 100) for n in noise_levels]
        ax.set_xticks(range(0, len(noise_levels), 4))
        ax.set_xticklabels([f'{noise_pct[i]}' for i in range(0, len(noise_levels), 4)])
        ax.set_xlabel(r'Noise level $\alpha$ (%)')

        ax.set_yticks(range(len(multipliers)))
        ax.set_yticklabels(mult_labels)
        if col == 0:
            ax.set_ylabel('Initial guess multiplier')

        ax.set_xticks(np.arange(-0.5, len(noise_levels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(multipliers), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.3, alpha=0.3)

        param_name = 'D' if param_mode == 'D' else 'k'
        true_val = D_TRUE if param_mode == 'D' else K_TRUE
        ax.set_title(f'{param_name} estimation (true = {true_val})')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Relative error')

        add_panel_label(ax, 'a' if col == 0 else 'b', color='white')

    fig.tight_layout()

    if save:
        save_figure(fig, 'fig_S5_param_heatmap_with_std')
    print_fig_size(fig, "Fig S5")
    plt.close(fig)
    return fig


# =============================================================================
# Figure S6: Loss Convergence
# =============================================================================

def plot_fig_S6(seed=44, save=False):
    print(f"Generating Fig S6: Loss convergence, seed={seed}")
    noise_levels = [0.0, 0.5, 1.0]

    case_configs = [
        ('Sin', DATA_ROOT_SIN, 51, 101, None),
        ('Gaussian', DATA_ROOT_GAUSSIAN, 51, 101, None),
        ('2D Sin', None, 21, 51, 21),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))
    panel_labels = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]

    for row, (case_name, data_root, nx, nt, ny) in enumerate(case_configs):
        for col, noise in enumerate(noise_levels):
            ax = axes[row, col]
            all_losses = []

            for model_type in ['P', 'PD']:
                if ny is None:
                    r = load_model_1d(data_root, model_type, noise, nx, nt, seed)
                else:
                    r = load_model_2d(model_type, noise, nx, ny, nt, seed)

                if r is None:
                    continue

                loss_history = r.get('loss_history', None)
                if loss_history is None or len(loss_history) == 0:
                    continue

                total_loss = np.sum(loss_history, axis=1)
                epochs = np.arange(0, len(total_loss) * DISPLAY_EVERY, DISPLAY_EVERY)
                all_losses.extend(total_loss)

                label = f'{model_type}-iPINN' if row == 0 and col == 0 else None
                ax.semilogy(epochs, total_loss, color=COLORS[model_type], linewidth=0.8, label=label)

            ax.set_xlim(0, 5000)
            if all_losses:
                ax.set_ylim(min(all_losses) * 0.5, max(all_losses) * 2)

            if row == 2:
                ax.set_xlabel('Epoch')
            if col == 0:
                ax.set_ylabel('Total loss')
            if row == 0:
                ax.set_title(f'$\\alpha$ = {int(noise*100)}%', fontsize=8)
            if col == 2:
                ax.text(1.02, 0.5, case_name, transform=ax.transAxes,
                       fontsize=8, va='center', ha='left', rotation=-90)

            add_panel_label(ax, panel_labels[row][col], loc='upper right')
            ax.xaxis.set_minor_locator(AutoMinorLocator())

    axes[0, 0].legend(loc='upper right', bbox_to_anchor=(1.0, 0.94), fontsize=6)
    plt.tight_layout()
    plt.subplots_adjust(right=0.92)

    if save:
        save_figure(fig, 'fig_S6_loss_convergence')
    print_fig_size(fig, "Fig S6")
    plt.close(fig)
    return fig


# =============================================================================
# Figure S7: Parameter Convergence
# =============================================================================

def plot_fig_S7(seed=44, save=False):
    print(f"Generating Fig S7: Parameter convergence, seed={seed}")
    noise_levels = [0.0, 0.5, 1.0]
    multipliers = [0.2, 0.1, 5, 10]
    mult_labels = ['×0.2', '×0.1', '×5', '×10']
    param_configs = [('D', D_TRUE, '$D$'), ('K', K_TRUE, '$k$')]

    fig, axes = plt.subplots(6, 4, figsize=(DOUBLE_COL, DOUBLE_COL * 1.5))
    panel_labels = [chr(ord('a') + i) for i in range(24)]

    for param_idx, (param_mode, true_val, param_label) in enumerate(param_configs):
        for noise_idx, noise in enumerate(noise_levels):
            row = param_idx * 3 + noise_idx

            for col, multiplier in enumerate(multipliers):
                ax = axes[row, col]
                panel_label = panel_labels[row * 4 + col]

                r = load_param_model(param_mode, noise, multiplier, 51, 101, seed)

                if r is None:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                           ha='center', va='center', fontsize=6)
                    add_panel_label(ax, panel_label, loc='upper right')
                    continue

                param_history = r.get('param_history', [])
                if len(param_history) == 0:
                    ax.text(0.5, 0.5, 'No history', transform=ax.transAxes,
                           ha='center', va='center', fontsize=6)
                    add_panel_label(ax, panel_label, loc='upper right')
                    continue

                epochs, values = zip(*param_history)
                epochs = np.array(epochs)
                values = np.array(values)

                ax.plot(epochs, values, color=COLORS['PD'], linewidth=0.8)
                ax.axhline(y=true_val, color='black', linestyle='--', linewidth=0.6)
                ax.scatter([epochs[0]], [values[0]], color=COLORS['PD'], s=10, zorder=5,
                          marker='o', facecolors='none', linewidths=0.6)
                ax.scatter([epochs[-1]], [values[-1]], color=COLORS['PD'], s=10, zorder=5, marker='o')

                ax.set_xlim(0, 5000)
                init_val = true_val * multiplier
                y_min = min(true_val, init_val, min(values)) * 0.8
                y_max = max(true_val, init_val, max(values)) * 1.2
                ax.set_ylim(y_min, y_max)

                if noise_idx == 2:
                    ax.set_xlabel('Epoch', fontsize=7)
                else:
                    ax.set_xticklabels([])

                if col == 0:
                    ax.set_ylabel(f'{param_label}', fontsize=7)
                if row == 0:
                    ax.set_title(mult_labels[col], fontsize=8)
                if col == 3:
                    ax.text(1.02, 0.5, f'$\\alpha$={int(noise*100)}%',
                           transform=ax.transAxes, fontsize=7, va='center', ha='left', rotation=-90)

                add_panel_label(ax, panel_label, loc='upper right')
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(axis='both', which='major', labelsize=6)

    fig.text(0.99, 0.75, '$D$ estimation', rotation=-90, va='center', ha='left', fontsize=9)
    fig.text(0.99, 0.30, '$k$ estimation', rotation=-90, va='center', ha='left', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(right=0.90, hspace=0.3, wspace=0.3)

    if save:
        save_figure(fig, 'fig_S7_param_convergence')
    print_fig_size(fig, "Fig S7")
    plt.close(fig)
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate figures for PD-iPINN paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_figures.py --data-dir ./results --fig-dir ./figures
  python plot_figures.py --main-only
  python plot_figures.py --supp-only
  python plot_figures.py --fig 2 3 5
        """
    )
    parser.add_argument('--data-dir', type=str, default='./results',
                        help='Directory containing training results')
    parser.add_argument('--fig-dir', type=str, default='./figures',
                        help='Directory to save figures')
    parser.add_argument('--main-only', action='store_true',
                        help='Generate main text figures only')
    parser.add_argument('--supp-only', action='store_true',
                        help='Generate supplementary figures only')
    parser.add_argument('--fig', type=int, nargs='+',
                        help='Generate specific figures (e.g., --fig 2 3 5)')
    parser.add_argument('--seed', type=int, default=44,
                        help='Seed for single-seed figures')
    parser.add_argument('--no-save', action='store_true',
                        help='Display figures without saving')

    args = parser.parse_args()

    mpl.rcParams.update(AIP_RC_PARAMS)
    init_paths(args.data_dir, args.fig_dir)

    save = not args.no_save

    print("=" * 60)
    print("PD-iPINN Figure Generation")
    print("=" * 60)
    print(f"Data directory: {DATA_ROOT}")
    print(f"Figure directory: {FIG_DIR}")
    print(f"Save figures: {save}")
    print("=" * 60)

    main_figs = {
        2: lambda: plot_fig2(save=save),
        3: lambda: plot_fig3(seed=args.seed, save=save),
        4: lambda: plot_fig4(save=save),
        5: lambda: plot_fig5(save=save),  # Fig 11 style (larger boxes)
        6: lambda: plot_fig6(seed=args.seed, save=save),
        7: lambda: plot_fig7(save=save),
        8: lambda: plot_fig8(seed=args.seed, save=save),
        9: lambda: plot_fig9(save=save),
        10: lambda: plot_fig10(save=save),
        11: lambda: plot_fig11(save=save),
    }

    supp_figs = {
        'S1': lambda: plot_fig_S1(save=save),
        'S2': lambda: plot_fig_S2(save=save),
        'S3': lambda: plot_fig_S3(save=save),
        'S4': lambda: plot_fig_S4(save=save, show_values=True),
        'S5': lambda: plot_fig_S5(save=save, show_values=True),
        'S6': lambda: plot_fig_S6(seed=args.seed, save=save),
        'S7': lambda: plot_fig_S7(seed=args.seed, save=save),
    }

    if args.fig:
        print(f"\nGenerating selected figures: {args.fig}")
        for fig_num in args.fig:
            if fig_num in main_figs:
                main_figs[fig_num]()
            else:
                print(f"  [WARNING] Unknown figure: {fig_num}")
    elif args.main_only:
        print("\nGenerating main text figures...")
        for fig_num in sorted(main_figs.keys()):
            main_figs[fig_num]()
    elif args.supp_only:
        print("\nGenerating supplementary figures...")
        for fig_id in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']:
            supp_figs[fig_id]()
    else:
        print("\nGenerating all figures...")
        print("\n--- Main Text Figures ---")
        for fig_num in sorted(main_figs.keys()):
            main_figs[fig_num]()
        print("\n--- Supplementary Figures ---")
        for fig_id in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']:
            supp_figs[fig_id]()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
