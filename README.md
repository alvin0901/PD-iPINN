# PD-iPINN

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18108529.svg)](https://doi.org/10.5281/zenodo.18108529)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Diffusion-Regularized Physics-Informed Neural Networks for Poisson Inverse Source Reconstruction of Charge Dynamics from Noisy Potential Measurements**

This repository contains the source code for reproducing the results in:

> J. Kim, D. Yang, and M. Lee, "Diffusion-Regularized PINNs for Poisson Inverse Source Reconstruction of Charge Dynamics from Noisy Potential Measurements," (2025). *Under review.*

## Overview

PD-iPINN couples the quasi-static Poisson equation with diffusion-decay dynamics to reconstruct time-resolved charge density from noisy potential measurements. The method does not require prior information on initial or boundary conditions (IC/BC), which are often unavailable in real experiments such as Kelvin probe force microscopy (KPFM).

**Key results:**
- 2–5× lower reconstruction error compared to Poisson-only approaches (P-iPINN)
- Robust performance across noise levels up to 100%
- No IC/BC information required
- Simultaneous parameter estimation (diffusion coefficient *D*, decay constant *k*)

## Repository Structure

```
PD-iPINN/
├── README.md
├── requirements.txt
├── LICENSE
├── demos/                             # Interactive notebooks (Colab-ready)
│   ├── demo_sin.ipynb                 # 1D sinusoidal case
│   ├── demo_gaussian.ipynb            # 1D Gaussian case
│   ├── demo_2d.ipynb                  # 2D sinusoidal case
│   └── demo_parameter_estimation.ipynb
├── reproducibility/                   # Full reproduction scripts
│   ├── train_sin.py                   # 1D sinusoidal (Fig. 3-5)
│   ├── train_gaussian.py              # 1D Gaussian (Fig. 6-7)
│   ├── train_2d.py                    # 2D sinusoidal (Fig. 8-9)
│   ├── train_param.py                 # Parameter estimation (Fig. 10)
│   └── plot_figures.py                # Generate all figures
└── results/                           # Training outputs (see Data Availability)
```

## Quick Start

### Option 1: Google Colab (Recommended for Demo)

Run any notebook directly in Colab without local installation:

| Demo | Description | Link |
|------|-------------|------|
| 1D Sinusoidal | Basic reconstruction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvin0901/PD-iPINN/blob/main/demos/demo_sin.ipynb) |
| 1D Gaussian | Localized charge | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvin0901/PD-iPINN/blob/main/demos/demo_gaussian.ipynb) |
| 2D Sinusoidal | 2D extension | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvin0901/PD-iPINN/blob/main/demos/demo_2d.ipynb) |
| Parameter Estimation | Learn D or k | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvin0901/PD-iPINN/blob/main/demos/demo_parameter_estimation.ipynb) |

### Option 2: Local Installation

```bash
git clone https://github.com/alvin0901/PD-iPINN.git
cd PD-iPINN
pip install -r requirements.txt
```

## Data Availability

Pre-trained results and figure data are archived on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18108529.svg)](https://doi.org/10.5281/zenodo.18108529)

The `results/` folder in this repository is empty. You can either:

**Option A: Download pre-computed results** (recommended for figure reproduction)
```bash
# Download from Zenodo and extract
wget https://zenodo.org/record/18108529/files/results.zip
unzip results.zip -d results/

# Generate figures
python reproducibility/plot_figures.py --data-dir ./results --fig-dir ./figures
```

**Option B: Run training from scratch** (requires GPU, ~50 hours total)
```bash
cd reproducibility
python train_sin.py
python train_gaussian.py
python train_2d.py
python train_param.py
```

## Reproducing Paper Results

All experiments use 5 random seeds (42–46) for statistical analysis.

```bash
cd reproducibility

# 1D Sinusoidal case (Fig. 3-5, S4)
python train_sin.py

# 1D Gaussian case (Fig. 6-7, S1, S3)
python train_gaussian.py

# 2D Sinusoidal case (Fig. 8-9, S2)
python train_2d.py

# Parameter estimation (Fig. 10, S5, S7)
python train_param.py

# Generate figures
python plot_figures.py --data-dir ./results --fig-dir ./figures
```

### Command-line Options

```bash
python train_sin.py --seeds 42 43        # Run specific seeds
python train_sin.py --dry-run            # Show configuration only
python train_sin.py --save-dir ./output  # Custom output directory

python plot_figures.py --main-only       # Main text figures only
python plot_figures.py --supp-only       # Supplementary figures only
python plot_figures.py --fig 3 4 5       # Specific figures
```

### Computational Requirements

| Experiment | Models | Est. Time (A100) |
|------------|--------|------------------|
| 1D Sinusoidal | 2,090 | ~12 hours |
| 1D Gaussian | 550 | ~3 hours |
| 2D Sinusoidal | 330 | ~30 hours |
| Parameter Est. | 1,470 | ~8 hours |

## Method

PD-iPINN solves the coupled inverse problem:

**Poisson equation (quasi-static):**

$$\nabla^2 \phi = -\frac{\rho}{\varepsilon}$$

**Diffusion-decay equation:**

$$\frac{\partial \rho}{\partial t} = D \nabla^2 \rho - k \rho$$

**Loss function:**

$$\mathcal{L}_{\text{PD-iPINN}} = \lambda_{\text{data}} \mathcal{L}_{\text{data}} + \lambda_{\text{poisson}} \mathcal{L}_{\text{poisson}} + \lambda_{\text{diffusion}} \mathcal{L}_{\text{diffusion}}$$

with weights $[\lambda_{\text{poisson}}, \lambda_{\text{diffusion}}, \lambda_{\text{data}}] = [1, 1, 1000]$.

### Network Architecture

- **Architecture:** Parallel Fourier Neural Network (PFNN)
- **Hidden layers:** 4 layers × [64, 64] neurons (2 branches)
- **Activation:** tanh
- **Optimizer:** Adam (lr = 10⁻³)
- **Epochs:** 5,000

## Requirements

- Python ≥ 3.10
- TensorFlow ≥ 2.15.0
- DeepXDE == 1.14.0
- NumPy ≥ 2.0.0
- SciPy ≥ 1.13.0
- Matplotlib ≥ 3.8.0

## Citation

```bibtex
@article{kim2025pdipinn,
  title={...},
  author={Kim, Jungmin and Yang, Dongin and Lee, Minbaek},
  journal={Submitted to APL Computational Physics},
  year={2025},
  note={Under review}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Minbaek Lee** (Corresponding Author): mlee@inha.ac.kr
- Department of Physics, Inha University, Incheon 22212, Republic of Korea
