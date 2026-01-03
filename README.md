# A Principled Framework for Residual-Based Adaptivity (vRBA)


This repository contains the official JAX implementation for the paper: **A Principled Framework for Residual-Based Adaptivity in Neural PDE Solvers and Operator Learning**.

This code provides a JAX-native implementation of **vRBA**, a principled adaptive sampling and weighting method for PINNs and Neural Operators. It also includes a custom, high-performance **SSBroyden optimizer** for second-order training to accelerate convergence and achieve state-of-the-art accuracy.

## Citation

If you use our code, please cite our work:
```
@article{toscano2025variational,
  title={A Variational Framework for Residual-Based Adaptivity in Neural PDE Solvers and Operator Learning},
  author={Toscano, Juan Diego and Chen, Daniel T and Oommen, Vivek and Darbon, J{\'e}r{\^o}me and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2509.14198},
  year={2025}
}
```

---

## About the Framework

Residual-based adaptive methods are powerful but often heuristic. Our work introduces **vRBA**, a unifying variational framework that provides a formal justification and a principled design strategy for these techniques.

The core idea is to connect adaptive schemes to a primal optimization objective using convex transformations of the PDE residual. This establishes a direct link between the choice of sampling distribution and the error metric being minimized. For example:
* **Exponential weights** correspond to minimizing the **$L^\infty$ (uniform) error**.
* **Linear/quadratic weights** correspond to minimizing the **$L^2$ (mean-squared) error**.

This approach transforms adaptive sampling from a heuristic into a principled optimization strategy.

---

## Key Contributions & Features

### 1. The vRBA Framework
* **Principled Design:** A unified method for creating adaptive sampling and weighting schemes based on formal optimization theory.
* **Reduced Discretization Error:** vRBA provably reduces the variance of the loss estimator, leading to more stable training and a smaller discretization error.
* **Improved Training Dynamics:** The framework enhances the gradient's signal-to-noise ratio (SNR), allowing models to enter the productive "diffusion" phase of training more rapidly and accelerating convergence.
* **Extension to Operator Learning:** We introduce a novel **hybrid strategy** that combines importance weighting for spatial/temporal domains with importance sampling over function instances. This makes vRBA highly effective for architectures like DeepONets, FNOs, and TC-UNets.

### 2. High-Performance Second-Order Optimizer
* This repository includes a custom JAX implementation of the **Self-Scaled Broyden (SSBroyden)** optimizer from Urbán et al..
* Our implementation is designed for **GPU acceleration**, overcoming the performance bottlenecks of the original CPU-bound SciPy version.
* It features a robust **three-stage fallback line search**, making it stable and effective for the challenging, ill-conditioned loss landscapes found in PINNs.
---

## Getting Started

### Data Availability

All datasets for the Operator Learning examplesare available on Google Drive:

[**Download Data Here (Google Drive)**](https://drive.google.com/drive/folders/1IiSt-g_Vs0wKcSbvzgnwCFTBP_v75Tc-?usp=drive_link)

#### Instructions

1.  Download the data from the link above.
2.  The downloaded folder contains several subfolders 
3.  Unzip the files.
4.  Move the contents of *each* of these subfolders into the corresponding, identically-named folder within this project repository.

For example, the 'data' folder from the downloaded `wave_tcunet` folder should be placed into the `wave_tcunet` folder in this repository.

### Prerequisites

This project has two sets of requirements. The core vRBA framework and PINN experiments are built in JAX, while the operator learning benchmarks (FNO, TC-UNet, etc.) are built in PyTorch.

#### 1. JAX Environment (Core & PINNs)

This environment is used for all PINN experiments and the core vRBA logic.

** JAX Version:**
 Our repository includes a custom, high-performance SSBroyden optimizer that was adapted from files within the `jax.scipy` library. Using a different version of `jax` or `jaxlib` than specified *may* lead to errors.

We recommend installing the following exact versions:

```
jax==0.6.0
jax-cuda12-pjrt==0.6.0
jax-cuda12-plugin==0.6.0
jaxlib==0.6.0
jaxopt==0.8.5
jaxtyping==0.3.2
```

#### 2. PyTorch Environment (Operator Learning)

For running the operator learning tasks, you will need a separate environment with PyTorch and related libraries.

Key requirements include:

```
torch==2.7.0
tensorly-torch==0.5.0
torch_harmonics==0.7.3
torchaudio==2.7.0
torchinfo==1.8.0
torchvision==0.22.0
```

#### 3. General Dependencies

Both environments will also require:
* NumPy
* Matplotlib (for visualizations)
