# A Principled Framework for Residual-Based Adaptivity (vRBA)


This repository contains the official JAX implementation for the paper: **A Principled Framework for Residual-Based Adaptivity in Neural PDE Solvers and Operator Learning**.

This code provides a JAX-native implementation of **vRBA**, a principled adaptive sampling and weighting method for PINNs and Neural Operators. It also includes a custom, high-performance **SSBroyden optimizer** for second-order training to accelerate convergence and achieve state-of-the-art accuracy.

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

### Prerequisites
* JAX
* NumPy
* Matplotlib (for visualizations)

