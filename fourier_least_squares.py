#!/usr/bin/env python3
"""
Least Squares with feature maps — Skeleton 

Complete the TODOs to:
  1) Build the design matrix X using
     ψ(x) = [1, x, sin(x), sin(2x), cos(x), cos(2x), sin(3x), cos(3x), sin(4x), cos(4x)]
  2) Fit Ordinary Least Squares (OLS) via the normal equations
  3) Plot the true step function, noisy samples, and your OLS fit
  4) Add ridge (L2) regularization with λ = 20 and plot that fit

DO NOT change function names or signatures.

How to run (after completing TODOs):
    python fourier_least_squares.py
This will save a figure `fourier_least_squares.png` in the current directory.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Configuration (kept in sync with the reference solution)
# ---------------------------
N_SAMPLES = 50
SEED = 42
NOISE_SIGMA = 0.5
LAMBDA = 20  # Used for ridge fit;

# ---------------------------
# 1) Data generation
# ---------------------------
def make_data(n: int = N_SAMPLES, seed: int = SEED, noise_sigma: float = NOISE_SIGMA):
    """
    Return:
        x: shape (n,) sampled uniformly from [-pi, pi]
        t: shape (n,) targets = step(x) + Gaussian noise with std = noise_sigma
    The true (noise-free) function is:
        f*(x) = -1 if x < 0 else +1
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-np.pi, np.pi, size=n)
    f_true = np.where(x < 0, -1.0, 1.0)
    t = f_true + rng.normal(0, noise_sigma, size=n)
    return x, t



# ---------------------------
# 2) Feature map  ψ(x)
# ---------------------------
def feature_map(x: np.ndarray) -> np.ndarray:
    """
    Input:
        x: shape (n,) array of inputs
    Returns:
        X: design matrix of shape (n, 10) with columns:
           [1, x, sin(x), sin(2x), cos(x), cos(2x), sin(3x), cos(3x), sin(4x), cos(4x)]
    """
    X = np.column_stack([np.ones_like(x), x, np.sin(x), np.sin(2 * x), np.cos(x), np.cos(2 * x), np.sin(3 * x),
    np.cos(3 * x), np.sin(4 * x), np.cos(4 * x)])
    return X


# ---------------------------
# 3) Ordinary Least Squares
# ---------------------------
def fit_ols(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Returns the OLS weights using the normal equations:
        w = (X^T X)^{-1} X^T t (see slides for derivation)
    """
    w = np.linalg.inv(X.T @ X) @ X.T @ t
    return w


# ---------------------------
# 4) Ridge regression (extra credit)
# ---------------------------
def fit_ridge(X: np.ndarray, t: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """
    Ridge solution (penalize large weights) using:
        w = (X^T X + λ I)^{-1} X^T t
    """
    n_features = X.shape[1]
    w = np.linalg.inv(X.T @ X + lam * np.eye(n_features)) @ X.T @ t
    return w


# ---------------------------
# 5) Plotting
# ---------------------------
def plot_results(x: np.ndarray, t: np.ndarray, w_ols: np.ndarray, w_ridge: np.ndarray | None = None,
                 filename: str = "fourier_least_squares.png") -> None:
    """
    Plots the true step function, noisy scatter, OLS fit, and (optional) ridge fit.
    Saves the figure to `filename` and closes the figure.
    """
    xx = np.linspace(-np.pi, np.pi, 1000)
    X_grid = feature_map(xx)

    # True function
    f_true = np.where(xx < 0, -1.0, 1.0)

    plt.figure(figsize=(8, 5))
    plt.plot(xx, f_true, 'k-', label="True step function")
    plt.scatter(x, t, color='red', label="Noisy samples")
    plt.plot(xx, X_grid @ w_ols, 'b-', label="=OLS fit")

    if w_ridge is not None:
        plt.plot(xx, X_grid @ w_ridge, 'g--', label=f"Ridge λ={LAMBDA}")

    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Least Squares with Fourier Feature Map")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ---------------------------
# 6) Main script
# ---------------------------
def main() -> None:
    x, t = make_data()

    # Design matrix
    X = feature_map(x)

    # Fit OLS
    w_ols = fit_ols(X, t)

    # Extra credit: ridge
    try:
        w_ridge = fit_ridge(X, t, lam=LAMBDA)
    except NotImplementedError:
        w_ridge = None

    # Plot both
    plot_results(x, t, w_ols, w_ridge=w_ridge, filename="fourier_least_squares.png")


if __name__ == "__main__":
    main()
