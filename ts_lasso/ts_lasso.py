import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def cost_function(X, B, lmbda=0.0, W=None):
    p, n, _ = B.shape
    T, _ = X.shape
    Xhat = np.zeros_like(X[p + 1:])
    for t in range(p + 1, T):
        Xhat[t] = predict(B, X[t - 1: t - p - 1: -1])
    E = X - Xhat
    cost = (1. / (2 * T)) * np.linalg.norm(E, "fro")

    if lmbda != 0.0:
        if W is not None:
            cost = cost + lmbda * np.linalg.norm(B * W, 1)
        else:
            cost = cost + lmbda * np.linalg.norm(B, 1)
    return cost


@numba.jit(nopython=True, cache=True)
def predict(B, X):
    p, n = X.shape
    Xhat = np.zeros(n)
    for tau in range(p):
        Xhat = Xhat + B[tau] @ X[p - tau]
    return Xhat


@numba.jit(nopython=True, cache=True)
def soft_threshold(Z, tau, W=None):
    """
    Weighted soft-threshold.  This implements

    prox_tau (W \cdot Z)
    """
    if W is None:
        X = np.abs(Z) - tau
    else:
        X = np.abs(Z) - tau * W
    return np.sign(Z) * X * (X >= 0)


@numba.jit(nopython=True, cache=True)
def descent_direction(B, R):
    """
    Returns a descent direction (i.e. grad g)
    """
    return np.zeros_like(B)
