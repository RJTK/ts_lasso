import numba
import numpy as np


def solve_lasso(X, p, lmbda=0.0, W=None, step_rule=0.1,
                line_srch=None):
    return


@numba.jit(nopython=True, cache=True)
def _basic_prox_descent(X, B0, lmbda, ss=0.1, W=None):
    """
    Most basic constant stepsize proximal gradient scheme.
    """
    return


@numba.jit(nopython=True, cache=True)
def cost_function(B, X, lmbda=0.0, W=None):
    p, n, _ = B.shape
    T, _ = X.shape
    err = np.sum(X[0]**2)
    for t in range(1, T):
        if t - p - 1 < 0:
            x_hat = predict(B, X[t - 1::-1])
        else:
            x_hat = predict(B, X[t - 1: t - p - 1: -1])
        err = err + np.sum((X[t] - x_hat)**2)
    cost = (1. / (2 * T)) * err

    if lmbda != 0.0:
        if W is not None:
            cost = cost + lmbda * l1_norm(B * W)
        else:
            cost = cost + lmbda * l1_norm(B)
    return cost


@numba.jit(nopython=True, cache=True)
def predict(B, X):
    """
    Calculates sum_tau B[tau] @ X[t - tau, :].  The array X should
    be in the order [X[t - 1], X[t - 2], ..., X[t - p].
    """
    p, n = X.shape
    Xhat = np.zeros(n)
    for tau in range(p):
        Xhat = Xhat + B[tau] @ X[tau, :]
    return Xhat


@numba.jit(nopython=True, cache=True)
def l1_norm(B):
    return np.sum(np.abs(B))


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
def cost_gradient(B, R):
    """
    Returns a descent direction (i.e. grad g)
    """
    p, n, _ = B.shape
    grad = np.zeros_like(B)

    for s in range(p):
        grad[s] = R[s]
        for tau in range(p):
            if tau - s < 0:
                grad[s] = grad[s] + R[tau - s].T @ B[tau]
            else:
                grad[s] = grad[s] + R[tau - s] @ B[tau]
    return grad
