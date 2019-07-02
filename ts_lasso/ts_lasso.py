import numba
import numpy as np

from levinson.levinson import (compute_covariance,
                               whittle_lev_durb,
                               A_to_B)

# TODO: Work out a proper stopping criteria

# TODO: Check my residual calculation, the subdiff is probably wrong

# TODO: Check gradient calculations -- it doesn't line up with the WLD
# TODO: solution, I think there is a misplaced transpose or something.

# TODO: Momentum terms


def regularization_path(X, p, lmbda_path, W=None, step_rule=0.1,
                        line_srch=None, eps=1e-6, maxiter=100):
    """
    Given an iterable for lmbda, return the whole regularization path
    as a 4D array indexed as [lmbda, tau, i, j].
    """
    R = compute_covariance(X, p_max=p)
    B0 = _wld_init(R)

    n = B0.shape[1]

    B_hat = np.empty((len(lmbda_path), p, n, n))

    if type(step_rule) is float:
        solver = lambda B_init, lmbda: _basic_prox_descent(
            R, B0=B_init, lmbda=lmbda, ss=step_rule, eps=eps,
            maxiter=maxiter)
    else:
        raise NotImplementedError("step_rule {} is not supported"
                                  "".format(step_rule))

    B_hat[-1] = B0
    for lmbda_i, lmbda in enumerate(list(lmbda_path)):
        _B, _ = solver(B_hat[lmbda_i - 1], lmbda)
        B_hat[lmbda_i] = _B
    return B_hat


def cost_path(B_path, X, W=None):
    cost = np.empty(B_path.shape[0])
    for i in range(len(cost)):
        cost[i] = cost_function(B_path[i], X, lmbda=0.0, W=W)
    return cost


def solve_lasso(X, p, lmbda=0.0, W=None, step_rule=0.1,
                line_srch=None, eps=1e-6, maxiter=100):
    R = compute_covariance(X, p_max=p)
    B0 = _wld_init(R)

    if type(step_rule) is float:
        B_hat, res = _basic_prox_descent(R, B0=B0, lmbda=lmbda,
                                         ss=step_rule, eps=eps,
                                         maxiter=maxiter)
    else:
        raise NotImplementedError("Only constant stepsize is implemented.")
    return B_hat, res


def _wld_init(R, sigma=0.1):
    """
    We add noise to the yule-walker estimator otherwise the cost
    function gradient will be 0, essentially starting us at a position
    of lower curvature -- which I think is bad.
    """
    A, _, _ = whittle_lev_durb(R)
    B0 = A_to_B(A)
    B0 = B0 + sigma * np.random.normal(size=B0.shape)
    return B0


@numba.jit(nopython=True, cache=True)
def _basic_prox_descent(R, B0, lmbda, ss=0.1, W=None, eps=1e-6,
                        maxiter=100):
    """
    Most basic constant stepsize proximal gradient scheme.
    """
    if W is None:
        W = 1.0

    B = B0
    res = np.inf
    for _ in range(maxiter):
        g = cost_gradient(B, R)
        res_vec = _compute_gradient_residual(B, g, lmbda, W)
        res = np.max(np.abs(res_vec))
        if res < eps:
            return B, res

        Z = B - ss * g
        B = soft_threshold(Z, lmbda * ss, W)
    return B, res


@numba.jit(nopython=True, cache=True)
def _compute_gradient_residual(B, g, lmbda, W):
    Z = (B == 0)

    res = (g * np.logical_and(Z, np.abs(g) > lmbda)) +\
        g * np.logical_not(Z) * W * B
    return res


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

    prox_tau (W * Z)
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

    for s in range(1, p + 1):
        grad[s - 1] = -R[s]
        for tau in range(1, p + 1):
            if s - tau >= 0:
                grad[s - 1] = grad[s - 1] + B[tau - 1] @ R[s - tau]
            else:
                grad[s - 1] = grad[s - 1] + B[tau - 1] @ R[tau - s].T
    return grad
