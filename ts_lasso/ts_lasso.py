import numba
import numpy as np

from levinson.levinson import (compute_covariance,
                               whittle_lev_durb,
                               A_to_B)

# TODO: Work out a proper stopping criteria


def adalasso_bic(X, p_max, nu=1.25):
    T = len(X)
    R = compute_covariance(X, p_max=p_max)
    B0 = _wld_init(R)
    W = 1. / np.abs(B0)**nu
    lmbda_path = np.logspace(-6, 1.0, 250)

    B_path = _regularization_path(R, B0, lmbda_path, W)
    cost = cost_path(B_path, R)

    bic = compute_bic(B_path, cost, T)
    lmbda_i_opt = np.argmax(bic)
    lmbda_star, B_star, cost_star = (lmbda_path[lmbda_i_opt],
                                     B_path[lmbda_i_opt],
                                     cost[lmbda_i_opt])
    return B_star, cost_star, lmbda_star


def compute_bic(B_path, cost, T):
    bic = -cost - (np.log(T) / T) * (np.sum(np.abs(B_path) > 0,
                                            axis=(1, 2, 3)))
    return bic


def regularization_path(X, p, lmbda_path, W=1.0, step_rule=0.1,
                        line_srch=None, eps=1e-6, maxiter=100,
                        method="ista"):
    """
    Given an iterable for lmbda, return the whole regularization path
    as a 4D array indexed as [lmbda, tau, i, j].

    line_srch can either be None for constant step sizes, or a tuple
    (L0, eta) specifying the initial stepsize as 1/L0 and with L
    increasing exponential with factor eta to find workable steps.
    Must have L0 > 0, eta > 1
    """
    R = compute_covariance(X, p_max=p)
    B0 = _wld_init(R)
    return _regularization_path(R, B0, lmbda_path, W=W, step_rule=step_rule,
                                line_srch=line_srch, eps=eps, maxiter=maxiter,
                                method=method)


def _regularization_path(R, B0, lmbda_path, W=1.0, step_rule=0.1,
                        line_srch=None, eps=1e-6, maxiter=100,
                        method="ista"):
    p, n, _ = B0.shape

    B_hat = np.empty((len(lmbda_path), p, n, n))

    solver = lambda B_init, lmbda: _solve_lasso(
        R, B0, lmbda, W, step_rule=step_rule, line_srch=line_srch,
        eps=eps, maxiter=maxiter, method=method)

    B_hat[-1] = B0
    for lmbda_i, lmbda in enumerate(list(lmbda_path)):
        _B, _ = solver(B_hat[lmbda_i - 1], lmbda)
        B_hat[lmbda_i] = _B
    return B_hat


@numba.jit(nopython=True, cache=True)
def cost_path(B_path, R):
    """
    Calculates the cost of each B in B_path without considering the
    reguarlization term.
    """
    cost = np.empty(B_path.shape[0])
    for i in range(len(cost)):
        cost[i] = cost_function(B_path[i], R, lmbda=0.0)
    return cost


@numba.jit(nopython=True, cache=True)
def exact_cost_path(B_path, X):
    cost = np.empty(B_path.shape[0])
    for i in range(len(cost)):
        cost[i] = exact_cost_function(B_path[i], X, lmbda=0.0)
    return cost


def solve_lasso(X, p, lmbda=0.0, W=1.0, step_rule=0.1,
                line_srch=None, eps=1e-6, maxiter=100,
                method="ista"):
    assert np.all(W >= 0), "W must be non-negative"
    R = compute_covariance(X, p_max=p)
    B0 = _wld_init(R)
    return _solve_lasso(R, B0, lmbda, W, step_rule=step_rule,
                        line_srch=line_srch, eps=eps, maxiter=maxiter,
                        method="ista")


def _solve_lasso(R, B0, lmbda, W, step_rule=0.1,
                 line_srch=None, eps=1e-6, maxiter=100,
                 method="ista"):
    if method == "ista":
        if line_srch is None:
            B_hat, res = _basic_prox_descent(
                R, B0=B0, lmbda=lmbda, ss=step_rule, eps=eps,
                maxiter=maxiter, W=W)
        elif type(line_srch) is float:
            eta = line_srch
            assert eta > 1

            B_hat, res, _ = _backtracking_prox_descent(
                R, B0=B0, lmbda=lmbda, W=W, eps=eps,
                maxiter=maxiter, L=1. / step_rule, eta=eta)
        else:
            raise NotImplementedError("line_srch {} is not supported"
                                      "".format(line_srch))
    elif method == "fista":
        if line_srch is None:
            B_hat, res, _, _, _ = _fast_prox_descent(
                R, B0, lmbda=lmbda, W=W, eps=eps, maxiter=maxiter,
                L=1. / step_rule, eta=1.1)
        elif type(line_srch) is float:
            eta = line_srch
            assert eta > 1
            B_hat, res, _, _, _ = _fast_prox_descent(
                R, B0, lmbda=lmbda, W=W, eps=eps, maxiter=maxiter,
                L=1. / step_rule, eta=eta)
        else:
            raise NotImplementedError("Line search {} is not available"
                                      "".format(line_srch))
    else:
        raise NotImplementedError("Method {} not available.".format(method))
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
def _basic_prox_descent(R, B0, lmbda, ss=0.1, W=1.0, eps=1e-6,
                        maxiter=100):
    """
    ISTA with a constant step size.
    """
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
def _backtracking_prox_descent(R, B0, lmbda, W=1.0, eps=1e-6,
                               maxiter=100, L=1.0, eta=1.1):
    """
    ISTA with backtracking line search for automatically
    tuning the stepsize.
    """
    B = B0
    res = np.inf
    for _ in range(maxiter):
        g = cost_gradient(B, R)
        res_vec = _compute_gradient_residual(B, g, lmbda, W)
        res = np.max(np.abs(res_vec))
        if res < eps:
            return B, res, L

        B, L = _line_search(B, R, g, lmbda, W, L, eta)
    return B, res, L


@numba.jit(nopython=True, cache=True)
def _fast_prox_descent(R, B0, lmbda, W=1.0, eps=1e-6,
                       maxiter=100, L=1.0, eta=1.1,
                       t=1.0, M0=None):
    """
    This if FISTA.
    """
    B = B0
    if M0 is not None:
        M = M0  # Momentum
    else:
        M = B0

    res = np.inf
    for _ in range(maxiter):
        # TODO: I should only calculate this every 10 or something
        # TODO: iterations cause the gradient doesn't get reused.
        res_vec = _compute_gradient_residual(B, cost_gradient(B, R),
                                             lmbda, W)
        res = np.max(np.abs(res_vec))
        if res < eps:
            return B, res, L, t, M

        g = cost_gradient(M, R)
        B_next, L = _line_search(M, R, g, lmbda, W, L, eta)
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        M = B_next + ((t - 1) / (t_next)) * (B_next - B)

        t = t_next
        B = B_next

    return B, res, L, t, M


@numba.jit(nopython=True, cache=True)
def _line_search(B, R, g, lmbda, W, L, eta):
    Q0 = cost_function(B, R, lmbda=0.0, W=1.0) - np.sum(B * g)

    for _ in range(100):
        B_hat = soft_threshold(B - g / L, lmbda / L, W)
        cost = cost_function(B_hat, R, lmbda, W)
        Q = (Q0 + np.sum(B_hat * g) + 0.5 * L * np.sum((B - B_hat)**2) +
             lmbda * l1_norm(W * B_hat))
        if cost <= Q:
            break
        else:
            L = eta * L
    return B_hat, L


@numba.jit(nopython=True, cache=True)
def _compute_gradient_residual(B, g, lmbda, W):
    abs_g = np.abs(g)
    lW = lmbda * W
    I_B = (np.abs(B) > 0)
    I_g = (abs_g > lW)
    return (np.abs(lW + np.sign(B) * g) * I_B +
            I_g * (1 - I_B) * np.abs(lW - abs_g))


@numba.jit(nopython=True, cache=True)
def exact_cost_function(B, X, lmbda=0.0, W=1.0):
    """
    Cost function evaluated exactly on the data X.
    """
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

    cost = cost + lmbda * l1_norm(B * W)
    return cost


@numba.jit(nopython=True, cache=True)
def cost_function(B, R, lmbda=0.0, W=1.0):
    """
    Cost function applied to the covariance sequence R(tau)
    """
    p = len(B)
    cost = np.trace(R[0])
    for tau in range(1, p + 1):
        Z = -2 * R[tau]
        for s in range(1, p + 1):
            if tau - s >= 0:
                Z = Z + B[s - 1] @ R[tau - s]
            else:
                Z = Z + B[s - 1] @ R[s - tau].T
        cost = cost + np.sum(Z * B[tau - 1])
    cost = 0.5 * cost

    cost = cost + lmbda * l1_norm(B * W)
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
def soft_threshold(Z, tau, W=1.0):
    """
    Weighted soft-threshold.  This implements

    prox_tau (W * Z)
    """
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
