import numba
import numpy as np
import warnings

from scipy.optimize import fminbound

from levinson.levinson import (compute_covariance,
                               whittle_lev_durb,
                               A_to_B, system_rho)

# TODO: Work out a proper stopping criteria

# TODO: Set the regularization path to end at lmbda_max, i.e. when B = 0
# TODO: there is a closed form expression that tells you apriori this value

DEFAULT_ETA = 1.1
MAXITER = 50000


def fit_VAR(X, p_max, nu=1.25, eps=1e-3):
    if nu is None:
        nu = 1.25
    fit_nu = True

    T = len(X)
    R = compute_covariance(X, p_max=p_max)

    bic_star = -np.inf
    cost_star = np.inf
    lmbda_star = None
    B_star = None
    for p in range(1, p_max + 1):
        B, cost, lmbda, bic = _adalasso_bic(R[:p + 1], T, p, nu,
                                            lmbda_max=None, eps=eps)

        if bic > bic_star:
            B_star = B
            cost_star = cost
            bic_star = bic
            lmbda_star = lmbda

        elif bic < 0.75 * bic_star:
            break

    while np.all(B_star[-1] == 0) and len(B_star) > 1:
        B_star = B_star[:-1]

    if fit_nu:
        p_star = len(B_star)
        B_star, cost_star, lmbda_star, bic_star, nu_star =\
            _adalasso_bic_nu(R[:p_star + 1], T, eps=eps)
        return B_star, cost_star, lmbda_star, bic_star, nu_star
    else:
        return B_star, cost_star, lmbda_star, bic_star


def adalasso_bic_nu(X, p, eps=1e-3):
    """
    Fits the whole deal as well as choosing nu via BIC.
    """
    T = len(X)
    R = compute_covariance(X, p_max=p)
    return _adalasso_bic_nu(R, T, eps)


def _adalasso_bic_nu(R, T, eps=1e-3):
    p = len(R) - 1

    def bic(nu):
        B_star, cost_star, lmbda_star, bic = _adalasso_bic(
            R, T, p, nu, eps=eps)
        return B_star, cost_star, lmbda_star, -bic

    nu_star, _bic_star, err, _ = fminbound(
        lambda nu: bic(nu)[-1], x1=0.5, x2=2.0,
        xtol=1e-2, full_output=True)

    if err:
        warnings.warn("fminbound exceeded maximum iterations")

    B_star, cost_star, lmbda_star, neg_bic_star = bic(nu_star)
    return B_star, cost_star, lmbda_star, -neg_bic_star, nu_star


def adalasso_bic(X, p, nu=1.25, lmbda_max=None):
    """
    Fit a VAR(p) model by optimizing BIC with a bisection method.
    This will be faster than adalasso_bic_path, but it is possible it
    will pick a bad regularization parameter.  It also (obviously)
    won't return the BIC path.
    """
    T = len(X)
    R = compute_covariance(X, p_max=p)
    return _adalasso_bic(R, T, p, nu, lmbda_max)


def _adalasso_bic(R, T, p, nu, lmbda_max=None, eps=1e-3):
    B0 = _wld_init(R)
    W = 1. / np.abs(B0)**nu
    B0 = dither(B0)

    if lmbda_max is None:
        lmbda_max = get_lmbda_max(R, W)

    def solve_cost(lmbda):
        B_hat, _ = _solve_lasso(R, B0, lmbda, W, method="fista",
                                eps=eps)
        cost = cost_function(B_hat, R)
        return B_hat, cost

    def bic(lmbda):
        B_hat, cost = solve_cost(lmbda)
        return -compute_bic(B_hat[None, ...], cost, T)[0]

    lmbda_star, _bic_star, err, _ = fminbound(
        bic, x1=0, x2=lmbda_max, xtol=1e-2, full_output=True)

    if err:
        warnings.warn("fminbound exceeded maximum iterations")
    bic_star = -1 * _bic_star

    B_star, cost_star = solve_cost(lmbda_star)
    return B_star, cost_star, lmbda_star, bic_star


def get_lmbda_max(R, W=None):
    if W is None:
        return np.max(np.abs(R[1:]))
    else:
        return np.max(np.abs(R[1:]) / W)


def adalasso_bic_path(X, p, nu=1.25, lmbda_path=np.logspace(-6, 1.0, 250),
                      eps=1e-3):
    """
    Fit a VAR(p) model by solving lasso and searching for optimal
    regularizer by solving lasso along a regularization path, and then
    using the BIC criterion.
    """
    T = len(X)
    R = compute_covariance(X, p_max=p)
    return _adalasso_bic_path(R, T, p, nu=nu, lmbda_path=lmbda_path, eps=eps)


def _adalasso_bic_path(R, T, p, nu, lmbda_path, eps=1e-3):
    B0 = _wld_init(R)
    W = 1. / np.abs(B0)**nu
    B0 = dither(B0)

    # eps around 1e-3 to 1e-4 is fast and I think sufficient accuracy
    B_path = _regularization_path(R, B0, lmbda_path, W,
                                  method="fista", eps=eps)
    cost = cost_path(B_path, R)

    bic = compute_bic(B_path, cost, T)
    lmbda_i_opt = np.argmax(bic)
    B_star, cost_star = (B_path[lmbda_i_opt],
                         cost[lmbda_i_opt])
    return B_star, cost_star, lmbda_path, bic


def compute_bic(B_path, cost, T):
    bic = -cost - (np.log(T) / T) * (np.sum(np.abs(B_path) > 0,
                                            axis=(1, 2, 3)))
    return bic


def regularization_path(X, p, lmbda_path, W=1.0, step_rule=0.1,
                        line_srch=None, eps=1e-6, maxiter=MAXITER,
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
    B0 = dither(_wld_init(R))
    return _regularization_path(R, B0, lmbda_path, W=W, step_rule=step_rule,
                                line_srch=line_srch, eps=eps, maxiter=maxiter,
                                method=method)


def _regularization_path(R, B0, lmbda_path, W=1.0, step_rule=0.1,
                         line_srch=None, eps=1e-6, maxiter=MAXITER,
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
                line_srch=None, eps=1e-6, maxiter=MAXITER,
                method="ista"):
    assert np.all(W >= 0), "W must be non-negative"
    R = compute_covariance(X, p_max=p)
    B0 = dither(_wld_init(R))
    return _solve_lasso(R, B0, lmbda, W, step_rule=step_rule,
                        line_srch=line_srch, eps=eps, maxiter=maxiter,
                        method="ista")


def _solve_lasso(R, B0, lmbda, W, step_rule=0.1,
                 line_srch=None, eps=1e-6, maxiter=MAXITER,
                 method="ista"):
    if line_srch is None:
        eta = DEFAULT_ETA
    elif type(line_srch) in (int, float):
        eta = float(line_srch)
    else:
        raise NotImplementedError("line_srch {} is not supported"
                                  "".format(line_srch))
    assert eta > 1

    if method == "ista":
        B_hat, res, _ = _backtracking_prox_descent(
            R, B0=B0, lmbda=lmbda, W=W, eps=eps,
            maxiter=maxiter, L=1. / step_rule, eta=eta)
    elif method == "fista":
        B_hat, res, _, _, _ = _fast_prox_descent(
            R, B0=B0, lmbda=lmbda, W=W, eps=eps, maxiter=maxiter,
            L=1. / step_rule, eta=eta)

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
    return B0


def dither(B0, sigma=0.1):
    """
    Adds some noise to the array B0.  It seems reasonable to initialize
    the algorithm with B0 = dither(_wld_init(R)) since if we just use
    B0 = _wld_init(R) we will be starting at a point with 0 cost gradient.
    However, it is NOT wise to use the dithered B0 for the weight matrix,
    for obvious reasons.
    """
    return B0 + sigma * np.random.normal(size=B0.shape)


@numba.jit(nopython=True, cache=True)
def _basic_prox_descent(R, B0, lmbda, ss=0.1, W=1.0, eps=1e-6,
                        maxiter=MAXITER):
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
                               maxiter=MAXITER, L=1.0, eta=1.1):
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
                       maxiter=MAXITER, L=1.0, eta=1.1,
                       t=1.0, M0=None):
    """
    This is FISTA.
    """
    # TODO: There seems to be some problem with numba when
    # TODO: leaving M0=None ??  For now I just always pass
    # TODO: in M0 = B0...
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

    max_iter = 100
    for it in range(1, max_iter + 1):
        B_hat = soft_threshold(B - g / L, lmbda / L, W)
        cost = cost_function(B_hat, R, lmbda, W)
        Q = (Q0 + np.sum(B_hat * g) + 0.5 * L * np.sum((B - B_hat)**2) +
             lmbda * l1_norm(W * B_hat))
        if cost <= Q:
            break
        else:
            L = eta * L

        # These are kind of hacks but they work.
        if it == max_iter // 2:
            eta *= 1.5
        if it == 2 * max_iter // 3:
            eta *= 1.5
        if L == np.inf:
            print("WARNING: Line search failed with L == inf "
                  "and lmbda = ", lmbda)
            return B_hat, L

    if it >= max_iter:
        print("WARNING: Line search exceeded", max_iter, "iterations and "
              "terminated with L = ", L, "and max(|B|) = ", np.max(np.abs(B)))
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
