import numpy as np
import matplotlib.pyplot as plt

from ts_lasso.ts_lasso import (solve_lasso, _basic_prox_descent,
                               cost_function, regularization_path,
                               cost_path, _backtracking_prox_descent,
                               _fast_prox_descent, _solve_lasso,
                               exact_cost_path)
from levinson.levinson import (whittle_lev_durb, compute_covariance,
                               A_to_B)


def convergence_example():
    np.random.seed(0)
    T = 1000
    n = 50
    p = 15
    X = np.random.normal(size=(T, n))
    X[1:] = 0.25 * np.random.normal(size=(T - 1, n)) + X[:-1, ::-1]
    X[2:] = 0.25 * np.random.normal(size=(T - 2, n)) + X[:-2, ::-1]
    X[2:, 0] = 0.25 * np.random.normal(size=T - 2) + X[:-2, 1]
    X[3:, 1] = 0.25 * np.random.normal(size=T - 3) + X[:-3, 2]

    R = compute_covariance(X, p)
    A, _, _ = whittle_lev_durb(R)
    B0 = A_to_B(A)
    B0 = B0 + 0.1 * np.random.normal(size=B0.shape)

    lmbda = 0.025

    B_basic = B0
    B_decay_step = B0
    B_bt = B0
    L_bt = 0.01
    B_f = B0
    L_f = 0.01
    t_f = 1.0
    M_f = B0

    W = 1. / np.abs(B0)**(1.25)  # Adaptive weighting

    B_star, _ = _solve_lasso(R, B0, lmbda, W,
                             step_rule=0.01, line_srch=1.1,
                             method="fista", eps=-np.inf,
                             maxiter=3000)
    cost_star = cost_function(B_star, R, lmbda=lmbda, W=W)

    N_iters = 100
    N_algs = 4
    GradRes = np.empty((N_iters, N_algs))
    Cost = np.empty((N_iters, N_algs))
    for it in range(N_iters):
        B_basic, err_basic = _basic_prox_descent(
            R, B_basic, lmbda=lmbda, maxiter=1, ss=0.01,
            eps=-np.inf, W=W)
        B_decay_step, err_decay_step = _basic_prox_descent(
            R, B_decay_step, lmbda=lmbda, maxiter=1, ss=1. / (it + 1),
            eps=-np.inf, W=W)
        B_bt, err_bt, L_bt = _backtracking_prox_descent(
            R, B_bt, lmbda, eps=-np.inf, maxiter=1, L=L_bt, eta=1.1,
            W=W)
        B_f, err_f, L_f, t_f, M_f = _fast_prox_descent(
            R, B_f, lmbda, eps=-np.inf, maxiter=1, L=L_f, eta=1.1,
            t=t_f, M0=M_f, W=W)

        Cost[it, 0] = cost_function(B_basic, R, lmbda, W=W)
        Cost[it, 1] = cost_function(B_decay_step, R, lmbda, W=W)
        Cost[it, 2] = cost_function(B_bt, R, lmbda, W=W)
        Cost[it, 3] = cost_function(B_f, R, lmbda, W=W)

        GradRes[it, 0] = err_basic
        GradRes[it, 1] = err_decay_step
        GradRes[it, 2] = err_bt
        GradRes[it, 3] = err_f

    Cost = Cost - cost_star
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(GradRes[:, 0], label="ISTA (constant stepsize)",
                 linewidth=2)
    axes[0].plot(GradRes[:, 1], label="ISTA (1/t stepsize)",
                 linewidth=2)
    axes[0].plot(GradRes[:, 2], label="ISTA with Backtracking Line Search",
                 linewidth=2)
    axes[0].plot(GradRes[:, 3], label="FISTA with Backtracking Line Search",
                 linewidth=2)

    axes[1].plot(Cost[:, 0], linewidth=2)
    axes[1].plot(Cost[:, 1], linewidth=2)
    axes[1].plot(Cost[:, 2], linewidth=2)
    axes[1].plot(Cost[:, 3], linewidth=2)

    axes[1].set_xlabel("Iteration Count")
    axes[0].set_ylabel("Log Gradient Residuals")
    axes[1].set_ylabel("Log (cost - cost_opt)")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")

    axes[0].legend()

    fig.suptitle("Prox Gradient for Time-Series AdaLASSO "
                 "(n = {}, p = {})".format(n, p))
    plt.savefig("figures/convergence.png")
    plt.savefig("figures/convergence.pdf")
    plt.show()
    return


def cross_validation_example():
    T = 2000
    n = 3
    p = 5
    X = np.random.normal(size=(T, n))
    X[1:] = 0.25 * np.random.normal(size=(T - 1, n)) + X[:-1, ::-1]
    X[2:] = 0.25 * np.random.normal(size=(T - 2, n)) + X[:-2, ::-1]
    X[2:, 0] = 0.25 * np.random.normal(size=T - 2) + X[:-2, 1]
    X[3:, 1] = 0.25 * np.random.normal(size=T - 3) + X[:-3, 2]

    X0 = X[:T // 2]
    X1 = X[T // 2:]

    lmbda_path = np.logspace(-5, 1, 1000)
    B_path = regularization_path(X0, p, lmbda_path, W=1.0,
                                 step_rule=0.05, eps=-np.inf,
                                 maxiter=500)
    cost = exact_cost_path(B_path, X1)

    plt.plot(lmbda_path, cost, linewidth=2)
    plt.xscale("log")
    plt.ylabel("Cross Validation MSE")
    plt.xlabel("Regularization Parameter")
    plt.title("Cost path")
    plt.savefig("figures/cost_path.pdf")
    plt.savefig("figures/cost_path.png")
    plt.show()

    tau_colors = ["b", "r", "m", "k", "g"]

    for tau in range(p):
        for i in range(n):
            for j in range(n):
                plt.plot(lmbda_path, B_path[:, tau, i, j],
                         linewidth=1, color=tau_colors[tau], alpha=0.5)
    plt.xscale("log")
    plt.ylabel("B")
    plt.xlabel("Regularization Parameter")
    plt.title("Regularization Path")
    plt.savefig("figures/regularization_path.pdf")
    plt.savefig("figures/regularization_path.png")
    plt.show()
    return


if __name__ == "__main__":
    convergence_example()
    cross_validation_example()
