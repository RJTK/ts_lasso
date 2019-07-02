import numpy as np
import matplotlib.pyplot as plt

from ts_lasso.ts_lasso import (solve_lasso, _basic_prox_descent,
                               cost_function, regularization_path,
                               cost_path)
from levinson.levinson import (whittle_lev_durb, compute_covariance,
                               A_to_B)


def convergence_example():
    T = 1000
    n = 30
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

    N_iters = 100
    GradRes = np.empty((N_iters, 2))
    Cost = np.empty((N_iters, 2))
    for it in range(N_iters):
        B_basic, err_basic = _basic_prox_descent(
            R, B_basic, lmbda=lmbda, maxiter=1, ss=0.01,
            eps=-np.inf)
        B_decay_step, err_decay_step = _basic_prox_descent(
            R, B_decay_step, lmbda=lmbda, maxiter=1, ss=1. / (it + 1),
            eps=-np.inf)

        Cost[it, 0] = cost_function(B_basic, X, lmbda)
        Cost[it, 1] = cost_function(B_decay_step, X, lmbda)

        GradRes[it, 0] = err_basic
        GradRes[it, 1] = err_decay_step

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(GradRes[:, 0], label="Basic (constant stepsize)",
                 linewidth=2)
    axes[0].plot(GradRes[:, 1], label="Basic (1/t stepsize)",
                 linewidth=2)

    axes[1].plot(Cost[:, 0], linewidth=2)
    axes[1].plot(Cost[:, 1], linewidth=2)

    axes[1].set_xlabel("Iteration Count")
    axes[0].set_ylabel("Log Gradient Residuals")
    axes[1].set_ylabel("Log Cost Function")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")

    axes[0].legend()

    fig.suptitle("Prox Gradient for Time-Series LASSO")
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
    B_path = regularization_path(X0, p, lmbda_path, W=None,
                                 step_rule=0.05, eps=-np.inf,
                                 maxiter=500)
    cost = cost_path(B_path, X1)

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
