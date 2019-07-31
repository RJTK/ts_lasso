"""
As the number of samples goes to oo, the fit_VAR method
should correctly recover the true data generating system.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cmath
import pandas as pd

from ts_lasso.ts_lasso import fit_VAR
from scipy import stats, signal
from sklearn.metrics import matthews_corrcoef


def cvg_example():
    nu = 1.25
    n = 6
    p = 2
    p_max = 2 * p
    T_iters = [150, 300, 500, 1000, 2000,
               5000, 15000, 55000]
    T_max = max(T_iters) + 1
    T_trials = len(T_iters)
    N_iters = 100

    Err_seq = np.zeros((N_iters, T_trials, p_max, n, n))
    MCC_seq = np.zeros((N_iters, T_trials))

    for N_it in range(N_iters):
        X, B, G = generate_data(T_max, n=n, p=p, q=0.5)
        for T_it, T in enumerate(T_iters):
            B_hat, _, _, _ = fit_VAR(X[:T, :], p_max=p_max, nu=nu)
            p_hat = B_hat.shape[0]
            B_true = np.zeros_like(B_hat)
            B_true[:p, ...] = B

            Err_seq[N_it, T_it, :p_hat] = B_true - B_hat
            MCC_seq[N_it, T_it] = matthews_corrcoef(
                np.ravel(np.abs(B_true)) > 0,
                np.ravel(np.abs(B_hat)) > 0)
        print("iter {} / {}".format(N_it + 1, N_iters))

    Err_seq = Err_seq.swapaxes(0, 1)
    MCC_seq = MCC_seq.swapaxes(0, 1)
    err = Err_seq.reshape((T_trials, -1))

    fig, ax_err = plt.subplots(1, 1)
    ax_mcc = ax_err.twinx()
    for T_it, T in enumerate(T_iters):
        E = err[T_it]
        E = E[E != 0]

        bp = ax_err.boxplot(E, positions=range(T_it, T_it + 1),
                            widths=0.70, patch_artist=True,
                            labels=[str(T)], sym="k.",
                            showfliers=False)
        c = plt.cm.viridis(T_it / T_trials)
        bp["boxes"][0].set(facecolor=c)
        bp["medians"][0].set(color="k")

        mcc = np.mean(MCC_seq[T_it])
        mcc_err = np.abs(mcc - np.percentile(MCC_seq[T_it], [10, 90])[:, None])
        ax_mcc.bar(x=T_it, height=mcc, width=0.75, yerr=mcc_err,
                   bottom=0, align="center", color="gray",
                   alpha=0.5, ecolor="r", capsize=10)
        ax_mcc.grid(False)
    ax_err.set_xlabel("Number of Samples $T$")
    ax_err.set_ylabel("Error")
    ax_mcc.set_ylabel("Mean MCC")
    ax_mcc.set_ylim(0, 1.1)
    ax_err.set_title("TS-LASSO Convergence "
                     "$(n = {}, p = {}, p_{{max}} = {})$"
                     "".format(n, p, p_max))
    ax_err.plot([], [], color="gray", label="Mean MCC")
    ax_err.plot([], [], color="r", label="MCC 10th to 90th percentiles")
    ax_err.legend(loc="upper center")
    fig.savefig("figures/ts_lasso_convergence.png",
                bbox_inches=0, pad_inches=0)
    fig.savefig("figures/ts_lasso_convergence.pdf",
                bbox_inches=0, pad_inches=0)
    plt.show()

    return


def make_B(n=2, p=1, q=0.3):
    B = np.random.normal(size=(p, n, n))
    G = np.random.binomial(n=1, p=q, size=(n, n))
    G[range(n), range(n)] = 1.0
    for i in range(n):
        _, b = random_arma(p, 1, k=1, p_radius=0.75)
        b = -b[1:]
        B[:, i, i] = b
    B = B * G.T

    tril_i, tril_j = np.tril_indices(n, k=-1)
    B[:, tril_i, tril_j] = 0
    G[tril_i, tril_j] = 0

    return B, G


def generate_data(T_max=1000, n=2, p=1, q=0.3, T_burn=5000):
    B, G = make_B(n, p, q=q)

    V = VAR(B)
    sigma = 0.5 + np.random.exponential(0.5, size=n)
    U = np.random.normal(size=(T_burn + T_max, n),
                         scale=sigma)
    X = V.drive(U)
    return X[T_burn:], B, G


def random_arma(p, q, k=1, z_radius=1, p_radius=0.75):
    """
    Returns a random ARMA(p, q) filter.  The parameters p and q define
    the order of the filter where p is the number of AR coefficients
    (poles) and q is the number of MA coefficients (zeros).  k is the
    gain of the filter.  The z_radius and p_radius paramters specify the
    maximum magnitude of the zeros and poles resp.  In order for the
    filter to be stable, we should have p_radius < 1.  The poles and
    zeros will be placed uniformly at random inside a disc of the
    specified radius.

    We also force the coefficients to be real.  This is done by ensuring
    that for every complex pole or zero, it's recipricol conjugate is
    also present.  If p and q are even, then all the poles/zeros could
    be complex.  But if p or q is odd, then one of the poles and or
    zeros will be purely real.

    The filter must be causal.  That is, we assert p >= q.
    Finally, note that in order to generate complex numbers uniformly
    over the disc we can't generate R and theta uniformly then transform
    them.  This will give a distribution concentrated near (0, 0).  We
    need to generate u uniformly [0, 1] then take R = sqrt(u).  This can
    be seen by starting with a uniform joint distribution f(x, y) =
    1/pi, then applying a transform to (r, theta) with x = rcos(theta),
    y = rsin(theta), calculating the distributions of r and theta, then
    applying inverse transform sampling.
    """
    assert(p >= q), "System is not causal"
    P = []
    Z = []
    for i in range(p % 2):
        pi_r = stats.uniform.rvs(loc=-p_radius, scale=2 * p_radius)
        P.append(pi_r)

    for i in range((p - (p % 2)) // 2):
        pi_r = np.sqrt(stats.uniform.rvs(loc=0, scale=p_radius))
        pi_ang = stats.uniform.rvs(loc=-np.pi, scale=2 * np.pi)
        P.append(cmath.rect(pi_r, pi_ang))
        P.append(cmath.rect(pi_r, -pi_ang))

    for i in range(q % 2):
        zi_r = stats.uniform.rvs(loc=-z_radius, scale=2 * z_radius)
        Z.append(zi_r)

    for i in range((q - (q % 2)) // 2):
        zi_r = stats.uniform.rvs(loc=0, scale=z_radius)
        zi_ang = stats.uniform.rvs(loc=-np.pi, scale=2 * np.pi)
        Z.append(cmath.rect(zi_r, zi_ang))
        Z.append(cmath.rect(zi_r, -zi_ang))

    b, a = signal.zpk2tf(Z, P, k)
    return b, a


class VAR:
    '''VAR(p) system'''
    def __init__(self, B, x_0=None):
        '''Initializes the model with a list of coefficient matrices
        B = [B(1), B(2), ..., B(p)] where each B(\tau) \in \R^{n \times n}

        x_0 can serve to initialize the system output (if len(x_0) == n)
        or the entire system state (if len(x_0) == n * p)
        '''
        self.p = len(B)
        self.n = (B[0].shape[0])
        if not all(
                len(B_tau.shape) == 2 and  # Check B(\tau) is a matrix
                B_tau.shape[0] == self.n and  # Check compatible sizes
                B_tau.shape[1] == self.n  # Check square
                for B_tau in B):
            raise ValueError('Coefficients must be square matrices of '
                             'equivalent sizes')
        self.B = B  # Keep the list of matrices
        self._B = np.hstack(B)  # Standard form layout \hat{x(t)} = B^\T z(t)
        self.t = 0

        self.reset(x_0=x_0)  # Reset system state
        return

    def get_companion(self):
        '''Return the companion matrix for the system

        C =
        [B0, B1, B2, ... Bp-1]
        [ I,  0,  0, ... 0   ]
        [ 0,  I,  0, ... 0   ]
        [ 0,  0,  I, ... 0   ]
        [ 0,  0, ..., I, 0   ]
        '''
        n, p = self.n, self.p
        C = np.hstack((np.eye(n * (p - 1)),  # The block diagonal I
                       np.zeros((n * (p - 1), n))))  # right col
        C = np.vstack((np.hstack((B_tau for B_tau in self.B)),  # top row
                       C))
        return C

    def is_stable(self, margin=1e-6):
        '''Check whether the system is stable.  See also self.get_rho().
        We return True if |\lambda_max(C)| <= 1 - margin.  Note that the
        default margin is very small.
        '''
        rho = self.get_rho()
        return rho <= 1 - margin

    def get_rho(self):
        '''Computes and returns the stability coefficient rho.  In order to do
        this we directly calculate the eigenvalues of the block companion
        matrix induced by B, which is of size (np x np).  This may be
        prohibitive for very large systems.

        Stability is determined by the spectral radius of the matrix:

        C =
        [B0, B1, B2, ... Bp-1]
        [ I,  0,  0, ... 0   ]
        [ 0,  I,  0, ... 0   ]
        [ 0,  0,  I, ... 0   ]
        [ 0,  0, ..., I, 0   ]

        .  Note that the
        default margin is very small.
        '''
        C = self.get_companion()
        ev = np.linalg.eigvals(C)  # Compute the eigenvalues
        return max(abs(ev))

    def reset(self, x_0=None, reset_t=True):
        '''Reset the system to some initial state.  If x_0 is specified,
        it may be of dimension n or n * p.  If it is dimension n, we simply
        dictate the value of the current output, otherwise we reset the
        whole system state.  If reset_t is True then we set the current
        time to reset_t'''
        n, p = self.n, self.p
        if x_0 is not None:
            if len(x_0) == n:  # Initialize just the output
                self._z = np.zeros(n * p)
                self._z[:n] = x_0
            elif len(x_0) == n * p:  # Initialize whole state
                self._z = x_0
            else:
                raise ValueError('Dimension %d of x_0 is not compatible with '
                                 'system dimensions n = %d, p = %d' %
                                 (len(x_0), n, p))

        else:
            self._z = np.zeros(n * p)
        self.x = self._z[:n]  # System output
        if reset_t:
            self.t = 0
        return

    def drive(self, u):
        '''
        Drives the system with input u.  u should be a T x n array
        containing a sequence of T inputs, or a single length n input.
        '''
        n, p = self.n, self.p
        if len(u.shape) == 1:  # A single input
            try:
                u = u.reshape((1, n))  # Turn it into a vector
            except ValueError:
                raise ValueError('The length %d of u is not compatible with '
                                 'system dimensions n = %d, p = %d'
                                 % (len(u), n, p))

        if u.shape[1] != n:  # Check dimensions are compatible
            raise ValueError('The dimension %d of the input vectors is '
                             'not compatible with system dimensions n = %d, '
                             ' p = %d' % (u.shape[1], n, p))

        T = u.shape[0]  # The number of time steps
        self.t += T

        # Output matrix to be returned
        Y = np.empty((T, n))
        for t in range(T):
            y = self.estimate_next_step() + u[t, :]
            Y[t, :] = y
            self.update_state(y)
        self.x = self._z[:n]  # System output
        return Y

    def estimate_next_step(self):
        return np.dot(self._B, self._z)

    def update_state(self, x):
        self._z = np.roll(self._z, self.n)
        self._z[:self.n] = x
        return

