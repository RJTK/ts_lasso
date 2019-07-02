import unittest
import numpy as np

from levinson.levinson import (yule_walker, A_to_B, B_to_A,
                               whittle_lev_durb)

from ts_lasso.ts_lasso import (soft_threshold, predict,
                               exact_cost_function, cost_function,
                               cost_gradient,
                               compute_covariance, solve_lasso)


class TestMain(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        return

    def test000(self):
        return

    def test_soft_threshold001(self):
        X = np.array([0, 1, 2, -1, -2, 0.5])
        X_xp = np.array([0, 0, 1, 0, -1, 0])
        X_st = soft_threshold(X, 1.0)
        np.testing.assert_array_almost_equal(X_xp, X_st)
        return

    def test_soft_threshold002(self):
        X = np.array([[0, 1, 2, -1, -2, 0.5],
                      [0, 2, -1, -2, 1, -0.5]])
        X_xp = np.array([[0, 0, 1, 0, -1, 0],
                         [0, 1, 0, -1, 0, 0]])
        X_st = soft_threshold(X, 1.0)
        np.testing.assert_array_almost_equal(X_xp, X_st)
        return

    def test_soft_threshold003(self):
        X = np.array([[[0, 1, 2, -1, -2, 0.5],
                       [0, 2, -1, -2, 1, -0.5]]])
        W = np.array([[[1.0, 0.5, 1.0, 1.0, 0.0, 1.0],
                       [1.0, 2.0, 1.0, 1.0, 0.5, 0.25]]])
        X_xp = np.array([[[0, 0.5, 1, 0, -2, 0],
                          [0, 0, 0, -1.0, 0.5, -0.25]]])
        return

    def test_predict001(self):
        X = np.array([[1.0, 2.0]])
        B = np.array([[[2.2, 3.1],
                       [-1.8, 2.6]]])
        x_hat_xp = np.array([8.4, 3.4])
        x_hat = predict(B, X)
        np.testing.assert_array_almost_equal(x_hat_xp, x_hat)
        return

    def test_predict002(self):
        X = np.array([[1.0, 2.0],
                      [-1.0, -2.0]])
        B = np.array([[[2.2, 3.1],
                       [-1.8, 2.6]],
                      [[4.4, 6.2],
                       [-3.6, 5.2]]])
        x_hat_xp = np.array([-8.4, -3.4])
        x_hat = predict(B, X)
        np.testing.assert_array_almost_equal(x_hat_xp, x_hat)
        return

    def test_predict003(self):
        for _ in range(10):
            X = np.random.normal(size=(5, 10))
            B = np.random.normal(size=(5, 10, 10))
            x_hat_xp = (B[0] @ X[0, :] +
                        B[1] @ X[1, :] +
                        B[2] @ X[2, :] +
                        B[3] @ X[3, :] +
                        B[4] @ X[4, :])
            x_hat = predict(B, X)
            np.testing.assert_array_almost_equal(x_hat_xp, x_hat)
        return

    def test_predict004(self):
        zero = np.zeros(3)
        B = np.random.normal(size=(2, 3, 3))
        X = np.random.normal(size=(2, 3))
        x_hat = predict(B, X[-1: max(0, -2)])
        np.testing.assert_array_equal(zero, x_hat)
        return


    def test_exact_cost_function001(self):
        X = np.array([[1.0, 2.0],
                      [-1.0, -2.0]])
        B = np.array([[[2.2, 3.1],
                       [-1.8, 2.6]]])
        cost_xp = (1. / 4) * (np.sum((X[1] - B[0] @ X[0])**2) +
                              np.sum(X[0]**2))
        cost = exact_cost_function(B, X, lmbda=0.0, W=1.0)
        self.assertAlmostEqual(cost, cost_xp)
        return


    def test_exact_cost_function002(self):
        for _ in range(10):
            X = np.random.normal(size=(3, 2))
            B = np.array([[[2.2, 3.1],
                           [-1.8, 2.6]]])
            cost_xp = (1. / 6) * (np.sum((X[2] - B[0] @ X[1])**2) +
                                  np.sum((X[1] - B[0] @ X[0])**2) +
                                  np.sum(X[0]**2))
            cost = exact_cost_function(B, X, lmbda=0.0, W=1.0)
            self.assertAlmostEqual(cost, cost_xp)
        return

    def test_exact_cost_function003(self):
        for _ in range(10):
            X = np.random.normal(size=(3, 2))
            B = np.array([[[2.2, 3.1],
                           [-1.8, 2.6]]])
            cost_xp = ((1. / 6) * (np.sum((X[2] - B[0] @ X[1])**2) +
                                   np.sum((X[1] - B[0] @ X[0])**2) +
                                   np.sum(X[0]**2)) +
                       2 * np.sum(np.abs(B)))
            cost = exact_cost_function(B, X, lmbda=2.0, W=1.0)
            self.assertAlmostEqual(cost, cost_xp)
        return

    def test_exact_cost_function004(self):
        for _ in range(10):
            X = np.random.normal(size=(3, 2))
            W = np.random.normal(size=(2, 2))
            B = np.random.normal(size=(2, 2, 2))
            cost_xp = ((1. / 6) * (
                np.sum((X[2] - B[0] @ X[1] - B[1] @ X[0])**2) +
                np.sum((X[1] - B[0] @ X[0])**2) +
                np.sum(X[0]**2)) +
                       2 * np.sum(np.abs(W * B)))
            cost = exact_cost_function(B, X, lmbda=2.0, W=W)
            self.assertAlmostEqual(cost, cost_xp)
        return

    def test_cost_function000(self):
        n = 2
        p = 2
        T = 200

        for _ in range(10):
            X = np.random.normal(size=(T, n))
            X[1:] = X[:-1] + 0.5 * np.random.normal(size=(T - 1, n))
            R = compute_covariance(X, p_max=p)
            A, _, _ = whittle_lev_durb(R)
            B = A_to_B(A)

            cost_opt = cost_function(B, R)
            cost_larger = cost_function(B + 1e-3 * np.random.normal(size=B.shape), R)
            self.assertGreater(cost_larger, cost_opt)
        return

    def test_cost_function001(self):
        n = 2
        p = 2
        T = 200

        for _ in range(10):
            X = np.random.normal(size=(T, n))
            X[1:] = X[:-1] + 0.5 * np.random.normal(size=(T - 1, n))
            R = compute_covariance(X, p_max=p)
            A, _, _ = whittle_lev_durb(R)
            B = A_to_B(A)

            cost0 = cost_function(B, R)
            cost = cost_function(B, R, lmbda=0.5)
            self.assertAlmostEqual(cost, cost0 + 0.5 * np.sum(np.abs(B)))
        return

    def test_cost_function002(self):
        n = 2
        p = 2
        T = 200

        for _ in range(10):
            X = np.random.normal(size=(T, n))
            X[1:] = X[:-1] + 0.5 * np.random.normal(size=(T - 1, n))
            W = np.random.normal(size=(p, n, n))
            R = compute_covariance(X, p_max=p)
            A, _, _ = whittle_lev_durb(R)
            B = A_to_B(A)

            cost0 = cost_function(B, R)
            cost = cost_function(B, R, lmbda=0.5, W=W)
            self.assertAlmostEqual(cost, cost0 + 0.5 * np.sum(np.abs(B * W)))
        return

    def test_cost_gradient000(self):
        # This only checks that it compiles / runs.
        B = np.random.normal(size=(3, 2, 2))
        R = np.random.normal(size=(3, 2, 2))
        cost_gradient(B, R)
        return

    def test_cost_gradient001(self):
        p = 4
        T = 1000
        n = 2
        X = np.random.normal(size=(T, n))
        X[1:] = X[:-1] + 0.5 * np.random.normal(size=(T - 1, n))
        R = compute_covariance(X, p_max=p)

        A, _, _ = whittle_lev_durb(R)
        B_hat = A_to_B(A)
        g = cost_gradient(B_hat, R)
        np.testing.assert_almost_equal(g, np.zeros_like(g))
        return


class TestProxDescent(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        return

    def _create_case(self, p=1, T=300):
        n = 2
        X = np.random.normal(size=(T, n))
        X[1:] = X[:-1] + 0.5 * np.random.normal(size=(T - 1, n))
        R = compute_covariance(X, p_max=p)
        return R, X

    def test001(self):
        p = 1
        R, X = self._create_case(p, T=1000)
        B_hat, eps = solve_lasso(X, p=p, lmbda=0.0, eps=-np.inf,
                                 maxiter=1000)
        A_hat = B_to_A(B_hat)
        YW = yule_walker(A_hat, R)

        np.testing.assert_almost_equal(YW[1], np.zeros_like(YW[1]))
        self.assertAlmostEqual(eps, 0.0)
        return

    def test002(self):
        p = 5
        R, X = self._create_case(p)
        B_hat, eps = solve_lasso(X, p=p, lmbda=0.0,
                                 eps=-np.inf, maxiter=1000)
        A_hat = B_to_A(B_hat)
        YW = yule_walker(A_hat, R)

        for tau in range(1, p + 1):
            np.testing.assert_almost_equal(YW[tau],
                                           np.zeros_like(YW[tau]))
        self.assertAlmostEqual(eps, 0.0)
        return

    def test003(self):
        p = 5
        lmbda = 0.05

        for _ in range(10):
            R, X = self._create_case(p, T=1000)
            B_hat, eps = solve_lasso(X, p=p, lmbda=lmbda,
                                     maxiter=250, eps=-np.inf)
            self.assertTrue(eps > 0)
            J_star = exact_cost_function(B_hat, X, lmbda=lmbda)
            for _ in range(10):
                J = exact_cost_function(
                    B_hat + 0.025 * np.random.normal(size=B_hat.shape),
                    X, lmbda=lmbda)
                self.assertTrue(J_star < J)
        return

    def test004(self):
        p = 5
        n = 2
        lmbda = 0.01

        for _ in range(10):
            R, X = self._create_case(p, T=1000)
            W = np.random.normal(size=(p, n, n))
            B_hat, eps = solve_lasso(X, p=p, lmbda=lmbda, W=W,
                                     maxiter=250, eps=1e-5)
            self.assertTrue(eps > 0)
            J_star = exact_cost_function(B_hat, X, lmbda=lmbda, W=W)
            for _ in range(10):
                J = exact_cost_function(
                    B_hat + 0.025 * np.random.normal(size=B_hat.shape),
                    X, lmbda=lmbda, W=W)
                self.assertTrue(J_star < J)
        return
