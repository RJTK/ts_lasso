import unittest
import numpy as np


from ts_lasso.ts_lasso import (soft_threshold, predict,
                               cost_function, cost_gradient)


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


    def test_cost_function001(self):
        X = np.array([[1.0, 2.0],
                      [-1.0, -2.0]])
        B = np.array([[[2.2, 3.1],
                       [-1.8, 2.6]]])
        cost_xp = (1. / 4) * (np.sum((X[1] - B[0] @ X[0])**2) +
                              np.sum(X[0]**2))
        cost = cost_function(B, X, lmbda=0.0, W=None)
        self.assertAlmostEqual(cost, cost_xp)
        return


    def test_cost_function002(self):
        for _ in range(10):
            X = np.random.normal(size=(3, 2))
            B = np.array([[[2.2, 3.1],
                           [-1.8, 2.6]]])
            cost_xp = (1. / 6) * (np.sum((X[2] - B[0] @ X[1])**2) +
                                  np.sum((X[1] - B[0] @ X[0])**2) +
                                  np.sum(X[0]**2))
            cost = cost_function(B, X, lmbda=0.0, W=None)
            self.assertAlmostEqual(cost, cost_xp)
        return

    def test_cost_function003(self):
        for _ in range(10):
            X = np.random.normal(size=(3, 2))
            B = np.array([[[2.2, 3.1],
                           [-1.8, 2.6]]])
            cost_xp = ((1. / 6) * (np.sum((X[2] - B[0] @ X[1])**2) +
                                   np.sum((X[1] - B[0] @ X[0])**2) +
                                   np.sum(X[0]**2)) +
                       2 * np.sum(np.abs(B)))
            cost = cost_function(B, X, lmbda=2.0, W=None)
            self.assertAlmostEqual(cost, cost_xp)
        return

    def test_cost_function004(self):
        for _ in range(10):
            X = np.random.normal(size=(3, 2))
            W = np.random.normal(size=(2, 2))
            B = np.random.normal(size=(2, 2, 2))
            cost_xp = ((1. / 6) * (
                np.sum((X[2] - B[0] @ X[1] - B[1] @ X[0])**2) +
                np.sum((X[1] - B[0] @ X[0])**2) +
                np.sum(X[0]**2)) +
                       2 * np.sum(np.abs(W * B)))
            cost = cost_function(B, X, lmbda=2.0, W=W)
            self.assertAlmostEqual(cost, cost_xp)
        return

    def test_cost_gradient000(self):
        # This only checks that it compiles / runs.
        B = np.random.normal(size=(3, 2, 2))
        R = np.random.normal(size=(3, 2, 2))
        cost_gradient(B, R)
        return
