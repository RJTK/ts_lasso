import unittest
import numpy as np


from ts_lasso.ts_lasso import soft_threshold


class TestMain(unittest.TestCase):
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


