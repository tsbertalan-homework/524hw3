#!/usr/bin/env python

import functions as F
import numpy as np
import unittest


class TestFunctions(unittest.TestCase):
    def testApproxJacobian1(self):
        '''For a linear function of one variable, checks
            (1) that the Jacobian's shape is (1, 1)
            (2) that the Jacobian is correct.'''
        slope = 3.0

        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1, 1))
        self.assertAlmostEqual(Df_x, slope)

    def testApproxJacobian2(self):
        '''For a linear function of two variables, checks
            (1) that the Jacobian's shape is (2, 2)
            (2) that the Jacobian is correct (via ...almost_equal...)'''
        A = np.matrix("1. 2.; 3. 4.")

        def f(x):
            return A * x
        x0 = np.matrix("5; 6")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2, 2))
        np.testing.assert_array_almost_equal(Df_x, A)

    def testApproxJacobianRandom(self):
        '''This currently fails because only matrices (not arrays, as returned
        by np.random.rand() are properly handled.'''
        N = 20
        A = np.random.rand(N,N)
        x0 = np.random.rand(N,1)
        dx = 1.e-6
        f = lambda x: A * x
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (N, N))
        np.testing.assert_array_almost_equal(Df_x, A)

    def testPolynomial(self):
        '''Tests the __call__() method of the functions.Polynomial() class
        by comparing its output to a (slightly) more explicity calculation.'''
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in np.linspace(-2, 2, 11):
            self.assertEqual(p(x), x ** 2 + 2 * x + 3)

    def testPolynomialNegativeCoeffs(self):
        '''Maybe if I tried using negative as well as positive coefficients,
        it would fail?'''
        pass


if __name__ == '__main__':
    unittest.main()



