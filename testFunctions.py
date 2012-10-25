#!/usr/bin/env python

import functions as F
import numpy as np
import unittest
ADf = F.ApproximateJacobian  # from functions import ApproximateJacobian as ADf


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
        N = 200
        A = np.matrix(np.random.rand(N, N))
        x0 = np.matrix(np.random.rand(N, 1))
        dx = 1.e-6
        f = lambda x: A * x
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (N, N))
        np.testing.assert_array_almost_equal(Df_x, A)

    def estApproxJacobianArrays(self):
        '''Same as testApproxJacobian2, but with arrays rather than matrices'''
        A = np.array(np.matrix("1. 2.; 3. 4."))

        def f(x):
            return A * x
        x0 = np.array(np.matrix("5; 6"))
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2, 2))
        np.testing.assert_array_almost_equal(Df_x, A)

    def testPolynomial(self):
        '''Tests the __call__() method of the functions.Polynomial() class
        by comparing its output to a (slightly) more explicit calculation.'''
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in np.linspace(-2, 2, 11):
            self.assertEqual(p(x), x ** 2 + 2 * x + 3)

    def testPolynomialNegativeCoeffs(self):
        '''Maybe if I tried using negative as well as positive coefficients,
        it would fail? I don't see why it should.'''
        p = F.Polynomial([-2, 5, -6])
        for x in np.linspace(200, 220, 11):
            self.assertEqual(p(x), -2 * x ** 2 + 5 * x - 6)

    def testJacobianLinear(self):
        '''Check several analytical Jacobian functions, Df, against the numerical
        approximation produced from the original function, f.'''
        test_cases_1D = np.arange(-2, 2, .1)
        f = F.Linear()['f']
        Df = F.Linear()['Df']
        for x in test_cases_1D:
            np.testing.assert_array_almost_equal(Df(x), ADf(f, x)[0, 0], decimal=4)

    def testJacobianSkewedSine(self):
        '''Check several analytical Jacobian functions, Df, against the numerical
        approximation produced from the original function, f.'''
        test_cases_1D = np.arange(-2, 2, .1)
        f = F.SkewedSine()['f']
        Df = F.SkewedSine()['Df']
        for x in test_cases_1D:
            np.testing.assert_array_almost_equal(Df(x), ADf(f, x)[0, 0], decimal=4)

    def testJacobianExponential(self):
        '''Check several analytical Jacobian functions, Df, against the numerical
        approximation produced from the original function, f.'''
        test_cases_1D = np.arange(-2, 2, .1)
        f = F.Exponential()['f']
        Df = F.Exponential()['Df']
        for x in test_cases_1D:
            np.testing.assert_array_almost_equal(Df(x), ADf(f, x)[0, 0], decimal=4)

    def testJacobianLogarithmic(self):
        '''Check several analytical Jacobian functions, Df, against the numerical
        approximation produced from the original function, f.'''
        test_cases_1D = np.arange(1, 2, .1)
        f = F.Logarithmic()['f']
        Df = F.Logarithmic()['Df']
        for x in test_cases_1D:
            np.testing.assert_array_almost_equal(Df(x), ADf(f, x)[0, 0], decimal=4)



    def testJacobianQuadraticStrings(self):
        '''Check several analytical Jacobian functions, Df, against the numerical
        approximation produced from the original function, f.'''
        test_cases_1D = np.arange(1, 2, .1)
        f = F.QuadraticStrings()['f']
        Df = F.QuadraticStrings()['Df']
        for x in test_cases_1D:
            np.testing.assert_array_almost_equal(Df(x), ADf(f, x)[0, 0], decimal=4)

    def testJacobianLinear2D(self):
        '''Check several analytical Jacobian functions, Df, against the numerical
        approximation produced from the original function, f.'''
        xs = np.arange(-2, 2, .5)
        ys = np.arange(-2, 2, .5)
        f = F.Linear2D()['f']
        Df = F.Linear2D()['Df']
        for x in xs:
            for y in ys:
                z = np.matrix([[y], [x]])
                np.testing.assert_array_almost_equal(Df(z), ADf(f, z), decimal=4)

if __name__ == '__main__':
    unittest.main()



