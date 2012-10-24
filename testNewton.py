#!/usr/bin/env python

import newton
import unittest
import numpy as np


class TestNewton(unittest.TestCase):
    def testLinear(self):
        '''Tests newton.solve() with a linear function of one variable.'''
        f = lambda x: 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

    def testStep1Var(self):
        '''Tests newton.step() with a single-variable linear function.
        x_{k+1} = x - Df(x)^{-1}*f(x)'''
        slope = 9.0
        intercept = 4.0
        f = lambda x: slope * x + intercept
        solver = newton.Newton(f)
        x0 = 4.0
        stepresult = solver.step(x0)
        correct = x0 - f(x0) / slope
        np.testing.assert_array_almost_equal(stepresult, np.matrix([[correct]]))

    def testStep2Var(self):
        '''Tests newton.step() with linear functions of two variables.
        x_{k+1} = x - Df(x)^{-1}*f(x)'''
        slope_matrix = np.matrix([[5.0, 12.8], [15.90, 3.14159]])
        intercept_matrix = np.matrix([[12], [16]])
        f = lambda x: np.dot(slope_matrix, x) + intercept_matrix
        solver = newton.Newton(f)
        x0 = np.matrix([[3.0], [5.6]])
        stepresult = solver.step(x0)
#        correct = x0 - np.linalg.solve(slope_matrix, f(x0))  # This also works.
        correct = x0 - np.dot(np.linalg.inv(slope_matrix), f(x0)) # TODO maybe I should do this manually
        np.testing.assert_array_almost_equal(stepresult, correct)

    def testStepBigVar(self):
        '''Tests newton.step() with linear functions of many variables.
        x_{k+1} = x - Df(x)^{-1}*f(x)'''
        N = 100
        slope_matrix = np.random.rand(N, N)
        intercept_matrix = np.random.rand(N, 1)
        f = lambda x: np.dot(slope_matrix, x) + intercept_matrix
        solver = newton.Newton(f)
        x0 = np.random.rand(N, 1)
        stepresult = solver.step(x0)
        correct = x0 - np.dot(np.linalg.inv(slope_matrix), f(x0))
        # the following sometimes fails erroneously for the default value of
        # decimal=6 . It seems to always fail for decimal=7
        np.testing.assert_array_almost_equal(stepresult, correct, decimal=5)

    def testFunctionKwarg(self):
        '''Tests newton.step() with a single-variable linear function,
        and also makes use of step()'s fx=??? keyword argment.'''
        pass

    def testQuadratic(self):
        '''Tests newton.solve() with a quadratic function of one variable.'''
        f = lambda x: 5 * x ** 2 + 3 * x - 6
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x1actual = -1.4358
        x2actual = 0.83578
        x1 = solver.solve(-2)
        x2 = solver.solve(1)
        np.testing.assert_array_almost_equal(x1, np.matrix([[x1actual]]), decimal=2)
        np.testing.assert_array_almost_equal(x2, np.matrix([[x2actual]]), decimal=2)

#        Alternately, it should fail if there aren't any (real ones)!

    def testPolynomial(self):
        '''Try solving a polynomial, using the Polynomial class.'''
        pass

    def testSine(self):
        '''This might actually be a bad idea.'''
        pass

if __name__ == "__main__":
    unittest.main()
