#!/usr/bin/env python

import newton
import unittest
import numpy as np
import functions as F

class TestNewton(unittest.TestCase):
    def testLinear(self):
        '''Tests newton.solve() with a linear function of one variable.'''
        f = lambda x: 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=3) # TODO why is maxiter=3 required?
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

    def testTolerance(self):
        '''Is the tol=??? keyword actually being set? Is it being used?'''
        f = lambda x: 3.0 * x + 6.0
        tolerance = 1.e-15
        solver = newton.Newton(f, tol=tolerance)
        self.assertEqual(solver._tol, tolerance)
        x = solver.solve(2.0)
        self.assertTrue(np.linalg.norm(f(x)) < tolerance)

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
        correct = x0 - np.linalg.solve(slope_matrix, f(x0)) # TODO I should also try using OpenMG.solve()
        np.testing.assert_array_almost_equal(stepresult, correct)
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
        np.testing.assert_array_almost_equal(stepresult, correct, decimal=4)

    def testFunctionKwarg(self):
        '''Tests newton.step() with a single-variable linear function,
        and also makes use of step()'s fx=??? keyword argment.'''
        pass

    def testQuadratic(self):
        '''Tests newton.solve() with a quadratic function of one variable.'''
        f = lambda x: 5 * x ** 2 + 3 * x - 6
        solver = newton.Newton(f, tol=1.e-15)
        x1actual = -1.4358
        x2actual = 0.83578
        x1 = solver.solve(-2)
        x2 = solver.solve(1)
        np.testing.assert_array_almost_equal(x1, np.matrix([[x1actual]]), decimal=2)
        np.testing.assert_array_almost_equal(x2, np.matrix([[x2actual]]), decimal=2)

    def testMaxIterationsException(self):
        '''Tests newton.solve() with a quadratic function of one variable THAT
        HAS NO REAL ROOTS. The maximum number of iterations should be reached
        quickly.'''
        f = lambda x: 5 * x ** 2 + 3 * x + 6
        solver = newton.Newton(f, tol=1.e-15, maxiter=3)
        self.assertRaises(newton.TooManyIterations, solver.solve, 2)

    def testAnalyticalJacobian1D(self):
        '''In 1D, Supply a Jacobian function to newton.__init__(), and check
        (1) that the solver._jacobian member function is that function
        (2) that the solution is still good.'''
        f = lambda x: 3.0 * x ** 2 + 4.0 * x - 9.0
        Df = lambda x: 6.0 * x + 4.0
        solver = newton.Newton(f, jacobian=Df)
        # self.assertIs(solver._jacobian, Df) # this doesn't work.
        a = 'dummy'  # solver._jacobian has an underscore for a reason:
        b = a  # it requires these dummy arguments, to be compatible with functions.ApproximateJacobian()
        for x in xrange(-10, 10, 30):
            self.assertEqual(solver._jacobian(a, x, b), Df(x))  # is this just busywork?
        x1actual = -2.52259
        x2actual = 1.18925
        x1 = solver.solve(-3.0)
        x2 = solver.solve(2)
        self.assertAlmostEqual(x1, x1actual, places=4)
        self.assertAlmostEqual(x2, x2actual, places=4)

    def testAnalyticalJacobian2D(self):
        '''In 2D, Supply a Jacobian function to newton.__init__(), and check
        (1) that the solver._jacobian member function is that function
        (2) that the solution is still good.'''
        def f(x):
            f1 = F.Polynomial([3.0, 4.0, -9.0])  # solutions of this polynomial are -2.522588 and 1.189285
            f2 = F.Polynomial([9.3, 2.1, -5.6])   # solutions of this polynomial are -0.897057 and 0.671251
            y = np.matrix(np.zeros((2, 1)))  # should have solutions (-2.52259, 0.491228) and (1.18925, 0.491228)
            y[0] = f1(x[0])
            y[1] = f2(x[1])
            return y

        def Df(x):
            f00 = F.Polynomial([6.0, 4.0])
            f11 = F.Polynomial([18.6, 2.1])
            dy = np.matrix(np.zeros((2, 2)))
            dy[0, 0] = f00(x[0])
            dy[1, 0] = 0
            dy[0, 1] = 0
            dy[1, 1] = f11(x[1])
            return dy
        solver = newton.Newton(f, jacobian=Df)
        # self.assertIs(solver._jacobian, Df) # this doesn't work.
        a = b = 'dummy'  # solver._jacobian has an underscore for a reason:
        # it requires these dummy arguments, to be compatible with functions.ApproximateJacobian()
#        for x in xrange(-10, 10, 30):
#            self.assertEqual(solver._jacobian(a, x, b), Df(x))  # is this just busywork?
        x1actual = np.matrix("-2.522588 ; -0.897057")
        x2actual = np.matrix("1.189285  ; 0.671251")
        x01 = np.matrix("-3; -1")
        x02 = np.matrix(" 2;  1")
        x1 = solver.solve(x01, verbose=False)
        x2 = solver.solve(x02, verbose=False)
        np.testing.assert_array_almost_equal(x1, x1actual, decimal=4)
        np.testing.assert_array_almost_equal(x2, x2actual, decimal=4)

    def testMixedJacobian(self):
        '''solves f1(x,y) and f2(x,y), rather than simply f1(x) and f2(y)'''
        def f(X):
            f1x = F.Polynomial([1,  2, -3])
            f1y = F.Polynomial([2,  1, -4])
            f2x = F.Polynomial([4, -3, -2])
            f2y = F.Polynomial([-2, 4, -2])
            y = np.matrix(np.zeros((2, 1)))
            y[0] = f1x(X[0]) + f1y(X[1])
            y[1] = f2x(X[0]) + f2y(X[1])
            return y

        def Df(X):
            f00 = F.Polynomial([2,  2])  # function of X[0] only
            f01 = F.Polynomial([4,  1])  # function of X[1] only
            f10 = F.Polynomial([8, -3])
            f11 = F.Polynomial([-4, 4])
            dy = np.matrix(np.zeros((2,2)))
            dy[0, 0] = f00(X[0])
            dy[0, 1] = f00(X[1])
            dy[1, 0] = f00(X[0])
            dy[1, 1] = f00(X[1])
            return dy

        solver = newton.Newton(f)
        a = b = 'dummy'
        x1actual = np.matrix("-1.98594 ; -2.14115")
        x2actual = np.matrix("-0.582749 ; 1.74385")
        x3actual = np.matrix("1.17623 ; 1.05174")
        x4actual = np.matrix("1.79246 ; -0.654438")
        x01 = np.matrix("-2; -2")
        x02 = np.matrix("-1;  2")
        x03 = np.matrix(" 2;  1")
        x04 = np.matrix(" 2;  -1")
        x1 = solver.solve(x01)
        x2 = solver.solve(x02)
        x3 = solver.solve(x03)
        x4 = solver.solve(x04)
        np.testing.assert_array_almost_equal(x1, x1actual, decimal=4)
        np.testing.assert_array_almost_equal(x2, x2actual, decimal=4)
        np.testing.assert_array_almost_equal(x3, x3actual, decimal=4)
        np.testing.assert_array_almost_equal(x4, x4actual, decimal=4)

    def testSine(self):
        '''This might actually be a bad idea.'''
        pass

    def testDivergenceException(self):
        '''arctangent'''
        f = lambda x: np.arctan(x)
        solver = newton.Newton(f)
        self.assertRaises(newton.ErrorTooLarge, solver.solve, 2)


if __name__ == "__main__":
    unittest.main()
