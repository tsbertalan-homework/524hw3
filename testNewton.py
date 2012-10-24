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
        '''Tests newton.step() with a single-variable linear function.'''
        pass

    def testStepNVar(self):
        '''Tests newton.step() with linear functions of multiple variables.'''
        pass

    def testFunctionKwarg(self):
        '''Tests newton.step() with a single-variable linear function,
        and also makes use of step()'s fx=??? keyword argment.'''
        pass

    def testQuadratic(self):
        '''Tests newton.solve() with a quadratic function.'''
        pass

    def testSine(self):
        '''This might actually be a bad idea.'''
        pass

if __name__ == "__main__":
    unittest.main()
