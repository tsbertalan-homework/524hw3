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

if __name__ == "__main__":
    unittest.main()
