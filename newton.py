# newton - Newton-Raphson solver
#
# For APC 524 Homework 3
# CWR, 18 Oct 2010

import numpy as np
import functions as F


class Newton(object):
    def __init__(self, f, tol=1.e-6, maxiter=20, dx=1.e-6, jacobian=None):
        '''Return a new object to find roots of f(x) = 0 using Newton's method.
        tol:     tolerance for iteration (iterate until |f(x)| < tol)
        maxiter: maximum number of iterations to perform
        dx:      step size for computing approximate Jacobian
        jacobian: function to return a N by N matrix of partial derivatives
                  of f, where N = len(x), num of dims of the f function'''
        if jacobian == None:
            self._jacobian = F.ApproximateJacobian  # args: self._f, x, self._dx
        else:  # TODO what a non-function is passed?
            self._jacobian = lambda f, x, dx: jacobian(x)
        self._f = f
        self._tol = tol
        self._maxiter = maxiter
        self._dx = dx

    def solve(self, x0, verbose=False):
        '''Return a root of f(x) = 0, using Newton's method, starting from
        initial guess x0'''
        x = x0
        for i in xrange(self._maxiter):
            fx = self._f(x)
            norm = np.linalg.norm(fx)
            if verbose:
                x_string = "(" + ",".join(str(a) for a in list(x)) + ")"
                fx_string = "(" + ",".join(str(a) for a in list(fx)) + ")"
                print "(i | x | fx | norm) = (", i, "|", x_string, "|", fx_string, "|", norm, ")"
            if norm < self._tol:
                return x
            x = self.step(x, fx)
        norm = np.linalg.norm(fx)
#        print "After %i iterations, norm was still %f." % (self._maxiter, norm)
        if norm < self._tol:
            return x
        else:
            raise TooManyIterations(self._maxiter, norm) # TODO Alexander had the idea to use a custom exception type here.

    def step(self, x, fx=None):
        '''Take a single step of a Newton method, starting from x
        If the argument fx is provided, assumes fx = f(x)
        x_{k+1} = x - Df(x)^{-1}*f(x)'''
        if fx is None:
            fx = self._f(x)
        Df_x = self._jacobian(self._f, x, self._dx)
        h = np.linalg.solve(np.matrix(Df_x), np.matrix(fx))
        return x - h

class TooManyIterations(Exception):

    def __init__(self, iters, norm):
        self.iters = iters
        self.norm = norm
        Exception.__init__(self, 'Maximum number of iterations exceeded, with final norm=%f'%self.norm)
    def __str__(self):
        return repr('After %i iterations, failed to converge (norm was still %f)' % (self.iters, self.norm))
