# newton - Newton-Raphson solver
#
# For APC 524 Homework 3
# CWR, 18 Oct 2010

import numpy as np
import functions as F


class Newton(object):
    def __init__(self, f, tol=1.e-6, maxiter=20, dx=1.e-6, jacobian=None, thresh=413):
        '''Creates a new object to find roots of f(x) = 0 using Newton's method.

        :param tol:      tolerance for iteration (iterate until `|f(x)| < tol`)
        :param maxiter:  maximum number of iterations to perform
        :param dx:       step size for computing approximate Jacobian
        :param jacobian: function to return a N by N matrix of partial derivatives of f, where N = len(x), num of dims of the f function
        :param thresh: error norm above which the solver should declare that it is diverging.'''

        if jacobian == None:
            self._jacobian = F.ApproximateJacobian  # args: self._f, x, self._dx
        else:  # TODO what a non-function is passed?
            self._jacobian = lambda f, x, dx: jacobian(x)
        self._f = f
        self._tol = tol
        self._maxiter = maxiter
        self._dx = dx
        self._thresh = thresh

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
            errornorm = np.linalg.norm(x - x0)
            if errornorm > self._thresh:
                raise ErrorTooLarge(errornorm, self._thresh)
            if norm < self._tol:
                return x
            x = self.step(x, fx)
        norm = np.linalg.norm(self._f(x))
        if norm < self._tol:
            return x
        else:
            raise TooManyIterations(self._maxiter, errornorm)

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
    '''This exception should be raised when the (possibly user-specified)
    maximum number of iterations has been surpassed without achieving the
    (again, possibly user-specified) convergence tolerance.'''
    def __init__(self, iters, norm):
        self.iters = iters
        self.norm = norm
        Exception.__init__(self, 'After %i iterations, failed to converge (norm was still %f)' % (self.iters, self.norm))
    def __str__(self):
        return repr('After %i iterations, failed to converge (norm was still %f)' % (self.iters, self.norm))

class ErrorTooLarge(Exception):
    '''This exception should be raised when the solver is diverging, and
    and therefore the error has surpassed some threshold value (that,
    presumably, is larger than the initial error.)'''
    def __init__(self, radius, threshold):
        self.radius = radius
        self.threshold = threshold
        Exception.__init__(self, 'Error radius of %i outside of threshold radius %i.' % (self.radius, self.threshold))
    def __str__(self):
        return repr('Error radius of %i outside of threshold radius %i.' % (self.radius, self.threshold))
