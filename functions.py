import numpy as np


def ApproximateJacobian(f, x, dx=1e-6):
    """Return an approximation of the Jacobian Df(x) as a numpy matrix"""
    try:
        n = len(x)
    except TypeError:
        n = 1
    fx = f(x)
    Df_x = np.matrix(np.zeros((n, n)))
    for i in range(n):
        v = np.matrix(np.zeros((n, 1)))
        v[i, 0] = dx
        Df_x[:, i] = (f(x + v) - fx) / dx
    return Df_x


def TestJacobian(f, Df, testcases, dx=1e-6, decimal=6):
    '''Check an analytical Jacobian function Df against the numerical
    approximation produced from the original function f. Check for each of the
    testcases (N-by-1 matrices) in the list testcases.'''
    for x in testcases:
        analytical = Df(x)
        numerical  = ApproximateJacobian(f, x, dx=dx)
        np.testing.assert_array_almost_equal(analytical, numerical, decimal=decimal)


class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 2x + 3,
    and evaluate p(5):

    p = Polynomial([1, 2, 3])
    p(5)"""

    def __init__(self, coeffs):
        self._coeffs = coeffs

    def __repr__(self):
        return "Polynomial(%s)" % (", ".join([str(x) for x in self._coeffs]))

    def f(self, x):
        ans = self._coeffs[0]
        for c in self._coeffs[1:]:
            ans = x * ans + c
        return ans

    def __call__(self, x):
        return self.f(x)

