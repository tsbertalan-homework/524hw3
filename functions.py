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


def Linear():
    f = lambda x: 3.0 * x ** 2 + 4.0 * x - 9.0
    Df = lambda x: 6.0 * x + 4.0
    return {'f': f, 'Df': Df}


def SkewedSine():
    f = lambda x: 2.0 * x + np.sin(x)
    Df = lambda x: 2.0 + np.cos(x)
    return {'f': f, 'Df': Df}


def Exponential():
    subjuggulator = 10**(-8)
    f = lambda x: 4.5 * np.exp(9.2 * x) * subjuggulator
    Df = lambda x: 4.5 * 9.2 * np.exp(9.2 * x) * subjuggulator
    return {'f': f, 'Df': Df}


def Logarithmic():
    f = lambda x: 4.5 * np.log(9.2 * x)
    Df = lambda x: 4.5 / x
    return {'f': f, 'Df': Df}


def AsciiSum(input_string):
    n = 0
    for c in input_string:
        n += ord(c)
    return n


def QuadraticStrings():
    a = AsciiSum('herp')
    b = AsciiSum('derp')
    subjuggulator = 10**(-8)
    f = lambda x: a * x ** 2 * subjuggulator + b * x
    Df = lambda x: 2 * a * x * subjuggulator + b
    return {'f': f, 'Df': Df}

def Linear2D():
    def f(x):
        y = np.matrix(np.zeros((2,1)))
        y[0,0] = 5.4 * x[0,0] + 3.4 * x[1,0] + 6
        y[1,0] = 5.2 * x[0,0] + 4.2 * x[1,0] + 4
        return np.matrix(y)

    def Df(x):
        return np.matrix("5.4 3.4 ; 5.2 4.2")

    return {'f': f, 'Df': Df}

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

