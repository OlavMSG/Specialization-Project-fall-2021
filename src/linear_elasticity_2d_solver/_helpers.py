# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np


def index_map(i, d):
    return 2 * i + d


def inv_index_map(k):
    return k // 2, k % 2


def expand_index(index):
    m = index.shape[0] * 2
    expanded_index = np.zeros(m, dtype=int)
    expanded_index[np.arange(0, m, 2)] = index_map(index, 0)
    expanded_index[np.arange(1, m, 2)] = index_map(index, 1)
    return expanded_index


def get_mu_lambda(e_young, nu_poisson):
    nu_p1 = nu_poisson + 1
    lam = e_young * nu_poisson / (nu_p1 * (1 - 2 * nu_poisson))
    mu = 0.5 * e_young / nu_p1
    return mu, lam


def get_e_young_nu_poisson(mu, lam):
    lam_p_mu = lam + mu
    nu_poisson = 0.5 * lam / lam_p_mu
    e_young = mu * (3 * lam + 2 * mu) / lam_p_mu
    return e_young, nu_poisson


def compute_a(e_young, nu_poisson, a1, a2):
    mu, lam = get_mu_lambda(e_young, nu_poisson)
    return 2 * mu * a1 + lam * a2


def get_u_exact(p, u_exact_func):
    x_vec = p[:, 0]
    y_vec = p[:, 1]
    u_exact = FunctionValues2D.from_nx2(VectorizedFunction2D(u_exact_func)(x_vec, y_vec))
    return u_exact.flatt_values


class VectorizedFunction2D:

    def __init__(self, func_non_vec):

        def vectorize_func_2d(x_vec, y_vec):
            if isinstance(x_vec, (float, int)):
                x_vec = np.array([x_vec])
            if isinstance(y_vec, (float, int)):
                y_vec = np.array([y_vec])
            x_vals = np.zeros_like(x_vec, dtype=float)
            y_vals = np.zeros_like(x_vec, dtype=float)
            for i, (x, y) in enumerate(zip(x_vec, y_vec)):
                x_vals[i], y_vals[i] = func_non_vec(x, y)
            return np.column_stack((x_vals, y_vals))

        self._func_vec = vectorize_func_2d

    def __call__(self, x_vec, y_vec):
        return self._func_vec(x_vec, y_vec)


class FunctionValues2D:

    def __init__(self):
        self._values = None
        self._n = None

    def __repr__(self):
        return self._values.__repr__()

    def __str__(self):
        return self._values.__str__()

    def _set_from_nx2(self, values):
        self._values = np.asarray(values, dtype=float)
        self._n = self._values.shape[0]

    def _set_from_1xn2(self, values):
        m = values.shape[0]
        if m % 2 != 0:
            raise ValueError("Shape of values must be (1, 2k), where k is an integer")
        self._n = m // 2
        self._values = np.zeros((self.n, 2))
        self._values[:, 0] = values[np.arange(0, m, 2)]
        self._values[:, 1] = values[np.arange(1, m, 2)]

    @classmethod
    def from_nx2(cls, values):
        out = cls()
        out._set_from_nx2(values)
        return out

    @classmethod
    def from_1xn2(cls, values):
        out = cls()
        out._set_from_1xn2(values)
        return out

    @property
    def values(self):
        return self._values

    @property
    def x(self):
        return self._values[:, 0]

    @property
    def y(self):
        return self._values[:, 1]

    @property
    def flatt_values(self):
        # return [x0, y0, x1, y1, ...]
        return self._values.reshape((1, self.n * 2)).ravel()

    @property
    def dim(self):
        return 2

    @property
    def n(self):
        return self._n

    @property
    def shape(self):
        return self._values.shape


if __name__ == "__main__":
    def f(x, y):
        xval, yval = x * y, x - x
        return xval, yval


    x = np.array([i for i in range(4)])
    y = np.array([i + 4 for i in range(4)])
    print(x, y)
    print(f(x, y))

    f_vec = VectorizedFunction2D(f)
    print(f_vec)

    values = np.arange(4).reshape((2, 2))
    print(values)

    val = FunctionValues2D.from_nx2(values)
    print(val)
    print(val.flatt_values)

    val2 = FunctionValues2D.from_1xn2(val.flatt_values)
    print(val2)
    print(val2.flatt_values)
