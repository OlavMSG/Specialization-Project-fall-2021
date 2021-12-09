# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import numpy as np

from linear_elasticity_2d_solver import LinearElasticity2DProblem
from linear_elasticity_2d_solver.default_constants import e_young_range, nu_poisson_range
from linear_elasticity_2d_solver.helpers import get_u_exact


def base(dirichlet_bc_func, u_exact_func, f=None):
    if f is None:
        f = 0
    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)

    le2d = LinearElasticity2DProblem.from_functions(n, f, dirichlet_bc_func=dirichlet_bc_func)

    le2d.solve(e_mean, nu_mean, print_info=False)
    u_exact = get_u_exact(le2d.p, u_exact_func)

    # discrete max norm, holds if u_exact is linear (Terms 1, x, y)
    test_res = np.all(np.abs(le2d.uh_full - u_exact.flatt_values) < tol)

    assert test_res


def test_case_1():
    print("Case 1: (x, 0)")

    def u_exact_func(x, y):
        return x, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(dirichlet_bc_func, u_exact_func)


def test_case_2():
    print("Case 2: (0, y)")

    def u_exact_func(x, y):
        return 0., y

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(dirichlet_bc_func, u_exact_func)


def test_case_3():
    print("Case 3: (y, 0)")

    def u_exact_func(x, y):
        return y, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(dirichlet_bc_func, u_exact_func)


def test_case_4():
    print("Case 4: (0, x)")

    def u_exact_func(x, y):
        return 0., x

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(dirichlet_bc_func, u_exact_func)


def main():
    # make_and_save_matrices()
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()


n = 2
tol = 1e-14
if __name__ == '__main__':
    main()
