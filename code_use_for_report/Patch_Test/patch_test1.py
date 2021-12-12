# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import sys

import numpy as np

from linear_elasticity_2d_solver import LinearElasticity2DProblem
from linear_elasticity_2d_solver.default_constants import e_young_range, nu_poisson_range
from linear_elasticity_2d_solver.helpers import get_u_exact, get_lambda_mu


def base(dirichlet_bc_func, u_exact_func, f=None, print_l2=False, print_free_node_values=False):
    if f is None:
        f = 0
    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)

    le2d = LinearElasticity2DProblem.from_functions(n, f, dirichlet_bc_func=dirichlet_bc_func)

    le2d.solve(e_mean, nu_mean, print_info=False)
    u_exact = get_u_exact(le2d.p, u_exact_func)
    if print_l2:
        from matplotlib.tri import LinearTriInterpolator, Triangulation
        # linearly interpolate uh on the triangulation
        tri = Triangulation(le2d.x, le2d.y)
        uh_x = LinearTriInterpolator(tri, le2d.uh.x)
        uh_y = LinearTriInterpolator(tri, le2d.uh.y)

        def err2(x):
            x, y = x
            u_ex = u_exact_func(x, y)
            return (u_ex[0] - uh_x(x, y)) ** 2 + (u_ex[1] - uh_y(x, y)) ** 2

        from quadpy import c2
        sq = np.array([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]])
        scheme = c2.get_good_scheme(20)
        l2norm = np.sqrt(scheme.integrate(err2, sq))
        print("L2 norm {}".format(l2norm))

    # discrete max norm, holds if u_exact is linear (Terms 1, x, y)
    test_res = np.all(np.abs(le2d.uh_full - u_exact.flatt_values) < tol)
    print("max norm {}".format(np.max(np.abs(le2d.uh_full - u_exact.flatt_values))))
    print("tolerance {}".format(tol))
    print("plate limits {}".format(le2d.plate_limits))
    print("test {} for n_rom={}".format(test_res, n))

    if print_free_node_values:
        print("Free node values (from [x1, y1, x2, y2, ...]):")
        print("uh: ", le2d.uh_free)
        print("u_ex: ", u_exact.flatt_values[le2d.free_index])

    print("-" * 10)

    assert test_res


def case_1(print_l2=False, print_free_node_values=False):
    print("Case 1: (x, 0)")

    def u_exact_func(x, y):
        return x, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(dirichlet_bc_func, u_exact_func, print_l2=print_l2, print_free_node_values=print_free_node_values)


def case_2(print_l2=False, print_free_node_values=False):
    print("Case 2: (0, y)")

    def u_exact_func(x, y):
        return 0., y

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(dirichlet_bc_func, u_exact_func, print_l2=print_l2, print_free_node_values=print_free_node_values)


def case_3(print_l2=False, print_free_node_values=False):
    print("Case 3: (y, 0)")

    def u_exact_func(x, y):
        return y, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(dirichlet_bc_func, u_exact_func, print_l2=print_l2, print_free_node_values=print_free_node_values)


def case_4(print_l2=False, print_free_node_values=False):
    print("Case 4: (0, x)")

    def u_exact_func(x, y):
        return 0., x

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    base(dirichlet_bc_func, u_exact_func, print_l2=print_l2, print_free_node_values=print_free_node_values)


# passes, but test does not hold since u_exact is not linear, see l2 norm
def case_5(print_l2=False, print_free_node_values=False):
    print("Case 5: (xy, 0)")
    print("Passes, but test does not hold since u_exact is not linear, see l2 norm")

    def u_exact_func(x, y):
        return x * y, 0.

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)
    mu, lam = get_lambda_mu(e_mean, nu_mean)

    def f(x, y):
        return 0., - mu - lam

    base(dirichlet_bc_func, u_exact_func, f, print_l2=print_l2, print_free_node_values=print_free_node_values)


# passes, but test does not hold since u_exact is not linear, see l2 norm
def case_6(print_l2=False, print_free_node_values=False):
    print("Case 6: (0, xy)")
    print("Passes, but test does not hold since u_exact is not linear, see l2 norm")

    def u_exact_func(x, y):
        return 0., x * y

    def dirichlet_bc_func(x, y):
        return u_exact_func(x, y)

    e_mean = np.mean(e_young_range)
    nu_mean = np.mean(nu_poisson_range)
    mu, lam = get_lambda_mu(e_mean, nu_mean)

    def f(x, y):
        return - mu - lam, 0.

    base(dirichlet_bc_func, u_exact_func, f, print_l2=print_l2, print_free_node_values=print_free_node_values)


def main():
    n = 2
    tol = 1e-14
    print_l2 = True
    print_free_node_values = True
    output_file = r"patch_test1_run_log.txt"
    with open(output_file, "w") as time_code_log:
        sys.stdout = time_code_log
        case_1(print_l2, print_free_node_values)
        case_2(print_l2, print_free_node_values)
        case_3(print_l2, print_free_node_values)
        case_4(print_l2, print_free_node_values)
        case_5(print_l2, print_free_node_values)
        case_6(print_l2, print_free_node_values)


if __name__ == '__main__':
    main()
