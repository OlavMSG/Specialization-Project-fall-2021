# -*- coding: utf-8 -*-
"""
Created on 04.10.2021

@author: Olav Milian Gran
"""
import numpy as np
from linear_elasticity_2d_solver import TOL_EDGE


# Here just example
def get_dirichlet_and_neumann_edge(p, edge):
    # know _edge = np.vstack((south_edge, east_edge, north_edge, west_edge))
    south_edge, east_edge, north_edge, west_edge = np.split(edge, 4)
    dirichlet_edge = west_edge
    neumann_edge = np.vstack((south_edge, east_edge, north_edge))
    return dirichlet_edge, neumann_edge


# Here just example used
def f(x, y):
    return -2 * (x * x + 4 * x * y + 3 * y * y - 4), -2 * (y * y + 4 * x * y + 3 * x * x - 4)


# Here just example
def neumann_bc_func(x, y):
    # sigma(u) @ normal_vec = _neumann_bc_func_non_vec
    if abs(y + 1) < TOL_EDGE:  # south _edge, y = -1
        return 1, 2
    elif abs(x - 1) < TOL_EDGE:  # east _edge, x = 1
        return 1, 2
    elif abs(y - 1) < TOL_EDGE:  # north _edge, y = 1
        return 1, 2
    else:
        # should not come here, but if we do return 0
        return 0, 0


# Here just example
def dirichlet_bc_func(x, y, d):
    if abs(x + 1) < TOL_EDGE:  # west _edge, x = -1
        return 1, 2
    else:
        # should not come here, but if we do return 0
        return 0, 0


# Here just example used
def u_exact(x, y):
    return (x * x - 1) * (y * y - 1)


def main():
    """n = 10
    print(n)
    e_young, nu_poisson = 2.1e5, 0.3
    directory_path = r"reduced_order_error_check_plots"
    le2d = LinearElasticity2DProblem.from_functions(f)
    le2d._hf_save(n, directory_path)
    le2d = LinearElasticity2DProblem.from_saves(n, directory_path)
    le2d.solve(e_young, nu_poisson)"""
    from time import perf_counter
    A = np.array([[1, 4], [2, 5], [3, 6], [10, 11], [33, 21]])
    B = np.array([[1, 4], [3, 6], [7, 8]])

    s = perf_counter()
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [A.dtype]}

    C = np.setdiff1d(A.view(dtype), B.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    print(perf_counter() - s)
    print(C)
    s = perf_counter()
    res = np.array(list(set(map(tuple, A)) - set(map(tuple, B))))
    res = res[np.argsort(res[:, 0]), :]
    print(perf_counter() - s)
    print(res)


if __name__ == '__main__':
    main()
