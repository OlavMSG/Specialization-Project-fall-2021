# -*- coding: utf-8 -*-
"""
@author: Olav Milian Gran
"""

from linear_elasticity_2d_solver import LinearElasticity2DProblem
from linear_elasticity_2d_solver.default_constants import default_tol


# Here just example
def get_dirichlet_edge(x, y):
    return abs(x) < default_tol


# Here just example used
def f(x, y):
    return -2 * (x * x + 4 * x * y + 3 * y * y - 4), -2 * (y * y + 4 * x * y + 3 * x * x - 4)


# Here just example
def neumann_bc_func(x, y):
    # sigma(u) @ normal_vec = _neumann_bc_func_non_vec
    if abs(y) <= default_tol:  # south _edge, y = 0
        return 1, 2
    elif abs(x - 1) <= default_tol:  # east _edge, x = 1
        return 1, 2
    elif abs(y - 1) <= default_tol:  # north _edge, y = 1
        return 1, 2
    else:
        # should not come here, but if we do return 0
        return 0, 0


# Here just example
def dirichlet_bc_func(x, y):
    if abs(x) <= default_tol:  # west _edge, x = 0
        return 1, 2
    else:
        # should not come here, but if we do return 0
        return 0, 0


def clamped_bc(x, y):
    return abs(x) <= default_tol


def main():
    n = 10
    print(n)
    e_young, nu_poisson = 2.1e5, 0.3
    le2d = LinearElasticity2DProblem.from_functions(n, f, get_dirichlet_edge_func=clamped_bc,
                                                    dirichlet_bc_func=dirichlet_bc_func,
                                                    neumann_bc_func=neumann_bc_func)
    le2d.hfsolve(e_young, nu_poisson)
    le2d.build_rb_model()
    le2d.rbsolve(e_young, nu_poisson)
    print(le2d.n_rom)


if __name__ == '__main__':
    main()
