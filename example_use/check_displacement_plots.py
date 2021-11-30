# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import matplotlib.pyplot as plt

from linear_elasticity_2d_solver import LinearElasticity2DProblem, get_mu_lambda
from linear_elasticity_2d_solver.default_constants import default_tol

e_mean = 160e3
nu_mean = 0.2
mu, lam = get_mu_lambda(e_mean, nu_mean)


def f(x, y):
    alpha = 8e3 * 9.81  # Newton/m^2...?
    return 0, 0


def dirichlet_bc_func(x, y):
    return 0, 0


def clamped_bc(x, y):
    return abs(x) <= default_tol


def neumannn_bc(x, y):
    val = 0.2e4
    if abs(x - 1) <= default_tol:
        return val, 0
    elif abs(y) <= default_tol:
        return 0, -val
    elif abs(y-1) >= default_tol:
        return 0, val
    else:
        return 0, 0


def main():
    n = 2
    print(n)
    le2d = LinearElasticity2DProblem.from_functions(n, f,
                                                    neumann_bc_func=neumannn_bc,
                                                    get_dirichlet_edge_func=clamped_bc)
    le2d.hfsolve(e_mean, nu_mean)
    le2d.hf_plot_displacement()
    plt.show()
    le2d.build_rb_model()
    """print(le2d.v.T @ le2d.compute_a_free(e_mean, nu_mean) @ le2d.v)
    print(le2d.v.T @ le2d.compute_a_free(e_mean, 0) @ le2d.v)
    print(le2d.v.T @ le2d.compute_a_free(10e3, nu_mean) @ le2d.v)"""
    le2d.rbsolve(e_mean, nu_mean)
    le2d.rb_plot_displacement()
    # plt.savefig("other_plots/test_plot.pdf")
    plt.show()


if __name__ == '__main__':
    main()
