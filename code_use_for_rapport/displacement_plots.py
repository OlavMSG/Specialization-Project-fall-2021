# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import matplotlib.pyplot as plt
import numpy as np

from linear_elasticity_2d_solver import LinearElasticity2DProblem
from linear_elasticity_2d_solver.default_constants import default_tol
from linear_elasticity_2d_solver.helpers import get_mu_lambda

e_mean = 160e3
nu_mean = 0.2
mu, lam = get_mu_lambda(e_mean, nu_mean)


def f(x, y):
    alpha = 8e3 * 9.81  # Newton/m^2...?
    return alpha*np.sin(np.pi*y), 0


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
    n = 3
    save = False
    print(n)

    le2d = LinearElasticity2DProblem.from_functions(n, f)
                                                    #get_dirichlet_edge_func=clamped_bc)
    le2d.hfsolve(e_mean, nu_mean)
    le2d.hf_plot_displacement()
    le2d.hf_plot_von_mises()
    if save:
        plt.savefig(f"other_plots/hf_displacement_e_nu_mean_n{n}.pdf")

    plt.show()
    # le2d.build_rb_model()
    """print(le2d.v.T @ le2d.compute_a_free(e_mean, nu_mean) @ le2d.v)
    print(le2d.v.T @ le2d.compute_a_free(e_mean, 0) @ le2d.v)
    print(le2d.v.T @ le2d.compute_a_free(10e3, nu_mean) @ le2d.v)"""
    # le2d.rbsolve(e_mean, nu_mean)
    # le2d.rb_plot_displacement()
    if save:
        plt.savefig(f"other_plots/rb_displacement_e_nu_mean_n{n}.pdf")
    # plt.show()
    # le2d.plot_pod_singular_values()
    # plt.show()


if __name__ == '__main__':
    main()
