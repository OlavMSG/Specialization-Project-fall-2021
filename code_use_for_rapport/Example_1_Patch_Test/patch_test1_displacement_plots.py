# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import matplotlib.pyplot as plt
import numpy as np

from linear_elasticity_2d_solver import LinearElasticity2DProblem
from linear_elasticity_2d_solver.default_constants import nu_poisson_range, e_young_range
from linear_elasticity_2d_solver.helpers import get_mu_lambda

e_mean = np.mean(e_young_range)
nu_mean = np.mean(nu_poisson_range)
mu, lam = get_mu_lambda(e_mean, nu_mean)

def f(x, y):
    return 0, 0


def u_exact_func1(x, y):
    return x, 0.


def u_exact_func3(x, y):
    return y, 0.


def main():
    n = 2
    save = True
    print(n)
    le2d = LinearElasticity2DProblem.from_functions(n, f, dirichlet_bc_func=u_exact_func1)

    le2d.hfsolve(e_mean, nu_mean)
    le2d.hf_plot_displacement()

    if save:
        plt.savefig(f"patch_plots/patch_displacement_e_nu_mean_n{n}_case_x.pdf")
    plt.show()

    le2d = LinearElasticity2DProblem.from_functions(n, f, dirichlet_bc_func=u_exact_func3)

    le2d.hfsolve(e_mean, nu_mean)
    le2d.hf_plot_displacement()

    if save:
        plt.savefig(f"patch_plots/patch_displacement_e_nu_mean_n{n}_case_y.pdf")
    plt.show()


if __name__ == '__main__':
    main()
