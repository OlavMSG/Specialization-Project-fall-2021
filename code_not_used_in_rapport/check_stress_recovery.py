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
    # alpha = 5e3  # Newton/m^2...?
    return 0, -mu - lam


def dirichlet_bc_func(x, y):
    return x * y, 0


def main():
    n = 5
    print(n)
    le2d = LinearElasticity2DProblem.from_functions(n, f, dirichlet_bc_func=dirichlet_bc_func)
    le2d.build_rb_model()
    le2d.hfsolve(e_mean, nu_mean)
    le2d.hf_von_mises_stress()
    le2d.rbsolve(e_mean, nu_mean)
    le2d.rb_von_mises_stress()

    le2d.hf_plot_von_mises()
    plt.show()
    le2d.rb_plot_von_mises()
    plt.show()


if __name__ == '__main__':
    main()
