# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import matplotlib.pyplot as plt

from linear_elasticity_2d_solver import LinearElasticity2DProblem
from linear_elasticity_2d_solver.helpers import get_mu_lambda

e_mean = 160e3
nu_mean = 0.2
mu, lam = get_mu_lambda(e_mean, nu_mean)

def f(x, y):
    # alpha = 5e3  # Newton/m^2...?
    return 0, -mu -lam


def dirichlet_bc_func(x, y):
    return x*y, 0


def main():
    n = 3
    print(n)
    e_mean = 160e3
    nu_mean = 0.2
    le2d = LinearElasticity2DProblem.from_functions(n, f, dirichlet_bc_func=dirichlet_bc_func)
    le2d.build_rb_model()
    le2d.hfsolve(e_mean, nu_mean)
    le2d.hf_von_mises_yield()
    le2d.rbsolve(e_mean, nu_mean)
    le2d.rb_von_mises_yield()

    le2d.hf_plot_von_mises()
    plt.show()
    le2d.rb_plot_von_mises()
    plt.show()

if __name__ == '__main__':
    main()
