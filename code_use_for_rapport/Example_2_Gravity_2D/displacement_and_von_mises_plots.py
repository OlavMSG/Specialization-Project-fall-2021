# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import matplotlib.pyplot as plt
import numpy as np

from linear_elasticity_2d_solver import LinearElasticity2DProblem
from linear_elasticity_2d_solver.default_constants import default_tol, e_young_range, nu_poisson_range
from linear_elasticity_2d_solver.helpers import check_and_make_folder

e_mean = np.mean(e_young_range)
nu_mean = np.mean(nu_poisson_range)


# Example 2: Gravity in 2D
def f(x, y):
    alpha = 8e3 * 9.81  # Newton/m^2...?
    return alpha, 0


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
    elif abs(y - 1) >= default_tol:
        return 0, val
    else:
        return 0, 0


def main():
    n = 40
    save = True
    print(n)
    save_dict = r"displacement_and_von_mises_plots"
    save_dict = check_and_make_folder(n, save_dict)
    levels = np.linspace(0, 60_000, 25)

    le2d = LinearElasticity2DProblem.from_functions(n, f, get_dirichlet_edge_func=clamped_bc)
    le2d.hfsolve(e_mean, nu_mean)
    le2d.hf_plot_displacement()
    if save:
        plt.savefig(save_dict + f"hf_displacement_e_nu_mean_n{n}.pdf")
    le2d.hf_plot_von_mises(levels=levels)
    if save:
        plt.savefig(save_dict + f"hf_von_mises_e_nu_mean_n{n}.pdf")
    plt.show()

    le2d.build_rb_model()
    le2d.rbsolve(e_mean, nu_mean, n_rom=1)  # note n_rom = 1
    le2d.rb_plot_displacement()
    if save:
        plt.savefig(save_dict + f"rb_displacement_e_nu_mean_n{n}.pdf")
    le2d.rb_plot_von_mises(levels=levels)
    if save:
        plt.savefig(save_dict + f"rb_von_mises_e_nu_mean_n{n}.pdf")
    plt.show()
    print(le2d.n_rom)


if __name__ == '__main__':
    main()
