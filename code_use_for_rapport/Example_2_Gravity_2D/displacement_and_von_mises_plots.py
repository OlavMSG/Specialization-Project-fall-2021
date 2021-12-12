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

alpha = 8e3 * 100 * 9.81 * 0.01  # N/m^2


# Example 2: Gravity in 2D
def f(x, y):
    return alpha, 0


def dirichlet_bc_func(x, y):
    return 0, 0


def clamped_bc(x, y):
    return abs(x) <= default_tol


def main():
    n_vec = [20, 40, 80]
    # !!! Set to True to save the plots!!!
    save = True
    levels = np.linspace(0, 65_000, 25)
    for n in n_vec:
        save_dict = r"displacement_and_von_mises_plots"
        if save:
            save_dict = check_and_make_folder(n, save_dict)
            print(save_dict)

        le2d = LinearElasticity2DProblem.from_functions(n, f, get_dirichlet_edge_func=clamped_bc)
        le2d.hfsolve(e_mean, nu_mean)
        le2d.hf_plot_displacement()
        if save:
            plt.savefig(save_dict + f"/hf_displacement_e_nu_mean.pdf")
        le2d.hf_plot_von_mises(levels=levels)
        if save:
            plt.savefig(save_dict + f"/hf_von_mises_e_nu_mean_.pdf")
        plt.show()

        le2d.build_rb_model()
        le2d.rbsolve(e_mean, nu_mean, n_rom=1)  # note n_rom = 1
        le2d.rb_plot_displacement()
        if save:
            plt.savefig(save_dict + f"/rb_displacement_e_nu_mean.pdf")
        le2d.rb_plot_von_mises(levels=levels)
        if save:
            plt.savefig(save_dict + f"/rb_von_mises_e_nu_mean_.pdf")
        plt.show()
        print(f"True n_rom={le2d.n_rom}")


if __name__ == '__main__':
    main()
