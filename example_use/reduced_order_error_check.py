# -*- coding: utf-8 -*-
"""
Created on 15.11.2021

@author: Olav Milian Gran
"""
import numpy as np
import matplotlib.pyplot as plt

from linear_elasticity_2d_solver import LinearElasticity2DProblem

rho_steal = 8e3  # kg/m^3


# Here just example used
def f(x, y):
    alpha = 5e3  # Newton/m^2...?
    return -alpha, 0


def dirichlet_bc_func(x, y):
    return 0, 0


def main():
    n = 5
    save = False
    save_dict = r"reduced_order_error_check_plots"
    for mode in ("uniform", "gauss lobatto"):
        for grid in (5, 11):
            print("-" * 20)
            print(mode, grid)
            # define problem
            le2d = LinearElasticity2DProblem.from_functions(n, f, dirichlet_bc_func=dirichlet_bc_func)
            le2d.build_rb_model(grid=grid, mode=mode)
            le2d.plot_pod_singular_values()
            if save:
                plt.savefig(save_dict + f"/singular_values_mode_{mode}_grid_{grid}_n{n}.pdf")
            le2d.plot_pod_relative_information_content()
            if save:
                plt.savefig(save_dict + f"/relative_information_content_mode_{mode}_grid_{grid}_n{n}.pdf")
            print(f"Chosen n_rom={le2d.n_rom}, max use is n_rom_max={le2d.n_rom_max}, "
                  f"grid size is ns_rom={le2d.ns_rom}, Number of node on one axis is n={n}")
            print("Singular values:")
            print(le2d.singular_values_pod)
            max_err = np.zeros(le2d.n_rom_max)
            mean_err = np.zeros(le2d.n_rom_max)
            for n_rom in range(1, le2d.n_rom_max + 1):
                errs = np.zeros(le2d.ns_rom)
                for i, (e_young, nu_poisson) in enumerate(le2d.e_young_nu_poisson_mat):
                    # print(i, e_young, nu_poisson)
                    errs[i] = le2d.error_a_rb_set_n_rom(n_rom, e_young, nu_poisson)
                max_err[n_rom - 1] = np.max(errs)
                mean_err[n_rom - 1] = np.mean(errs)

            plt.figure("relative_information_content_{mode}_{grid}", figsize=(12, 7))
            plt.title("Reduced order errors v. $n_{rom}$")
            plt.semilogy(np.arange(1, le2d.n_rom_max + 1), mean_err, "cx--", label="$mean$")
            plt.semilogy(np.arange(1, le2d.n_rom_max + 1), max_err, "mx-", label="$max$")
            plt.grid()
            plt.legend()
            if save:
                plt.savefig(save_dict + f"/reduced_order_errors_mode_{mode}_grid_{grid}_n{n}.pdf")
            plt.show()


if __name__ == '__main__':
    main()
