# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import matplotlib.pyplot as plt

from linear_elasticity_2d_solver import LinearElasticity2DProblem


def f(x, y):
    alpha = 5e3  # Newton/m^2...?
    return -alpha, 0


def dirichlet_bc_func(x, y):
    return 0, 0


def main():
    n = 20
    print(n)
    save = True
    e_mean = 160e3
    nu_mean = 0.2
    le2d = LinearElasticity2DProblem.from_functions(n, f, dirichlet_bc_func=dirichlet_bc_func)
    le2d.plot_mesh()
    if save:
        plt.savefig(f"other_plots/mesh_plot_n{n}.pdf")
    plt.show()


if __name__ == '__main__':
    main()
