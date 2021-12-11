# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import matplotlib.pyplot as plt

from linear_elasticity_2d_solver import LinearElasticity2DProblem


def f(x, y):
    return 0, 0


def dirichlet_bc_func(x, y):
    return 0, 0


def main():
    n = 20
    print(n)
    save = True
    directory_path = r"../code_not_used_in_rapport/saved_data"
    le2d = LinearElasticity2DProblem.from_functions(n, 0)
    le2d.plot_mesh()
    if save:
        plt.savefig(f"mesh_plots/mesh_plot_n{n}.pdf")
    plt.show()


if __name__ == '__main__':
    main()
