# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import matplotlib.pyplot as plt

from linear_elasticity_2d_solver import LinearElasticity2DProblem


def main():
    n_vec = [2, 20, 80]
    # !!! Set to True to save the plots!!!
    save = True
    for n in n_vec:
        le2d = LinearElasticity2DProblem.from_functions(n, 0)
        le2d.plot_mesh()
        if save:
            plt.savefig(f"mesh_plots/mesh_plot_n{n}.pdf")
        plt.show()


if __name__ == '__main__':
    main()
