# -*- coding: utf-8 -*-
"""
@author: Olav Milian Gran
"""

from linear_elasticity_2d_solver import LinearElasticity2DProblem

rho_steal = 8e3  # kg/m^3


# Here just example used
def f(x, y):
    alpha = 5e3  # Newton/m^2...?
    return -alpha, 0


def dirichlet_bc_func(x, y):
    return 0, 0


def main():
    n_vec = [3, 4, 5, 10, 20, 30, 40, 50]
    for n in n_vec:
        print(n)
        e_mean = 160e3
        nu_mean = 0.2
        directory_path = r"saved_data"
        le2d = LinearElasticity2DProblem.from_functions(n, f)
        le2d.build_rb_model()
        le2d.hfsolve(e_mean, nu_mean)
        le2d.rbsolve(e_mean, nu_mean)
        le2d.save(directory_path)
        # le2d.solve(_n, e_young, nu_poisson)
        # e_max = 3.1e5
        # nu_max = 4e-1
        # print(le2d.error_a_rb(e_max, nu_max))
        le2d = LinearElasticity2DProblem.from_saves(n, directory_path)


if __name__ == '__main__':
    main()
