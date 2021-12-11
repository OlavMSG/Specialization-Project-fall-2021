# -*- coding: utf-8 -*-
"""
@author: Olav Milian Gran
"""

from linear_elasticity_2d_solver import LinearElasticity2DProblem
from linear_elasticity_2d_solver.default_constants import default_tol

# rho_steal = 8e3  # kg/m^3


# Example 2: Gravity in 2D
def f(x, y):
    alpha = 8e3 * 9.81  # Newton/m^2...?
    return alpha, 0


def clamped_bc(x, y):
    return abs(x) <= default_tol


def main():
    n_vec = [2, 3, 4, 5, 10, 20, 40, 80]
    for n in n_vec:
        print(n)
        # e_mean = np.mean(e_young_range)
        # nu_mean = np.mean(nu_poisson_range)
        directory_path = r"saved_data"
        le2d = LinearElasticity2DProblem.from_functions(n, f, get_dirichlet_edge_func=clamped_bc)
        le2d.build_rb_model()
        # le2d.hfsolve(e_mean, nu_mean)
        # le2d.rbsolve(e_mean, nu_mean)
        le2d.save(directory_path)
        # le2d.save(directory_path)
        # le2d.solve(n, e_young, nu_poisson)
        """e_max = 3.1e5
        nu_max = 4e-1
        print(le2d.error_a_rb(e_max, nu_max, print_info=True))
        print(le2d.error_a_rb(e_max, nu_max, print_info=True))"""
        le2d = LinearElasticity2DProblem.from_saves(n, directory_path)
        # le2d.hfsolve(e_mean, nu_mean)
        # le2d.rbsolve(e_mean, nu_mean)


if __name__ == '__main__':
    main()
