# -*- coding: utf-8 -*-
"""
@author: Olav Milian Gran
"""

import numpy as np

from linear_elasticity_2d_solver import LinearElasticity2DProblem
from linear_elasticity_2d_solver.default_constants import default_tol, e_young_range, nu_poisson_range

# rho_steal = 8e3  # kg/m^3

e_mean = np.mean(e_young_range)
nu_mean = np.mean(nu_poisson_range)


# Example 2: Gravity in 2D
def f(x, y):
    alpha = 8e3 * 9.81  # Newton/m^2...?
    return alpha, 0


def clamped_bc(x, y):
    return abs(x) <= default_tol


if __name__ == '__main__':
    import sys
    import timeit

    n = 80
    directory_path = r"saved_data"
    with open(f"time_code_logs/time_code_log_n{n}.txt", "w") as time_code_log:
        sys.stdout = time_code_log

        # Degrees of freedom info
        le2d = LinearElasticity2DProblem.from_saves(n, directory_path, print_info=False)
        le2d.build_rb_model(print_info=False)
        print("Degrees of freedom info")
        print("-" * 50)
        print(f"Nodes along one axis n: {le2d.n}")
        print(f"HF system full size n_full: {le2d.n_full}")
        print(f"Sample mode used for E_young and nu_poisson: {le2d.pod_mode}")
        print(f"Number of sample values used for E_young and nu_poisson (E, nu): {le2d.rb_grid}")
        print(f"ns for solution matrix (ns_rom): {le2d.ns_rom}")
        print(f"Solution matrix rank: {le2d.solution_matrix_rank}")
        print("-" * 50)
        print(f"HF dofs Nh (n_free): {le2d.n_free}")
        print(f"RB dofs N (n_rom): {le2d.n_rom}")
        print(f"Dofs reduction: {round(le2d.n_free / le2d.n_rom)} to 1, ({le2d.n_free / le2d.n_rom} to 1)")
        print("-" * 50)

        # Assemble HF system
        num1 = 30
        code = "le2d = LinearElasticity2DProblem.from_functions(n, f, " \
               "get_dirichlet_edge_func=clamped_bc, print_info=False)"
        time1 = timeit.timeit(code, number=num1, globals=globals())
        print("Assemble HF system:")
        print(f"total : {time1}  sec, mean time: {time1 / num1} sec, runs: {num1}")
        print("-" * 50)

        # Solve HF system
        num2 = 300
        code = "le2d.hfsolve(e_mean, nu_mean, print_info=False)"
        setup = "LinearElasticity2DProblem.from_saves(n, directory_path, print_info=False)"
        time2 = timeit.timeit(code, number=num2, globals=globals(), setup=setup)
        print("Solve HF system:")
        print(f"total : {time2} sec, mean time: {time2 / num2} sec, runs: {num2}")
        print("-" * 50)

        # Build RB system
        num3 = 30
        code = "le2d.build_rb_model(print_info=False)"
        setup = "LinearElasticity2DProblem.from_saves(n, directory_path, print_info=False)"
        time3 = timeit.timeit(code, number=num3, globals=globals(), setup=setup)
        print("Build RB model:")
        print(f"total : {time3} sec, mean time: {time3 / num3} sec, runs: {num3}")
        print("-" * 50)

        # Solve RB system
        num4 = 300
        code = "le2d.rbsolve(e_mean, nu_mean, print_info=False)"
        setup = "LinearElasticity2DProblem.from_saves(n, directory_path, print_info=False)"
        time4 = timeit.timeit(code, number=num4, globals=globals(), setup=setup)
        print("Solve RB system:")
        print(f"total : {time4} sec, mean time: {time4 / num4} sec, runs: {num4}")
        print("-" * 50)

        print(f"Offline CPU time: {time1 / num1 + time3 / num3} sec")
        print(f"Online CPU time: {time4 / num4} sec")
