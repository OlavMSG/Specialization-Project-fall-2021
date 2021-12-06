# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin, cos

from linear_elasticity_2d_solver import LinearElasticity2DProblem
from linear_elasticity_2d_solver.default_constants import nu_poisson_range, e_young_range
from linear_elasticity_2d_solver.helpers import get_mu_lambda, get_u_exact

e_mean = np.mean(e_young_range)
nu_mean = np.mean(nu_poisson_range)
mu, lam = get_mu_lambda(e_mean, nu_mean)


def u_exact_i(x, y):
    return sin(pi * x) * sin(pi * y)


def u_exact_func(x, y):
    return u_exact_i(x, y), u_exact_i(x, y)


def f(x, y):
    ret = -lam * (
            -pi ** 2 * sin(pi * x) * sin(pi * y) + pi ** 2 * cos(pi * x) * cos(pi * y)) + 2.0 * mu * pi ** 2 * sin(
        pi * x) * sin(pi * y) \
          - 2 * mu * (-0.5 * pi ** 2 * sin(pi * x) * sin(pi * y) + 0.5 * pi ** 2 * cos(pi * x) * cos(pi * y))
    return ret, ret


def line(x, order, init_error):
    x = x.astype(float)
    return init_error * x ** (-order) / (x[0]) ** (-order)


def main(mode="energy norm"):
    print(mode)
    n_vec = np.array([2, 4, 8, 16])
    # triangle elements have to sides a = b = 1/n this gives c = sqrt(2)/n
    h_vec = np.sqrt(2) / n_vec
    norm_err_vec = np.zeros_like(n_vec, dtype=float)

    if mode == "L2 norm":
        # compute the exact solution in the sample points
        def u_exact_func_xy(x):
            x, y = x
            return 2 * u_exact_i(x, y) ** 2
        from quadpy import c2
        sq = np.array([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]])
        scheme = c2.get_good_scheme(20)
        norm2_u_exact = scheme.integrate(u_exact_func_xy, sq)
        norm_u_exact = np.sqrt(norm2_u_exact)
    else:
        # set the exact solution as the solution on the grid n = max(n_vec) * 4
        le2d = LinearElasticity2DProblem.from_functions(np.max(n_vec) * 4, f, dirichlet_bc_func=u_exact_func)
        u_exact = get_u_exact(le2d.p, u_exact_func)
        norm_u_exact = np.sqrt(u_exact.T @ le2d.compute_a_full(e_mean, nu_mean) @ u_exact)

    # get exact solution and compute the hf error
    for i, n in enumerate(n_vec):
        # set up and solve system
        le2d = LinearElasticity2DProblem.from_functions(n, f, dirichlet_bc_func=u_exact_func)
        le2d.hfsolve(e_mean, nu_mean)

        if mode == "L2 norm":
            from matplotlib.tri import LinearTriInterpolator, Triangulation
            # linearly interpolate uh on the triangulation
            tri = Triangulation(le2d.x, le2d.y)
            uh_x = LinearTriInterpolator(tri, le2d.uh.x)
            uh_y = LinearTriInterpolator(tri, le2d.uh.y)

            def err2(x):
                x, y = x
                return (u_exact_i(x, y) - uh_x(x, y)) ** 2 + (u_exact_i(x, y) - uh_y(x, y)) ** 2

            from quadpy import c2
            sq = np.array([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]])
            scheme = c2.get_good_scheme(20)
            norm2_u_exact = scheme.integrate(err2, sq)
            norm_err_vec[i] = np.sqrt(norm2_u_exact)
        else:
            u_exact = get_u_exact(le2d.p, u_exact_func)
            err = u_exact - le2d.uh_full
            # compute the norm
            norm_err_vec[i] = np.sqrt(err.T @ le2d.compute_a_full(e_mean, nu_mean) @ err)

    le2d = LinearElasticity2DProblem.from_functions(n_vec[-1], f, dirichlet_bc_func=u_exact_func)
    le2d.hfsolve(e_mean, nu_mean)
    le2d.plot_displacement()
    plt.show()

    # compute the relative norm
    rel_err_vec = norm_err_vec / norm_u_exact

    # plot convergence
    plt.figure(f"Convergence in the {mode}")
    plt.title(f"Convergence in the {mode}")
    p = -np.polyfit(np.log(n_vec), np.log(rel_err_vec), deg=1)[0].astype(float)
    print(p)
    plt.plot(n_vec, rel_err_vec, "D-", label=f"Relative error, \n slope $\\approx {p:.2f}$")
    plt.plot(n_vec, line(n_vec, 2, rel_err_vec[0]), "--", label="Slope 2")

    plt.xscale("log", base=2)
    plt.yscale("log", base=10)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    mode = "energy norm"
    main(mode)
