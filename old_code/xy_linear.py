# -*- coding: utf-8 -*-
"""
Created on 06.11.2021

@author: Olav Milian Gran
"""

import numpy as np
from linear_elasticity_2d_class_v1 import LinearElasticity2DProblem
from linerar_elasticity_2d_helpers import get_u_exact, get_mu_lambda, get_e_young_nu_poisson
import matplotlib.pyplot as plt
n = 3


def plot(le2d, u_exact_func):
    u_exact = u_exact_func(le2d.x, le2d.y, 0)
    # the absolute error

    # restrict color coding to (-1, 1),
    plt.figure(figsize=(7, 7))
    # Create plot of numerical solution
    plt.subplot(1, 1, 1)
    plt.gca().set_aspect('equal')
    plt.tricontourf(le2d.x, le2d.y, le2d.tri, u_exact, extend='both')
    plt.triplot(le2d.x, le2d.y, le2d.tri)
    plt.colorbar()
    plt.show()

    # restrict color coding to (-1, 1),
    plt.figure(figsize=(7, 7))
    # Create plot of numerical solution
    plt.subplot(1, 1, 1)
    plt.gca().set_aspect('equal')
    plt.tricontourf(le2d.x, le2d.y, le2d.tri, np.abs(le2d.uh_x - u_exact), extend='both')
    plt.triplot(le2d.x, le2d.y, le2d.tri)
    plt.colorbar()
    plt.show()

    # restrict color coding to (-1, 1),
    plt.figure(figsize=(7, 7))
    # Create plot of numerical solution
    plt.subplot(1, 1, 1)
    plt.gca().set_aspect('equal')
    plt.tricontourf(le2d.x, le2d.y, le2d.tri, np.abs(le2d.uh_y), extend='both')
    plt.triplot(le2d.x, le2d.y, le2d.tri)
    plt.colorbar()
    plt.show()

def base(dirichlet_bc_func, u_exact_func):
    e_mean = 160e3
    nu_mean = 0.2
    mu, lam = get_mu_lambda(e_mean, nu_mean)
    print(mu, lam)
    print(get_e_young_nu_poisson(mu, lam))
    print((e_mean, nu_mean))


    def f(x, y):
        return 0, - mu - lam

    print((mu + lam) / 1e5)

    le2d = LinearElasticity2DProblem.from_functions(f, plate_limits=(0, 1), dirichlet_bc_func=dirichlet_bc_func)
    uh = le2d.hfsolve(e_mean, nu_mean, n=n)
    uex = get_u_exact(n, le2d.p, u_exact_func)

    print(-np.round(2 * mu * le2d._a1_dirichlet @ le2d._rg / 1e5, 2), "a1 dir load")
    print(-np.round(lam * le2d._a2_dirichlet @ le2d._rg / 1e5, 2), "a2 dir load")
    print(np.round(le2d._f_load_lv_free / 1e5, 2), "f lv load")

    print(np.round(le2d.compute_f_load_free(e_mean, nu_mean) / 1e5, 2), "f load")
    print(np.round(le2d.compute_a_free(e_mean, nu_mean) @ uex[le2d._expanded_free_index] / 1e5, 2), "needed f_load")
    print(le2d._a1_dirichlet.shape, le2d._rg.shape)


    print("_-------"*10)
    print(np.round(uh.reshape((n*n, 2)), 2))
    print(np.round(uex.reshape((n * n, 2)), 2))

    print(le2d._a2_free.A, "a2_free")
    print(le2d._free_index)
    print(le2d.p)
    print(le2d.p[le2d._free_index, :])
    print("_"*1000)
    print(le2d._a2_full.A)

    print("-"*100)
    print(np.round(uh, 2))
    print(uex)

    test_res = True
    if not test_res:
        raise ValueError("test not passed")

def case_5():
    print("Case 5: (xy, 0)")

    def u_exact_func(x, y, d):
        return x * y, 0.

    def dirichlet_bc_func(x, y, d):
        return x * y, 0.

    base(dirichlet_bc_func, u_exact_func)


def main():
    case_5()

if __name__ == '__main__':
    main()
