# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import numpy as np
import scipy.sparse as sparse

from ._gauss_quadrature import quadrature2D, line_integral_with_basis, get_area_triangle
from ._helpers import inv_index_map, expand_index


def phi(x, y, ck, i):
    # Ck = [[ck_1,  ck_2,  ck_3 ],  1  row index 0
    #       [ckx_1, ckx_2, ckx_3],  x  row index 1
    #       [cky_1, cky_2, cky_3]]  y  row index 2
    # col in : 0  ,   1  ,  2
    # phi1 = lambda x, y: [1, x, y] @ Ck[:, 0]
    # phi2 = lambda x, y: [1, x, y] @ Ck[:, 1]
    # phi3 = lambda x, y: [1, x, y] @ Ck[:, 2]
    return ck[0, i] + ck[1, i] * x + ck[2, i] * y


def get_basis_coef(p1, p2, p3):
    # calculate basis functions.
    # row_k: [1, x_k, y_k]
    mk = np.array([[1, p1[0], p1[1]],
                   [1, p2[0], p2[1]],
                   [1, p3[0], p3[1]]])
    ck = np.linalg.inv(mk)  # here faster than solving Mk @ Ck = I_3
    return ck


def nabla_grad(ckx, cky, d):
    if d == 0:
        # case y-part equal 0 of basisfunc
        return np.array([[ckx, 0.],
                         [cky, 0.]])
    else:
        # case x-part equal 0 of basisfunc
        return np.array([[0., ckx],
                         [0., cky]])


def epsilon(ck, i, d):
    nabla_grad_ck = nabla_grad(ck[1, i], ck[2, i], d)
    return 0.5 * (nabla_grad_ck + nabla_grad_ck.T)


def nabla_div(ck, i, d):
    # d = 0
    # case y-part 0 of basisfunc
    # d = 1
    # case x-part 0 of basisfunc
    return ck[1 + d, i]


def assemble_a1_a2_local(area, ck):
    a1_local = np.zeros((6, 6), dtype=float)
    a2_local = np.zeros((6, 6), dtype=float)

    for ki in range(6):
        i, di = inv_index_map(ki)
        for kj in range(ki + 1):
            j, dj = inv_index_map(kj)
            # construct A1_local and A2_local
            eps_i_double_dot_eps_j = area * np.sum(epsilon(ck, i, di) * epsilon(ck, j, dj))
            div_i_div_j = area * nabla_div(ck, i, di) * nabla_div(ck, j, dj)
            a1_local[ki, kj] = eps_i_double_dot_eps_j
            a2_local[ki, kj] = div_i_div_j
            if ki != kj:
                a1_local[kj, ki] = eps_i_double_dot_eps_j
                a2_local[kj, ki] = div_i_div_j
    return a1_local, a2_local


def assemble_f_local(ck, f_func, p1, p2, p3):
    f_local = np.zeros(6, dtype=float)

    for ki in range(6):
        i, di = inv_index_map(ki)

        def f_phi(x, y):
            return f_func(x, y)[:, di] * phi(x, y, ck, i)

        f_local[ki] = quadrature2D(p1, p2, p3, 4, f_phi)
    return f_local


def assemble_a1_a2_f(n, p, tri, f_func):
    n2d = n * n * 2
    # Stiffness matrix
    a1 = sparse.dok_matrix((n2d, n2d), dtype=float)
    a2 = sparse.dok_matrix((n2d, n2d), dtype=float)

    # dok_matrix
    # Allows for efficient O(1) access of individual elements
    # load vector
    f_load_lv = np.zeros(n2d, dtype=float)
    count = 0
    for nk in tri:
        # nk : node-numbers for the k'th triangle
        # the points of the triangle
        # p1 = p[nk[0], :]
        # p2 = p[nk[1], :]
        # p3 = p[nk[2], :]
        # using indexmap k = 2 * i + d, d=0 for x, 1 for y, i is the node number
        # calculate the area of the triangle
        # and basis functions coef. or Jacobin inverse
        ck = get_basis_coef(*p[nk, :])
        area = get_area_triangle(*p[nk, :])
        a1_local, a2_local = assemble_a1_a2_local(area, ck)
        f_local = assemble_f_local(ck, f_func, *p[nk, :])

        expanded_nk = expand_index(nk)
        index = np.ix_(expanded_nk, expanded_nk)
        a1[index] += a1_local
        a2[index] += a2_local
        f_load_lv[expanded_nk] += f_local
        a = np.array([8, 9])
        count += 1
        if count % 10000 == 0:
            print("a1, a2, f computed for {} element".format(count))
    return a1, a2, f_load_lv


def assemble_f_neumann(n, p, neumann_edge, neumann_bc_func):
    n2d = n * n * 2
    # load vector
    f_load_neumann = np.zeros(n2d, dtype=float)
    count = 0
    for ek in neumann_edge:
        # p1 = p[ek[0], :]
        # p2 = p[ek[1], :]
        expanded_ek = expand_index(ek)
        f_load_neumann[expanded_ek] += line_integral_with_basis(*p[ek, :], 4, neumann_bc_func)
        count += 1
        if count % 10000 == 0:
            print("a1, a2, f computed for {} element".format(count))
    return f_load_neumann
