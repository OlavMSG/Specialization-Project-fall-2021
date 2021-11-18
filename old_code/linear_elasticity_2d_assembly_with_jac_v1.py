# -*- coding: utf-8 -*-
"""
Created on 18.10.2021

@author: Olav Milian Gran
"""

import numpy as np
import scipy.sparse as sparse
from gauss_quadrature import quadrature2D_area_coor, line_integral_with_basis, get_area_triangle
from linerar_elasticity_2d_helpers import inv_index_map, expand_index


def phi_area_coor(z, i):
    # z[:, i] = lam_i
    return z[:, i]


def get_jac_inv(p1, p2, p3):
    jac = np.column_stack((p1 - p3, p2 - p3))
    det_jac = (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    jac_inv = np.array([[jac[1, 1], -jac[0, 1]], [-jac[1, 0], jac[0, 0]]]) / det_jac
    return jac_inv


def nabla_grad_with_jac(i, d):
    # Ck_ref = [[0, 0, 1],
    #           [1, 0, -1],
    #           [0, 1, -1]]
    # Ck = [[ck_1,  ck_2,  ck_3 ],  1  row index 0
    #       [ckx_1, ckx_2, ckx_3],  x  row index 1
    #       [cky_1, cky_2, cky_3]]  y  row index 2
    # col in : 0  ,   1  ,  2
    ck_area_coor_reduced = np.array([[1, 0, -1],
                                     [0, 1, -1]], dtype=np.float64)
    ckx, cky = ck_area_coor_reduced[:, i]
    if d == 0:
        # case y-part equal 0 of basisfunc
        return np.array([[ckx, 0.],
                         [cky, 0.]])
    else:
        # case x-part equal 0 of basisfunc
        return np.array([[0., ckx],
                         [0., cky]])


def epsilon_with_jac(jac_inv, i, d):
    # epsilon with reference functions, using the Jacobin inverse for scaling
    #
    # phi1 = lambda x, y: [1, x, y] @ Ck[:, 0]
    # phi2 = lambda x, y: [1, x, y] @ Ck[:, 1]
    # phi3 = lambda x, y: [1, x, y] @ Ck[:, 2]
    # jac_inv = [[a, b], | jac_inv.T = [[a, c],
    #            [c, d]] |              [b, d]]

    jac_inv_nabla_grad = jac_inv.T @ nabla_grad_with_jac(i, d)
    return 0.5 * (jac_inv_nabla_grad + jac_inv_nabla_grad.T)


def nabla_div_with_jac(jac_inv, i, d):
    # nabla div with reference functions, using the Jacobin inverse for scaling
    # Ck_ref = [[0, 0, 1],
    #           [1, 0, -1],
    #           [0, 1, -1]]
    # Ck = [[ck_1,  ck_2,  ck_3 ],  1  row index 0
    #       [ckx_1, ckx_2, ckx_3],  x  row index 1
    #       [cky_1, cky_2, cky_3]]  y  row index 2
    # col in : 0  ,   1  ,  2
    # phi1 = lambda x, y: [1, x, y] @ Ck[:, 0]
    # phi2 = lambda x, y: [1, x, y] @ Ck[:, 1]
    # phi3 = lambda x, y: [1, x, y] @ Ck[:, 2]
    ck_area_coor_reduced = np.array([[1, 0, -1],
                                     [0, 1, -1]], dtype=np.float64)
    if d == 0:
        # case y-part 0 of basisfunc
        # div = trace(jac_inv.T @ nabla_grad)
        return ck_area_coor_reduced[0, i] * jac_inv[0, 0] + ck_area_coor_reduced[1, i] * jac_inv[1, 0]
    else:
        # case x-part equal 0 of basisfunc
        # div = trace(jac_inv.T @ nabla_grad)
        return ck_area_coor_reduced[0, i] * jac_inv[0, 1] + ck_area_coor_reduced[1, i] * jac_inv[1, 1]


def assemble_a1_a2_local_with_jac(area, jac_inv):
    a1_local = np.zeros((6, 6), dtype=np.float64)
    a2_local = np.zeros((6, 6), dtype=np.float64)

    for ki in range(6):
        i, di = inv_index_map(ki)
        for kj in range(ki + 1):
            j, dj = inv_index_map(kj)
            # construct A1_local and A2_local
            eps_i_double_dot_eps_j = area * np.sum(epsilon_with_jac(jac_inv, i, di) * epsilon_with_jac(jac_inv, j, dj))
            div_i_div_j = area * nabla_div_with_jac(jac_inv, i, di) * nabla_div_with_jac(jac_inv, j, dj)
            a1_local[ki, kj] = eps_i_double_dot_eps_j
            a2_local[ki, kj] = div_i_div_j
            if ki != kj:
                a1_local[kj, ki] = eps_i_double_dot_eps_j
                a2_local[kj, ki] = div_i_div_j
    return a1_local, a2_local


def assemble_f_local_with_area_coor(f_func, p1, p2, p3):
    f_local = np.zeros(6, dtype=np.float64)

    for ki in range(6):
        i, di = inv_index_map(ki)

        def f_phi_area_coor(z):
            xy = np.multiply.outer(p1, z[:, 0]) + np.multiply.outer(p2, z[:, 1]) + np.multiply.outer(p3, z[:, 2])
            return f_func(*xy)[:, di] * phi_area_coor(z, i)

        f_local[ki] = quadrature2D_area_coor(p1, p2, p3, 4, f_phi_area_coor)
    return f_local


def assemble_a1_a2_f(n, p, tri, f_func):
    n2d = n * n * 2
    # Stiffness matrix
    a1 = sparse.dok_matrix((n2d, n2d), dtype=np.float64)
    a2 = sparse.dok_matrix((n2d, n2d), dtype=np.float64)
    # dok_matrix
    # Allows for efficient O(1) access of individual elements
    # load vector
    f_load_lv = np.zeros(n2d, dtype=np.float64)

    for nk in tri:
        # nk : node-numbers for the k'th triangle
        # the points of the triangle
        # p1 = _p[nk[0], :]
        # p2 = _p[nk[1], :]
        # p3 = _p[nk[2], :]
        # using indexmap k = 2 * i + d, d=0 for x, 1 for y, i is the node number
        # calculate the area of the triangle
        # and basis functions coef. or jacobian inverse
        jac_inv = get_jac_inv(*p[nk, :])
        area = get_area_triangle(*p[nk, :])
        a1_local, a2_local = assemble_a1_a2_local_with_jac(area, jac_inv)
        f_local = assemble_f_local_with_area_coor(f_func, *p[nk, :])

        expanded_nk = expand_index(nk)
        index = np.ix_(expanded_nk, expanded_nk)
        a1[index] += a1_local
        a2[index] += a2_local
        f_load_lv[expanded_nk] += f_local

    return a1, a2, f_load_lv


def assemble_f_neumann(n, p, neumann_edge, neumann_bc_func):
    n2d = n * n * 2
    # load vector
    f_load_neumann = np.zeros(n2d, dtype=np.float64)
    for ek in neumann_edge:
        # p1 = _p[ek[0], :]
        # p2 = _p[ek[1], :]
        expanded_ek = expand_index(ek)
        f_load_neumann[expanded_ek] += line_integral_with_basis(*p[ek, :], 4, neumann_bc_func)
    return f_load_neumann