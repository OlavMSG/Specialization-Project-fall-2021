# -*- coding: utf-8 -*-
"""
Created on 07.11.2021

@author: Olav Milian Gran
"""
# -*- coding: utf-8 -*-
"""
Created on 18.10.2021

@author: Olav Milian Gran
"""

import numpy as np
import scipy.sparse as sparse
from gauss_quadrature import quadrature2D, line_integral_with_basis
from linerar_elasticity_2d_helpers import func_vec, inv_index_map, expand_index
from getplate import getPlatev3
from scipy.sparse.linalg import spsolve


def phi(x, y, ck, i):
    # Ck = [[ck_1,  ck_2,  ck_3 ],  1  row index 0
    #       [ckx_1, ckx_2, ckx_3],  x  row index 1
    #       [cky_1, cky_2, cky_3]]  y  row index 2
    # col in : 0  ,   1  ,  2
    # phi1 = lambda x, y: [1, x, y] @ Ck[:, 0]
    # phi2 = lambda x, y: [1, x, y] @ Ck[:, 1]
    # phi3 = lambda x, y: [1, x, y] @ Ck[:, 2]
    return ck[0, i] + ck[1, i] * x + ck[2, i] * y


def f_phi(x, y, f_func, ck, i, d):
    return func_vec(x, y, d, f_func) * phi(x, y, ck, i)


def get_basis_coef_area(p1, p2, p3):
    det_jac = (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    area = 0.5 * det_jac
    # calculate basis functions.
    # row_k: [1, x_k, y_k]

    mk = np.array([[1, p1[0], p1[1]],
                   [1, p2[0], p2[1]],
                   [1, p3[0], p3[1]]])
    ck = np.linalg.inv(mk)  # here faster than solving Mk @ Ck = I_3
    return ck, area

def grad(ck, i):
    return np.array([ck[1, i], ck[2, i]])




def assemble_a1_a2_local(area, ck):
    a1_local = np.zeros((3, 3), dtype=float)
    a2_local = np.zeros((3, 3), dtype=float)

    for i in range(3):
        for j in range(i + 1):
            # construct A1_local and A2_local
            eps_i_double_dot_eps_j = area * grad(ck, i) @ grad(ck, j)
            a1_local[i, j] = eps_i_double_dot_eps_j
            a2_local[i, j] = 0
            if i != j:
                a1_local[j, i] = eps_i_double_dot_eps_j
                a2_local[j, i] = 0
    return a1_local, a2_local


def assemble_f_local(area, ck, f_func, p1, p2, p3):
    f_local = np.zeros(3, dtype=float)

    for i in range(3):
        f_local[i] = quadrature2D(p1, p2, p3, 4, f_phi, f_func, ck, i, 0, area)
    return f_local


def assemble_a1_a2_f(n, p, tri, f_func):
    n2d = n * n
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
        # p1 = _p[nk[0], :]
        # p2 = _p[nk[1], :]
        # p3 = _p[nk[2], :]
        # using indexmap k = 2 * i + d, d=0 for x, 1 for y, i is the node number
        # calculate the area of the triangle
        # and basis functions coef. or Jacobin inverse
        ck, area = get_basis_coef_area(*p[nk, :])
        a1_local, a2_local = assemble_a1_a2_local(area, ck)
        f_local = assemble_f_local(area, ck, f_func, *p[nk, :])

        index = np.ix_(nk, nk)
        a1[index] += a1_local
        a2[index] += a2_local
        f_load_lv[nk] += f_local
        count += 1
        if count % 10000 == 0:
            print("a1, a2, f computed for {} element".format(count))
    return a1, a2, f_load_lv


def assemble_f_neumann(n, p, neumann_edge, neumann_bc_func):
    n2d = n * n
    # load vector
    f_load_neumann = np.zeros(n2d, dtype=float)
    count = 0
    for ek in neumann_edge:
        # p1 = _p[ek[0], :]
        # p2 = _p[ek[1], :]
        f_load_neumann[ek] += line_integral_with_basis(*p[ek, :], 4, func_vec, neumann_bc_func)[0, 2]
        count += 1
        if count % 10000 == 0:
            print("a1, a2, f computed for {} element".format(count))
    return f_load_neumann


def main(n):
    n2 = n * n
    def f(x, y, d):
        return 0
    p, tri, edge = getPlatev3(n, 0, 1)
    a1, a2, f = assemble_a1_a2_f(n, p, tri, f)
    # find the unique neumann _edge index
    dirichlet_edge_index = np.unique(edge)
    # find the unique indexes
    unique_index = np.unique(tri)
    # free index is unique index minus neumann _edge indexes
    free_index = np.setdiff1d(unique_index, dirichlet_edge_index)

    ind_free = np.ix_(free_index, free_index)
    ind_dir = np.ix_(free_index, dirichlet_edge_index)
    a_free = a1[ind_free]
    a_dir = a1[ind_dir]

    def u_bc_func(x, y):
        return x * y

    def u_ex(x, y):
        return x * y

    x_edge = p[dirichlet_edge_index, 0]
    y_edge = p[dirichlet_edge_index, 1]
    rg = u_bc_func(x_edge, y_edge)
    f_load = - a_dir @ rg
    print(a_free.shape, f_load.shape)
    uh = np.zeros(n2)
    print(free_index)
    print(dirichlet_edge_index)
    uh[free_index] = spsolve(a_free.tocsr(), f_load)
    uh[dirichlet_edge_index] = rg

    x_vec = p[:, 0]
    y_vec = p[:, 1]
    uex = u_ex(x_vec, y_vec)

    print(uh)
    print(uex)
    print(np.all(np.abs(uh - uex) <= 1e-8))



if __name__ == '__main__':
    main(4)
