# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np

from itertools import product
from scipy.linalg import eigh

# le2d is a LinearElasticity2dProblem, not imported due to circular import
# from ._linear_elasticity_2d_problem_class import LinearElasticity2DProblem
# rb_data is ReducedOrderData, not imported due to circular import
# from ._rb_data_class import ReducedOrderData


def get_vec_from_range(range_, m, mode):
    if mode == "uniform":
        return np.linspace(range_[0], range_[1], m)

    elif mode == "gauss lobatto":
        from quadpy.c1 import gauss_lobatto
        return 0.5 * ((range_[1] - range_[0]) * gauss_lobatto(m).points + (range_[1] + range_[0]))
    else:
        raise NotImplementedError(
            f"Mode {mode} is not implemented. The implemented modes are uniform and gauss lobatto")


def get_mean(range_):
    return 0.5 * (range_[1] + range_[0])


def make_solution_matrix(e_young_vec, nu_poisson_vec, le2d):
    ns = e_young_vec.shape[0] * nu_poisson_vec.shape[0]
    s_mat = np.zeros((le2d.n_free, ns))
    i = 0
    for (e_young, nu_poisson) in product(e_young_vec, nu_poisson_vec):
        le2d.hfsolve(e_young, nu_poisson, print_info=False)
        s_mat[:, i] = le2d.uh_free
        i += 1
    return s_mat


def pod(le2d, rb_data):
    e_young_vec = get_vec_from_range(rb_data.e_young_range, rb_data.rb_grid[0], rb_data.pod_mode)
    nu_poisson_vec = get_vec_from_range(rb_data.nu_poisson_range, rb_data.rb_grid[1], rb_data.pod_mode)
    e_mean = get_mean(rb_data.e_young_range)
    nu_mean = get_mean(rb_data.nu_poisson_range)
    rb_data.s_mat = make_solution_matrix(e_young_vec, nu_poisson_vec, le2d)
    # build correlation matrix
    corr_mat = rb_data.s_mat.T @ le2d.compute_a_free(e_mean, nu_mean) @ rb_data.s_mat
    # find the eigenvalues and eigenvectors of it
    sigma2_vec, z_mat = eigh(corr_mat)
    # reverse arrays
    rb_data.sigma2_vec = sigma2_vec[::-1]
    rb_data.z_mat = z_mat[:, ::-1]
    # compute n_rom from relative information content
    i_n = np.cumsum(rb_data.sigma2_vec) / np.sum(rb_data.sigma2_vec)
    rb_data.n_rom = np.min(np.argwhere(i_n >= 1 - rb_data.eps_pod ** 2)) + 1


def compute_v(n, rb_data):
    rb_data.v = rb_data.s_mat @ rb_data.z_mat[:, :n] / np.sqrt(rb_data.sigma2_vec[:n])


def get_e_young_nu_poisson_mat(rb_data):
    e_young_vec = get_vec_from_range(rb_data.e_young_range, rb_data.rb_grid[0], rb_data.pod_mode)
    nu_poisson_vec = get_vec_from_range(rb_data.nu_poisson_range, rb_data.rb_grid[1], rb_data.pod_mode)
    return np.array(list(product(e_young_vec, nu_poisson_vec)))
