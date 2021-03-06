# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from itertools import product

import numpy as np
from scipy.linalg import eigh, fractional_matrix_power

from .exceptions import SolutionMatrixIsZeroCanNotComputePODError


# le2d is a LinearElasticity2dProblem, not imported due to circular import
# rb_data is ReducedOrderData, not imported due to circular import
# from ._linear_elasticity_2d_problem_class import LinearElasticity2DProblem
# from ._rb_data_class import ReducedOrderData


def get_vec_from_range(range_, m, mode):
    """
    Get vector of m uniform or Gauss-Lobatto points from range_

    Parameters
    ----------
    range_ : tuple
        the range of numbers to consider.
    m : int
        number of points in vector.
    mode : str
        sampling mode, uniform or Gauss-Lobatto.

    Raises
    ------
    NotImplementedError
        if mode is not uniform or Gauss-Lobatto.

    Returns
    -------
    np.array
        array of sampling points.

    """
    if mode.lower() == "uniform":
        return np.linspace(range_[0], range_[1], m)

    elif mode.lower() == "gauss-lobatto":
        from quadpy.c1 import gauss_lobatto
        return 0.5 * ((range_[1] - range_[0]) * gauss_lobatto(m).points + (range_[1] + range_[0]))
    else:
        raise NotImplementedError(
            f"Mode {mode} is not implemented. The implemented modes are uniform and gauss lobatto.")


def make_solution_matrix(ns, e_young_vec, nu_poisson_vec, le2d):
    """
    

    Parameters
    ----------
    ns : int
        number of snapshots.
    e_young_vec : TYPE
        array of young's modules.
    nu_poisson_vec : np.array
        array of poisson ratios.
    le2d : 
        the solver.

    Raises
    ------
    SolutionMatrixIsZeroCanNotComputePODError
        If all values is the snapshot matrix s_mat are zero.

    Returns
    -------
    s_mat : np.array
        snapshot matrix.

    """
    s_mat = np.zeros((le2d.n_free, ns))
    i = 0
    # solve system for all combinations of (e_young, nu_poisson)
    for (e_young, nu_poisson) in product(e_young_vec, nu_poisson_vec):
        le2d.hfsolve(e_young, nu_poisson, print_info=False)
        s_mat[:, i] = le2d.uh_free
        i += 1
    if (s_mat == 0).all():
        error_text = "Solution matrix is zero, can not compute POD for building a reduced model. " \
                     + "The most likely cause is f_func=0, dirichlet_bc_func=0 and neumann_bc_func=0, " \
                      + "where two last may be None."
        raise SolutionMatrixIsZeroCanNotComputePODError(error_text)
    return s_mat


def pod_with_energy_norm(le2d, rb_data):
    """
    Proper orthogonal decomposition with respect to the energy norm

    Parameters
    ----------
    le2d : 
        the solver.
    rb_data : 
        reduced-order data.

    Returns
    -------
    None.

    """
    e_young_vec = get_vec_from_range(rb_data.e_young_range, rb_data.rb_grid[0], rb_data.pod_sampling_mode)
    nu_poisson_vec = get_vec_from_range(rb_data.nu_poisson_range, rb_data.rb_grid[1], rb_data.pod_sampling_mode)

    e_mean = np.mean(rb_data.e_young_range)
    nu_mean = np.mean(rb_data.nu_poisson_range)

    rb_data.s_mat = make_solution_matrix(rb_data.ns_rom, e_young_vec, nu_poisson_vec, le2d)
    a_free = le2d.compute_a_free(e_mean, nu_mean)
    if rb_data.ns_rom <= le2d.n_free:
        # build correlation matrix
        corr_mat = rb_data.s_mat.T @ a_free @ rb_data.s_mat
        # find the eigenvalues and eigenvectors of it
        sigma2_vec, z_mat = eigh(corr_mat)
        # reverse arrays because they are in ascending order
        rb_data.sigma2_vec = sigma2_vec[::-1]
        rb_data.z_mat = z_mat[:, ::-1]
    else:
        rb_data.x05 = fractional_matrix_power(a_free.A, 0.5)
        # build correlation matrix
        corr_mat = rb_data.x05 @ rb_data.s_mat @ rb_data.s_mat.T @ rb_data.x05
        # find the eigenvalues and eigenvectors of it
        sigma2_vec, z_mat = eigh(corr_mat)
        # reverse arrays because they are in ascending order
        rb_data.sigma2_vec = sigma2_vec[::-1]
        rb_data.z_mat = z_mat[:, ::-1]
    # compute n_rom from relative information content
    i_n = np.cumsum(rb_data.sigma2_vec) / np.sum(rb_data.sigma2_vec)
    rb_data.n_rom = np.min(np.argwhere(i_n >= 1 - rb_data.eps_pod ** 2)) + 1


def compute_v(n_rom, n_free, rb_data):
    """
    Compute the matrix V

    Parameters
    ----------
    n_rom : int
        our chosen "reduced-order degrees of freedom" ("n_rom"),
        can be set to different from n_rom-true.
    n_free : int
        the high-fidelity degrees of freedom.
    rb_data : 
        reduced-order data.

    Returns
    -------
    None.

    """
    if rb_data.ns_rom <= n_free:
        rb_data.v = rb_data.s_mat @ rb_data.z_mat[:, :n_rom] / np.sqrt(rb_data.sigma2_vec[:n_rom])
    else:
        rb_data.v = np.linalg.solve(rb_data.x05, rb_data.z_mat[:, :n_rom])


def get_e_young_nu_poisson_mat(rb_data):
    """
    Get the matrix of all combinations of (e_young, nu_piosson)

    Parameters
    ----------
    rb_data : 
        reduced-order data.

    Returns
    -------
    np.array
        the matrix of all combinations of (e_young, nu_piosson).

    """
    e_young_vec = get_vec_from_range(rb_data.e_young_range, rb_data.rb_grid[0], rb_data.pod_sampling_mode)
    nu_poisson_vec = get_vec_from_range(rb_data.nu_poisson_range, rb_data.rb_grid[1], rb_data.pod_sampling_mode)
    return np.array(list(product(e_young_vec, nu_poisson_vec)))
