# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np

from ._assembly import get_basis_coef, epsilon, nabla_div
from .helpers import get_lambda_mu, inv_index_map


def sigma(e_young, nu_poisson, ck, i, d):
    """
    Compute the stress tensor

    Parameters
    ----------
    e_young : float
        young's module.
    nu_poisson : float
        poisson ratio.
    ck : np.array
        basis function coef. matrix.
    i : int
        which basis function to use.
    d : int
        which dimension.

    Returns
    -------
    np.array
        the stress tensor.

    """
    # get Lambert's coeffs.
    mu, lam = get_lambda_mu(e_young, nu_poisson)
    return 2 * mu * epsilon(ck, i, d) + lam * nabla_div(ck, i, d) * np.identity(2)


def get_element_stress(uh, p, tri):
    """
    Recover the element stress

    Parameters
    ----------
    uh : SolutionFunctionValues2D
        numerical solution.
    p : np.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    tri : np.array
        Elements. Index to the three corners of element i given in row i.

    Returns
    -------
    element_stress : np.array
        recovered element stress.

    """
    n_el = tri.shape[0]
    element_stress = np.zeros((n_el, 2, 2))
    for el_nr, nk in enumerate(tri):
        # nk : node-numbers for the k'th triangle
        # the points of the triangle
        # p1 = p[nk[0], :], uh1 = uh.values[nk[0], :]
        # p2 = p[nk[1], :], uh2 = uh.values[nk[1], :]
        # p3 = p[nk[2], :], uh3 = uh.values[nk[2], :]
        # using indexmap k = 2 * i + d, d=0 for x, 1 for y, i is the node number
        # and basis functions coef. or Jacobin inverse
        ck = get_basis_coef(*p[nk, :])
        for k in range(6):
            i, d = inv_index_map(k)
            # i gives basis, d gives dimension
            element_stress[el_nr, :, :] += uh.values[nk[i], d] * sigma(uh.e_young, uh.nu_poisson, ck, i, d)
    return element_stress


def get_node_neighbour_elements(node_nr, tri):
    """
    get an array of the index for neighbour elements of a node

    Parameters
    ----------
    node_nr : int
        index of the node in tri.
    tri : np.array
        Elements. Index to the three corners of element i given in row i.

    Returns
    -------
    np.array
        array of the index for neighbour elements of a node.

    """
    return np.argwhere(tri == node_nr)[:, 0]


def get_nodal_stress(uh, p, tri):
    """
    Recover the nodal stress

    Parameters
    ----------
    uh : SolutionFunctionValues2D
        numerical solution.
    p : np.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    tri : np.array
        Elements. Index to the three corners of element i given in row i.

    Returns
    -------
    None.

    """
    n_nodes = p.shape[0]
    # recover the element stress
    element_stress = get_element_stress(uh, p, tri)
    nodal_stress = np.zeros((n_nodes, 2, 2))
    for node_nr in np.unique(tri):
        # get index of the neighbour elements
        node_n_el = get_node_neighbour_elements(node_nr, tri)
        # recovery, calciulate the mean value.
        nodal_stress[node_nr, :, :] = np.mean(element_stress[node_n_el, :, :], axis=0)
    uh.set_nodal_stress(nodal_stress)


def get_von_mises_stress(uh):
    """
    Calculate the von mises stress, using the stress recovery process above.

    Parameters
    ----------
    uh : SolutionFunctionValues2D
        numerical solution.

    Returns
    -------
    None.

    """
    n_nodes = uh.nodal_stress.shape[0]
    von_mises = np.zeros(n_nodes)
    for node_nr in range(n_nodes):
        # deviatoric stress
        s = uh.nodal_stress[node_nr, :, :] - np.trace(uh.nodal_stress[node_nr, :, :]) * np.identity(2) / 3
        von_mises[node_nr] = np.sqrt(3 / 2 * np.sum(s * s))
    uh.set_von_mises(von_mises)
