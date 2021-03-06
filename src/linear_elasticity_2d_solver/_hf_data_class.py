# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np

from ._assembly import assemble_a1_a2_f, assemble_f_neumann
from .default_constants import plate_limits
from .exceptions import EdgesAreIllegalError
from .get_plate import getPlatev3
from .helpers import expand_index, FunctionValues2D


class HighFidelityData:

    def __init__(self):
        """
        Setup

        Returns
        -------
        None.

        """
        # function to get the Dirichlet edge
        self.get_dirichlet_edge_func = None
        # Triangulation
        self.p = None
        self.tri = None
        self.edge = None
        # full matrices and load vectors
        self.a1_full = None
        self.a2_full = None
        self.f_load_lv_full = None
        self.f_load_neumann_full = None
        # lifting function
        self.rg = None
        # Dirichlet and Neumann edge
        self.dirichlet_edge = None
        self.neumann_edge = None
        # size parameter
        self.n = None
        # full number of degrees
        self.n_full = None
        # hf number of degrees of freedom
        self.n_free = None
        # the plate limits
        self.plate_limits = plate_limits
        # free and dirichlet indexes
        self.free_index = None
        self.dirichlet_edge_index = None
        self.expanded_free_index = None
        self.expanded_dirichlet_edge_index = None

    def hf_assemble_a1_a2_f(self, f_func_vec, f_func_is_not_zero):
        """
        Assemble the high-fidelity matrices and body force load vector

        Parameters
        ----------
        f_func_vec : function
            the body force function.
        f_func_is_not_zero : bool
            True if the function f_func_vec does not return zero for all x,y.

        Returns
        -------
        None.

        """
        self.p, self.tri, self.edge = getPlatev3(self.n, *self.plate_limits)
        #   p		Nodal points, (x,y)-coordinates for point i given in row i.
        #   tri   	Elements. Index to the three corners of element i given in row i.
        #   edge  	Edge lines. Index list to the two corners of _edge line i given in row i.

        self.a1_full, self.a2_full, self.f_load_lv_full = \
            assemble_a1_a2_f(self.n, self.p, self.tri, f_func_vec, f_func_is_not_zero)

    def _get_dirichlet_edge(self):
        """
        Get the Dirichlet edge

        Returns
        -------
        None.

        """
        if self.get_dirichlet_edge_func is not None:
            dirichlet_edge_func_vec = np.vectorize(self.get_dirichlet_edge_func, otypes=[bool])
            index = dirichlet_edge_func_vec(self.p[self.edge, 0], self.p[self.edge, 1]).all(axis=1)
            if index.any():
                self.dirichlet_edge = self.edge[index, :]

    def get_neumann_edge(self):
        """
        Get the Neumann edge

        Returns
        -------
        None.

        """
        if self.dirichlet_edge is None:
            self.neumann_edge = self.edge
        elif self.dirichlet_edge.shape != self.edge.shape:
            neumann_edge = np.array(list(set(map(tuple, self.edge)) - set(map(tuple, self.dirichlet_edge))))
            self.neumann_edge = neumann_edge[np.argsort(neumann_edge[:, 0]), :]

    def _are_edges_illegal(self):
        """
        Check if the edges are illegal for the solver, if so raise an error

        Raises
        ------
        EdgesAreIllegalError
            if the edges are for the solver.

        Returns
        -------
        None.

        """
        if self.get_dirichlet_edge_func is None:
            if np.all(self.neumann_edge == self.edge):
                error_text = "Only neumann conditions are not allowed, gives neumann_edge=edge, " \
                             + "please define get_dirichlet_edge_func."
                raise EdgesAreIllegalError(error_text)
        else:
            if (self.dirichlet_edge is None) and np.all(self.neumann_edge == self.edge):
                raise EdgesAreIllegalError("get_dirichlet_edge_func gives dirichlet_edge=None and neumann_edge=edge.")
            if (self.neumann_edge is None) and np.all(self.dirichlet_edge == self.edge):
                raise EdgesAreIllegalError("get_dirichlet_edge_func gives dirichlet_edge=edge and neumann_edge=None.")

    def edges(self):
        """
        Get the edges

        Returns
        -------
        None.

        """
        self._get_dirichlet_edge()
        self.get_neumann_edge()
        self._are_edges_illegal()

    def hf_assemble_f_neumann(self, neumann_bc_func_vec):
        """
        Assemble the high-fidelity Neumann load vector

        Parameters
        ----------
        neumann_bc_func_vec : function
            Neumann boundary condition function.

        Returns
        -------
        None.

        """
        self.f_load_neumann_full = assemble_f_neumann(self.n, self.p, self.neumann_edge, neumann_bc_func_vec)

    def _set_free_and_dirichlet_edge_index(self):
        """
        Set the free and dirichlet edge indexes

        Returns
        -------
        None.

        """
        if self.dirichlet_edge is None:
            self.dirichlet_edge_index = np.array([])
            self.free_index = np.unique(self.tri)
        else:
            # find the unique neumann _edge index
            self.dirichlet_edge_index = np.unique(self.dirichlet_edge)
            # free index is unique index minus dirichlet edge index
            self.free_index = np.setdiff1d(self.tri, self.dirichlet_edge_index)

    def _set_expanded_free_and_dirichlet_edge_index(self):
        """
        Expand the free and dirichlet edge indexes

        Returns
        -------
        None.

        """
        self.expanded_free_index = expand_index(self.free_index)
        self.expanded_dirichlet_edge_index = expand_index(self.dirichlet_edge_index)

    def _set_a1_a2_dirichlet(self):
        """
        Set the dirichlet matrices form a1 and a2

        Returns
        -------
        None.

        """
        dirichlet_xy_index = np.ix_(self.expanded_free_index, self.expanded_dirichlet_edge_index)
        self.a1_dirichlet = self.a1_full[dirichlet_xy_index]
        self.a2_dirichlet = self.a2_full[dirichlet_xy_index]

    def _set_a1_a2_and_f_free(self, has_neumann, has_non_homo_neumann):
        """
        

        Parameters
        ----------
        has_neumann : bool
            True if the problem has Neumann boundary conditions.
        has_non_homo_neumann : bool
            True if the problem has non homogeneous Neumann boundary conditions.

        Returns
        -------
        None.

        """
        free_xy_index = np.ix_(self.expanded_free_index, self.expanded_free_index)
        self.a1_free = self.a1_full[free_xy_index]
        self.a2_free = self.a2_full[free_xy_index]
        self.f_load_lv_free = self.f_load_lv_full[self.expanded_free_index]
        if has_neumann and has_non_homo_neumann:
            self.f_load_neumann_free = self.f_load_neumann_full[self.expanded_free_index]

    def compute_free_and_expanded_edges(self, has_neumann, has_non_homo_neumann):
        """
        Compute the free and expanded edges, and more parameters

        Parameters
        ----------
        has_neumann : bool
            True if the problem has Neumann boundary conditions.
        has_non_homo_neumann : bool
            True if the problem has non homogeneous Neumann boundary conditions.

        Returns
        -------
        None.

        """
        # set self.p, self.tri, self.edge
        # self.a1_full, self.a2_full
        # self.f_load_lv_full , self.dirichlet_edge
        # optionally: self.f_load_neumann_full,  neumann_edge
        # before calling this function

        self.n_full = self.a1_full.shape[0]
        self._set_free_and_dirichlet_edge_index()
        self._set_expanded_free_and_dirichlet_edge_index()
        self._set_a1_a2_and_f_free(has_neumann, has_non_homo_neumann)
        self._set_a1_a2_dirichlet()
        self.n_free = self.a1_free.shape[0]

    def compute_lifting_function_on_edge(self, dirichlet_bc_func_vec):
        """
        Compute the lifting function

        Parameters
        ----------
        dirichlet_bc_func_vec : function
            Dirichlet boundary condition function.

        Returns
        -------
        None.

        """
        # compute a_dirichlet
        # x and y on the dirichlet_edge
        x_vec = self.p[self.dirichlet_edge_index][:, 0]
        y_vec = self.p[self.dirichlet_edge_index][:, 1]
        # lifting function
        self.rg = FunctionValues2D.from_nx2(dirichlet_bc_func_vec(x_vec, y_vec)).flatt_values
