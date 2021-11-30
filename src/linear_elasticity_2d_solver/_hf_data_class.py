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

        self.get_dirichlet_edge_func = None

        self.p = None
        self.tri = None
        self.edge = None

        self.a1_full = None
        self.a2_full = None
        self.f_load_lv_full = None
        self.f_load_neumann_full = None
        self.rg = None

        self.dirichlet_edge = None
        self.neumann_edge = None

        self.n = None
        self.n_full = None
        self.n_free = None

        self.plate_limits = plate_limits

        self.free_index = None
        self.dirichlet_edge_index = None
        self.expanded_free_index = None
        self.expanded_dirichlet_edge_index = None

    def hf_assemble_a1_a2_f(self, f_func_vec):
        self.p, self.tri, self.edge = getPlatev3(self.n, *self.plate_limits)
        #   p		Nodal points, (x,y)-coordinates for point i given in row i.
        #   tri   	Elements. Index to the three corners of element i given in row i.
        #   edge  	Edge lines. Index list to the two corners of _edge line i given in row i.

        self.a1_full, self.a2_full, self.f_load_lv_full = \
            assemble_a1_a2_f(self.n, self.p, self.tri, f_func_vec)

    def _get_dirichlet_edge(self):
        if self.get_dirichlet_edge_func is not None:
            dirichlet_edge_func_vec = np.vectorize(self.get_dirichlet_edge_func, otypes=[bool])
            index = dirichlet_edge_func_vec(self.p[self.edge, 0], self.p[self.edge, 1]).all(axis=1)
            if index.any():
                self.dirichlet_edge = self.edge[index, :]

    def get_neumann_edge(self):
        if self.dirichlet_edge is None:
            self.neumann_edge = self.edge
        elif self.dirichlet_edge.shape != self.edge.shape:
            neumann_edge = np.array(list(set(map(tuple, self.edge)) - set(map(tuple, self.dirichlet_edge))))
            self.neumann_edge = neumann_edge[np.argsort(neumann_edge[:, 0]), :]

    def _are_edges_illegal(self):
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

    def hf_assemble_f_neumann(self, neumann_bc_func_vec, has_homo_neumann):
        self._get_dirichlet_edge()
        self.get_neumann_edge()
        self._are_edges_illegal()
        self.f_load_neumann_full = assemble_f_neumann(self.n, self.p, self.neumann_edge,
                                                      neumann_bc_func_vec, has_homo_neumann)

    def _set_free_and_dirichlet_edge_index(self):
        if self.dirichlet_edge is None:
            self.dirichlet_edge_index = np.array([])
            self.free_index = np.unique(self.tri)
        else:
            # find the unique neumann _edge index
            self.dirichlet_edge_index = np.unique(self.dirichlet_edge)
            # free index is unique index minus dirichlet edge index
            self.free_index = np.setdiff1d(self.tri, self.dirichlet_edge_index)

    def _set_expanded_free_and_dirichlet_edge_index(self):
        self.expanded_free_index = expand_index(self.free_index)
        self.expanded_dirichlet_edge_index = expand_index(self.dirichlet_edge_index)

    def _set_a1_a2_dirichlet(self):
        dirichlet_xy_index = np.ix_(self.expanded_free_index, self.expanded_dirichlet_edge_index)
        self.a1_dirichlet = self.a1_full[dirichlet_xy_index]
        self.a2_dirichlet = self.a2_full[dirichlet_xy_index]

    def _set_a1_a2_and_f_free(self, has_neumann):
        free_xy_index = np.ix_(self.expanded_free_index, self.expanded_free_index)
        self.a1_free = self.a1_full[free_xy_index]
        self.a2_free = self.a2_full[free_xy_index]
        self.f_load_lv_free = self.f_load_lv_full[self.expanded_free_index]
        if has_neumann:
            self.f_load_neumann_free = self.f_load_neumann_full[self.expanded_free_index]

    def compute_free_and_expanded_edges(self, has_neumann):
        # set self.p, self.tri, self.edge
        # self.a1_full, self.a2_full
        # self.f_load_lv_full , self.dirichlet_edge
        # optionally: self.f_load_neumann_full,  neumann_edge
        # before calling this function

        self.n_full = self.a1_full.shape[0]
        self._set_free_and_dirichlet_edge_index()
        self._set_expanded_free_and_dirichlet_edge_index()
        self._set_a1_a2_and_f_free(has_neumann)
        self._set_a1_a2_dirichlet()
        self.n_free = self.a1_free.shape[0]

    def compute_lifting_function_on_edge(self, dirichlet_bc_func_vec):
        # compute a_dirichlet
        # x and y on the dirichlet_edge
        x_vec = self.p[self.dirichlet_edge_index][:, 0]
        y_vec = self.p[self.dirichlet_edge_index][:, 1]
        # lifting function
        self.rg = FunctionValues2D.from_nx2(dirichlet_bc_func_vec(x_vec, y_vec)).flatt_values
