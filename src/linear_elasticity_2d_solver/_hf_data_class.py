# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np

from ._getplate import getPlatev3
from ._assembly import assemble_a1_a2_f, assemble_f_neumann
from ._helpers import expand_index, FunctionValues2D
from ._default_constants import PLATE_LIMITS


class HighFidelityData:

    def __init__(self):

        self.get_dirichlet_and_neumann_edge = None

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

        self.plate_limits = PLATE_LIMITS

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

    def hf_assemble_f_neumann(self, neumann_bc_func_vec):
        self.dirichlet_edge, self.neumann_edge = \
            self.get_dirichlet_and_neumann_edge(self.p, self.edge)
        self.f_load_neumann_full = assemble_f_neumann(self.n, self.p, self.neumann_edge, neumann_bc_func_vec)

    def _set_free_and_dirichlet_edge_index(self):
        # find the unique neumann _edge index
        self.dirichlet_edge_index = np.unique(self.dirichlet_edge)
        # find the unique indexes
        unique_index = np.unique(self.tri)
        # free index is unique index minus neumann _edge indexes
        self.free_index = np.setdiff1d(unique_index, self.dirichlet_edge_index)

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
