# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np

from ._default_constants import EPS_POD, E_YOUNG_RANGE, NU_POISSON_RANGE, RB_GRID, POD_MODE, N_ROM_CUT
from ._pod import compute_v


class ReducedOrderData:
    def __init__(self):
        # set to something it never is
        self.last_n_rom = -1
        # set as a large number
        self.n_rom_max = np.inf
        self.n_rom_cut = N_ROM_CUT

        self.e_young_range = E_YOUNG_RANGE
        self.nu_poisson_range = NU_POISSON_RANGE
        self.rb_grid = RB_GRID
        self.ns_rom = RB_GRID[0] * RB_GRID[1]
        self.eps_pod = EPS_POD
        self.pod_mode = POD_MODE

        self.v = None
        self.n_rom = None
        self.s_mat = None
        self.z_mat = None
        self.sigma2_vec = None

        self.a1_free_rom = None
        self.a2_free_rom = None
        self.f_load_lv_free_rom = None
        self.f_load_neumann_free_rom = None
        self.a1_dirichlet_rom = None
        self.a2_dirichlet_rom = None

    def set_rb_model_params(self, grid=None, e_young_range=None, nu_poisson_range=None, eps_pod=None,
                            mode=None, n_rom_cut=None):
        if grid is not None:
            if isinstance(grid, int):
                self.rb_grid = (grid, grid)
            elif isinstance(grid, (tuple, list)):
                self.rb_grid = grid
            else:
                raise ValueError("Parameter gid is not int, tuple or list (of ints).")

        self.ns_rom = self.rb_grid[0] * self.rb_grid[1]

        if e_young_range is not None:
            self.e_young_range = e_young_range
        if nu_poisson_range is not None:
            self.nu_poisson_range = nu_poisson_range
        if eps_pod is not None:
            self.eps_pod = eps_pod
        if mode is not None:
            self.pod_mode = mode
        if n_rom_cut is not None:
            self.n_rom_cut = n_rom_cut

    def _compute_v(self, n_rom):
        if n_rom > self.n_rom_max:
            raise ValueError(f"n_rom={n_rom} is larger than maximum reduced order dept: {self.n_rom_max}")
        compute_v(n_rom, self)

    def _compute_rom_matrices_and_vectors(self, hf_data, has_neumann, has_non_homo_dirichlet):
        self.a1_free_rom = self.v.T @ hf_data.a1_free @ self.v
        self.a2_free_rom = self.v.T @ hf_data.a2_free @ self.v
        self.f_load_lv_free_rom = self.v.T @ hf_data.f_load_lv_free
        if has_neumann:
            self.f_load_neumann_free_rom = self.v @ hf_data.f_load_neumann_free
        if has_non_homo_dirichlet:
            self.a1_dirichlet_rom = self.v.T @ hf_data.a1_dirichlet
            self.a2_dirichlet_rom = self.v.T @ hf_data.a2_dirichlet

    def compute_rb_matrices_and_vectors(self, n_rom, hf_data, has_neumann, has_non_homo_dirichlet):
        self._compute_v(n_rom)
        self._compute_rom_matrices_and_vectors(hf_data, has_neumann, has_non_homo_dirichlet)

    def set_n_rom_max(self):
        self.n_rom_max = np.max(np.argwhere(self.sigma2_vec > self.n_rom_cut)) + 1
