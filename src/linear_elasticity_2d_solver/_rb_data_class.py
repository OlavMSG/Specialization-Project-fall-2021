# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np

from ._pod import compute_v
from .default_constants import eps_pod, e_young_range, nu_poisson_range, rb_grid, pod_sampling_mode, n_rom_cut


class ReducedOrderData:
    def __init__(self):
        """
        Setup

        Returns
        -------
        None.

        """
        # set to something it never is
        self.last_n_rom = -1
        # set as a large number
        self.n_rom_max = np.inf
        self.n_rom_cut = n_rom_cut

        self.e_young_range = e_young_range
        self.nu_poisson_range = nu_poisson_range
        self.rb_grid = rb_grid
        self.ns_rom = rb_grid[0] * rb_grid[1]
        self.eps_pod = eps_pod
        self.pod_sampling_mode = pod_sampling_mode

        self.v = None
        self.n_rom = None
        self.s_mat = None
        self.z_mat = None
        self.sigma2_vec = None
        self.x05 = None

        self.a1_free_rom = None
        self.a2_free_rom = None
        self.f_load_lv_free_rom = None
        self.f_load_neumann_free_rom = None
        self.f1_dirichlet_rom = None
        self.f2_dirichlet_rom = None

    def set_rb_model_params(self, grid=None, e_young_range=None, nu_poisson_range=None, eps_pod=None,
                            mode=None, n_rom_cut=None):
        """
        Set the reduced-order model parameters

        Parameters
        ----------
        grid : int, tuple, list optional
            The grid size for the sampling, if int us it for both parameters.
            If None set to rb_grid from default_constants
            The default is None.
        e_young_range : tuple, optional
            range for the Young's module E. 
            If None set to e_young_range from default_constants
            The default is None.
        nu_poisson_range : tuple, optional
            range for the poisson ratio nu.
            If None set to nu_poisson_range from default_constants
            The default is None.
        eps_pod : float, optional
            tolerance for the POD algorithm.
            If None set to eps_POD from default_constants
            The default is None.
        mode : str, optional
            sampling mode.
            If None set to pod_sampling_mode from default_constants
            The default is None.
        n_rom_cut : int, str, optional
            value to cut n_rom at, may be string "rank" fro the rank of the solution matrix.
            If None set to n_rom_cut from default_constants
            The default is None.

        Raises
        ------
        ValueError
            if the parameter grid is not an int, tuple or list (of ints)..

        Returns
        -------
        None.

        """
        if grid is not None:
            if isinstance(grid, int):
                self.rb_grid = (grid, grid)
            elif isinstance(grid, (tuple, list)):
                self.rb_grid = grid
            else:
                raise ValueError("Parameter grid is not an int, tuple or list (of ints).")

        self.ns_rom = self.rb_grid[0] * self.rb_grid[1]

        if e_young_range is not None:
            self.e_young_range = e_young_range
        if nu_poisson_range is not None:
            self.nu_poisson_range = nu_poisson_range
        if eps_pod is not None:
            self.eps_pod = eps_pod
        if mode is not None:
            self.pod_sampling_mode = mode
        if n_rom_cut is not None:
            self.n_rom_cut = n_rom_cut

    def _compute_v(self, n_rom, n_free):
        """
        Compute the matrix V

        Parameters
        ----------
        n_rom : int
            our chosen "reduced-order degrees of freedom" ("n_rom"),
            can be set to different from n_rom-true.
        n_free : int
            the high-fidelity degrees of freedom.

        Raises
        ------
        ValueError
            if n_rom if larger than n_rom_max.

        Returns
        -------
        None.

        """
        if n_rom > self.n_rom_max:
            raise ValueError(f"n_rom={n_rom} is larger than maximum reduced order dept: {self.n_rom_max}")
        compute_v(n_rom, n_free, self)

    def _compute_rom_matrices_and_vectors(self, hf_data, has_neumann, has_non_homo_dirichlet, has_non_homo_neumann):
        """
        Compute the reduced-order matrices a1 and a2, and load vectors

        Parameters
        ----------
        hf_data :
            High-fidelity data.
        has_neumann : bool
            Does the problem have Neumann boundary conditions.
        has_non_homo_dirichlet : bool
            Does the problem have non homogeneous Dirichlet boundary conditions.
        has_non_homo_neumann : bool
            Does the problem have non homogeneous Neumann boundary conditions.

        Returns
        -------
        None.

        """
        self.a1_free_rom = self.v.T @ hf_data.a1_free @ self.v
        self.a2_free_rom = self.v.T @ hf_data.a2_free @ self.v
        self.f_load_lv_free_rom = self.v.T @ hf_data.f_load_lv_free
        if has_neumann and has_non_homo_neumann:
            self.f_load_neumann_free_rom = self.v.T @ hf_data.f_load_neumann_free
        if has_non_homo_dirichlet:
            self.f1_dirichlet_rom = self.v.T @ hf_data.a1_dirichlet @ hf_data.rg
            self.f2_dirichlet_rom = self.v.T @ hf_data.a2_dirichlet @ hf_data.rg

    def compute_rb_matrices_and_vectors(self, n_rom, hf_data, has_neumann,
                                        has_non_homo_dirichlet, has_non_homo_neumann):
        """
        Compute the reduced-order matrices v, a1 and a2, and load vectors

        Parameters
        ----------
        n_rom : int
            our chosen "reduced-order degrees of freedom" ("n_rom"),
            can be set to different from n_rom-true.
        hf_data :
            High-fidelity data.
        has_neumann : bool
            Does the problem have Neumann boundary conditions.
        has_non_homo_dirichlet : bool
            Does the problem have non homogeneous Dirichlet boundary conditions.
        has_non_homo_neumann : bool
            Does the problem have non homogeneous Neumann boundary conditions.

        Returns
        -------
        None.

        """
        self._compute_v(n_rom, hf_data.n_free)
        self._compute_rom_matrices_and_vectors(hf_data, has_neumann, has_non_homo_dirichlet, has_non_homo_neumann)

    def set_n_rom_max(self):
        """
        Set the maximum maximum reduced order dept

        Returns
        -------
        None.

        """
        if self.n_rom_cut == "rank":
            self.n_rom_max = self.solution_matrix_rank()
        else:
            self.n_rom_max = np.max(np.argwhere(self.sigma2_vec > self.n_rom_cut)) + 1

    def solution_matrix_rank(self):
        """
        Get the rank of the solution matrix

        Returns
        -------
        int
            the rank of the solution matrix.

        """
        return np.linalg.matrix_rank(self.s_mat)
