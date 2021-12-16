# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import os
import warnings
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import spsolve

from ._hf_data_class import HighFidelityData
from ._plotting import plot_singular_values, plot_relative_information_content, plot_mesh, plot_displacement, \
    plot_von_mises
from ._pod import pod_with_energy_norm, get_e_young_nu_poisson_mat
from ._rb_data_class import ReducedOrderData
from ._save_and_load import hf_save, rb_save, rb_from_files, hf_from_files
from ._solution_function_class import SolutionFunctionValues2D
from ._stress_recovery import get_von_mises_stress, get_nodal_stress
from .default_constants import file_names_dict, eps_pod, e_young_range, nu_poisson_range, rb_grid, pod_sampling_mode, \
    n_rom_cut
from .exceptions import IsNotAssembledError, PodNotComputedError, CanNotForceNromError, DirectoryDoesNotExistsError, \
    MissingInputFunctionPointerError, LinearElasticity2DProblemNotSolved, PlateLimitsNotFoundError, \
    CanNotComputeSolutionMatrixRankError
from .helpers import compute_a, VectorizedFunction2D


class LinearElasticity2DProblem:
    DEFAULT_FILE_NAMES_DICT = file_names_dict

    def __init__(self):
        """
        Setup

        Returns
        -------
        None.

        """
        # body force function
        self._f_func_non_vec = None
        # prescribed traction force function on Neumann boundary
        self._neumann_bc_func_non_vec = None
        # prescribed displacement on Dirichlet boundary
        self._dirichlet_bc_func_non_vec = None
        # vectorized functions of the functions above
        self._f_func_vec = None
        self._neumann_bc_func_vec = None
        self._dirichlet_bc_func_vec = None
        # is the body force function not zero for all x,y
        self._f_func_is_not_zeros = True
        # do we have non homogeneous Dirichlet conditions
        self._has_non_homo_dirichlet = False
        # do we have Neumann conditions
        self._has_neumann = False
        # do we have non homogeneous Neumann conditions
        self._has_non_homo_neumann = True
        # are the high-fidelity matrices and vectors assembled, 
        # and the free parts extracted 
        self._is_assembled_and_free = False
        # high-fidelity data
        self._hf_data = HighFidelityData()
        # reduced-order data
        self._rb_data = ReducedOrderData()
        # high-fidelity solution
        self._uh = SolutionFunctionValues2D()
        # reduced-order recovered solution
        self._uh_rom = SolutionFunctionValues2D()
        # is pod computed, such that the reduced-order data exists
        self._is_pod_computed = False
        # is the problem loaded from saved files
        self._is_form_files = False

    def _hf_assemble(self):
        """
        Assemble the high-fidelity matrices and load vectors

        Returns
        -------
        None.

        """
        # assemble the matrices a1 and a2, and the body force load vector
        self._hf_data.hf_assemble_a1_a2_f(self._f_func_vec, self._f_func_is_not_zeros)
        # assemble neumann load vector if we have Neumann conditions, else only have Dirichlet conditions
        if self._has_neumann:
            self._hf_data.edges()
            if self._has_non_homo_neumann:
                self._hf_data.hf_assemble_f_neumann(self._neumann_bc_func_vec)
        else:
            self._hf_data.dirichlet_edge = self._hf_data.edge
        # set free and Dirichlet indexes
        self._hf_data.compute_free_and_expanded_edges(self._has_neumann, self._has_non_homo_neumann)
        # lifting function if needed
        if self._has_non_homo_dirichlet:
            self._hf_data.compute_lifting_function_on_edge(self._dirichlet_bc_func_vec)
        # the high-fidelity matrices and vectors are assembled, 
        # and the free parts are extracted 
        self._is_assembled_and_free = True

    def compute_a_free(self, e_young, nu_poisson):
        """
        Compute the matrix a_free from a1_free and a2_free, given a Young's module and poisson ratio

        Parameters
        ----------
        e_young : float
            Young's module.
        nu_poisson : float
            poisson ratio.

        Raises
        ------
        IsNotAssembledError
            if the matrices and load vectors are not assembled.

        Returns
        -------
        scipy.sparse.dox_matrix, np.array
            the free matrix a_free.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return compute_a(e_young, nu_poisson, self._hf_data.a1_free, self._hf_data.a2_free)

    def compute_a_full(self, e_young, nu_poisson):
        """
        Compute the matrix a_full from a1_full and a2_full, given a Young's module and poisson ratio

        Parameters
        ----------
        e_young : float
            Young's module.
        nu_poisson : float
            poisson ratio.

        Raises
        ------
        IsNotAssembledError
            if the matrices and load vectors are not assembled.

        Returns
        -------
        scipy.sparse.dox_matrix, np.array
            the full matrix a_full.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return compute_a(e_young, nu_poisson, self._hf_data.a1_full, self._hf_data.a2_full)

    def _compute_a_dirichlet(self, e_young, nu_poisson):
        """
        Compute the matrix a_dirichlet from a1_dirichlet and a2_dirichlet, given a Young's module and poisson ratio

        Parameters
        ----------
        e_young : float
            Young's module.
        nu_poisson : float
            poisson ratio.

        Raises
        ------
        IsNotAssembledError
            if the matrices and load vectors are not assembled.

        Returns
        -------
        scipy.sparse.dox_matrix, np.array
            the dirichlet matrix a_dirichlet.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return compute_a(e_young, nu_poisson, self._hf_data.a1_dirichlet, self._hf_data.a2_dirichlet)

    def compute_f_load_free(self, e_young=None, nu_poisson=None):
        """
        Compute the load vector

        Parameters
        ----------
        e_young : float, optional
            Young's module, needed for Dirichlet load vector in the non homogeneous case. The default is None.
        nu_poisson : float, optional
            poisson ratio, needed for Dirichlet load vector in the non homogeneous case. The default is None.

        Raises
        ------
        ValueError
            if we have non homogeneous Dirichlet conditions, but either e_young or nu_poisson is none.

        Returns
        -------
        f_load : np.array
            the load vector.

        """
     
        # copy the body force load vector
        f_load = self._hf_data.f_load_lv_free.copy()
        # add the neumann load vector if it exists
        if self._has_neumann and self._has_non_homo_neumann:
            f_load += self._hf_data.f_load_neumann_free
        # compute and add the dirichlet load vector if it exists
        if self._has_non_homo_dirichlet:
            if e_young is None or nu_poisson is None:
                raise ValueError("e_young and/or nu_poisson are not given, needed for non homo. dirichlet conditions.")
            f_load -= self._compute_a_dirichlet(e_young, nu_poisson) @ self._hf_data.rg
        return f_load

    def compute_a_free_rom(self, e_young, nu_poisson):
        """
        Compute the matrix a_free_rom from a1_free_rom and a2_free_rom, given a Young's module and poisson ratio

        Parameters
        ----------
        e_young : float
            Young's module.
        nu_poisson : float
            poisson ratio.

        Raises
        ------
        PodNotComputedError
            if the reduced-order data does not exist since the pod is not computed.

        Returns
        -------
        scipy.sparse.dox_matrix, np.array
            the dirichlet matrix a_rom.

        """
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not solve.")
        return compute_a(e_young, nu_poisson, self._rb_data.a1_free_rom, self._rb_data.a2_free_rom)

    def _compute_f_dirichlet_rom(self, e_young, nu_poisson):
        """
        Compute the load f_dirichlet_rom from f1_dirichlet_rom and f2_dirichlet_rom,
        given a Young's module and poisson ratio

        Parameters
        ----------
        e_young : float
            Young's module.
        nu_poisson : float
            poisson ratio.

        Raises
        ------
        PodNotComputedError
            if the reduced-order data does not exist since the pod is not computed.

        Returns
        -------
        scipy.sparse.dox_matrix, np.array
            the dirichlet load f_dirichlet_rom.

        """
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not solve.")
        return compute_a(e_young, nu_poisson, self._rb_data.f1_dirichlet_rom, self._rb_data.f2_dirichlet_rom)

    def compute_f_load_rom(self, e_young=None, nu_poisson=None):
        """
        Compute the rom load vector

        Parameters
        ----------
        e_young : float, optional
            Young's module, needed for Dirichlet load vector in the non homogeneous case. The default is None.
        nu_poisson : float, optional
            poisson ratio, needed for Dirichlet load vector in the non homogeneous case. The default is None.

        Raises
        ------
        PodNotComputedError
            if the reduced-order data does not exist since the pod is not computed.
        ValueError
            if we have non homogeneous Dirichlet conditions, but either e_young or nu_poisson is none.

        Returns
        -------
        f_load_rom : np.array
            the rom load vector.

        """
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not solve.")
        # compute the load f for the rb solution
        # copy the body force load vector
        f_load_rom = self._rb_data.f_load_lv_free_rom.copy()
        # add the neumann load vector if it exists
        if self._has_neumann and self._has_non_homo_neumann:
            f_load_rom += self._rb_data.f_load_neumann_free_rom
        # compute and add the dirichlet load vector if it exists
        if self._has_non_homo_dirichlet:
            if e_young is None or nu_poisson is None:
                raise ValueError("e_young and/or nu_poisson are not given, needed for non homo. dirichlet conditions.")
            f_load_rom -= self._compute_f_dirichlet_rom(e_young, nu_poisson)
        return f_load_rom

    def hfsolve(self, e_young, nu_poisson, print_info=True):
        """
        Solve the High-fidelity system given a Young's module and poisson ratio

        Parameters
        ----------
        e_young : float, np.ndarray
            Young's module.
        nu_poisson : float, np.ndarray
            poisson ratio.
        print_info : bool, optional
            print solver info. The default is True.

        Returns
        -------
        None.

        """
        # compute a and convert to csr
        a = self.compute_a_free(e_young, nu_poisson).tocsr()
        f_load = self.compute_f_load_free(e_young, nu_poisson)  # e_young, nu_poisson needed if non homo. dirichlet
        # initialize uh
        uh = np.zeros(self._hf_data.n_full)
        start_time = perf_counter()
        # solve system
        uh[self._hf_data.expanded_free_index] = spsolve(a, f_load)
        if print_info:
            print("Solved a @ uh = f_load in {:.6f} sec".format(perf_counter() - start_time))
        if self._has_non_homo_dirichlet:
            uh[self._hf_data.expanded_dirichlet_edge_index] = self._hf_data.rg
        # set uh, and save it in a nice way.
        self._uh = SolutionFunctionValues2D.from_1x2n(uh)
        self._uh.set_e_young_and_nu_poisson(e_young, nu_poisson)
        if print_info:
            print("Get solution by the property uh, uh_free or uh_full of the class.\n" +
                  "The property uh, extra properties values, x and y are available.")

    def solve(self, e_young, nu_poisson, print_info=True):
        """
        The default solver, solves the hih-fidelity system given a Young's module and poisson ratio

        Parameters
        ----------
        e_young : float, np.ndarray
            Young's module.
        nu_poisson : float, np.ndarray
            poisson ratio.
        print_info : bool, optional
            print solver info. The default is True.

        Returns
        -------
        None.

        """
        # set normal solver as the high_fidelity solver
        self.hfsolve(e_young, nu_poisson, print_info)

    def rbsolve(self, e_young, nu_poisson, n_rom=None, print_info=True):
        """
        Solve the reduced-order system given a Young's module and poisson ratio

        Parameters
        ----------
        e_young : float, np.ndarray
            Young's module.
        nu_poisson : float, np.ndarray
            poisson ratio.
        n_rom : int, optional
            if set, the desired pod dept, else use n_rom-true. The default is None.
        print_info : bool, optional
            print solver info. The default is True.
        
        Raises
        ------
        PodNotComputedError
            if the reduced-order data does not exist since the pod is not computed.
        CanNotForceNromError
            if the problem comes form saved files and n_rom is not None.
            Setting n_rom not supported for problem from saved files.

        Returns
        -------
        None.

        """
        
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not solve.")
        if self._is_form_files and n_rom is not None:
            raise CanNotForceNromError("Can not force n_rom. Not implemented for LinearElasticity2DProblem for files.")
        # set n_rom to n_rom-true if it is None
        if n_rom is None:
            n_rom = self._rb_data.n_rom
        # compute the rom matrices and load vectors if n_rom is different from the last used n_rom n_rom_last
        if n_rom != self._rb_data.last_n_rom and not self._is_form_files:
            self._rb_data.compute_rb_matrices_and_vectors(n_rom, self._hf_data, self._has_neumann,
                                                          self._has_non_homo_dirichlet, self._has_non_homo_neumann)
        # set last n_rom
        self._rb_data.last_n_rom = self._rb_data.n_rom
        # compute the matrix a and the load vector
        a_rom = self.compute_a_free_rom(e_young, nu_poisson)
        f_load_rom = self.compute_f_load_rom(e_young, nu_poisson)  # e_young, nu_poisson needed if non homo. dirichlet
        # initialize uh
        uh_rom = np.zeros(self._hf_data.n_full)
        start_time = perf_counter()
        # solve and project rb solution
        uh_rom[self._hf_data.expanded_free_index] = self._rb_data.v @ np.linalg.solve(a_rom, f_load_rom)
        if print_info:
            print("Solved a_rom @ uh_rom = f_load_rom in {:.6f} sec".format(perf_counter() - start_time))
        if self._has_non_homo_dirichlet:
            # lifting function
            uh_rom[self._hf_data.expanded_dirichlet_edge_index] = self._hf_data.rg
        # set uh_rom, save it in a nice way.
        self._uh_rom = SolutionFunctionValues2D.from_1x2n(uh_rom)
        self._uh_rom.set_e_young_and_nu_poisson(e_young, nu_poisson)
        if print_info:
            print("Get solution by the property uh_rom, uh_rom_free or uh_rom_full of the class.\n" +
                  "The property uh_rom, extra properties values, x and y are available.")

    def build_rb_model(self, grid=rb_grid, mode=pod_sampling_mode, e_young_range=e_young_range,
                       nu_poisson_range=nu_poisson_range, eps_pod=eps_pod, n_rom_cut=n_rom_cut,
                       print_info=True):
        """
        Build a reduced-order model, i.e compute pod and the  rom matrices and load vectors

        Parameters
        ----------
        grid : int, tuple, list optional
            The grid size for the sampling, if int us it for both parameters.
            If None set to rb_grid from default_constants
            The default is None.
        mode : str, optional
            sampling mode.
            If None set to pod_sampling_mode from default_constants
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
        n_rom_cut : int, str, optional
            value to cut n_rom at, may be string "rank" fro the rank of the solution matrix.
            If None set to n_rom_cut from default_constants
            The default is None.
        print_info : bool, optional
            print builder info. The default is True.

        Raises
        ------
        NotImplementedError
            if the functions is called on a problem from saved files.

        Returns
        -------
        None.

        """
        if self._is_form_files:
            error_text = "Computing the reduced basis form saved data gives different results, " \
                         + "and is therefore not implemented. (Most likely because instability in " \
                         + "singular_values_squared computation in POD algorithm)."
            raise NotImplementedError(error_text)
        # set the parameters for building the reduce model
        self._rb_data.set_rb_model_params(grid, e_young_range, nu_poisson_range, eps_pod, mode, n_rom_cut)
        start_time = perf_counter()
        # compute a reduced model by the POD algorithm using the energy norm.
        pod_with_energy_norm(self, self._rb_data)
        # compute the rb matrices and vectors
        self._rb_data.compute_rb_matrices_and_vectors(self._rb_data.n_rom, self._hf_data, self._has_neumann,
                                                      self._has_non_homo_dirichlet, self._has_non_homo_neumann)
        # save the last n_rom
        self._rb_data.last_n_rom = self._rb_data.n_rom
        # do not allow computation of v for larger n_rom
        self._rb_data.set_n_rom_max()
        if print_info:
            print("Built reduced order model in {:.6f} sec".format(perf_counter() - start_time))
        # reset uh
        self._uh = self._uh = SolutionFunctionValues2D()
        self._is_pod_computed = True

    def error_a_rb(self, e_young, nu_poisson, n_rom=None, compute_again=False, print_info=False):
        """
        compute the error between the high-fidelity solution and 
        the recovered reduced-order solution in the energy norm
        given a Young's module and poisson ratio

        Parameters
        ----------
        e_young : float
            Young's module.
        nu_poisson : float
            poisson ratio.
        n_rom : int, optional
            if set, the desired pod dept, else use n_rom-true. The default is None.
        compute_again : bool, optional
            force to compute the solutions again, i.e do not use the snapshot matrix. 
            The default is False.
        print_info : bool, optional
            compute error info. The default is False.

        Raises
        ------
        CanNotForceNromError
            if the problem comes form saved files and n_rom is not None.
            Setting n_rom not supported for problem from saved files.

        Returns
        -------
        error_a : float
            the error between the high-fidelity solution and 
            the recovered reduced-order solution in the energy norm.

        """
        if self._is_form_files and n_rom is not None:
            raise CanNotForceNromError("Can not force n_rom. Not implemented for LinearElasticity2DProblem for files.")
        # set n_rom to n_rom-true if it is None
        if n_rom is None:
            n_rom = self._rb_data.n_rom
        
        if compute_again:
            # solve new
            self.hfsolve(e_young, nu_poisson, print_info=print_info)
            self.rbsolve(e_young, nu_poisson, n_rom=n_rom, print_info=print_info)
        else:
            # check if e_young and nu_young where used when solving the hf system
            if not self._uh.check_e_young_and_nu_poisson(e_young, nu_poisson):
                # check if the solution matrix does not exist 
                if self._rb_data.s_mat is None:
                    # solve new
                    self.hfsolve(e_young, nu_poisson, print_info=print_info)
                else:
                    # get solution from s_mat
                    e_nu_mat = self.e_young_nu_poisson_mat
                    index = np.argwhere((e_nu_mat[:, 0] == e_young) & (e_nu_mat[:, 1] == nu_poisson)).ravel()
                    # check if e_young and nu_poisson where not used in pod alfgorithm
                    if len(index) == 0:
                        # solve new
                        self.hfsolve(e_young, nu_poisson, print_info=print_info)
                    else:
                        # build from s_mat
                        uh = np.zeros(self._hf_data.n_full)
                        start_time = perf_counter()
                        uh[self._hf_data.expanded_free_index] = self._rb_data.s_mat[:, index].flatten()
                        if self._has_non_homo_dirichlet:
                            uh[self._hf_data.expanded_dirichlet_edge_index] = self._hf_data.rg
                        # set uh
                        self._uh = SolutionFunctionValues2D.from_1x2n(uh)
                        self._uh.set_e_young_and_nu_poisson(e_young, nu_poisson)
                        if print_info:
                            print("Loaded a @ uh = f_load from s_mat in {:.6f} sec".format(perf_counter() - start_time))
                            
            # check if e_young and nu_young where used when solving the rb system or n_rom is different from the last
            # used n_rom
            if not self._uh_rom.check_e_young_and_nu_poisson(e_young, nu_poisson) \
                    or n_rom != self._rb_data.last_n_rom:
                self.rbsolve(e_young, nu_poisson, n_rom=n_rom, print_info=print_info)
        # compute the error in the energy norm
        err = self.uh.flatt_values - self.uh_rom.flatt_values
        error_a = np.sqrt(err.T @ self.compute_a_full(e_young, nu_poisson) @ err)
        return error_a

    def f_func(self, x_vec, y_vec):
        """
        The vectorized body force function

        Parameters
        ----------
        x_vec : np.array
            array of x-values.
        y_vec : np.array
            array of y-values.

        Raises
        ------
        MissingInputFunctionPointerError
            if we have no body force function.

        Returns
        -------
        np.array
            function values in the (x,y)-values.

        """
        if self._f_func_vec is None:
            raise MissingInputFunctionPointerError(
                "f_func is not given, set it in the Linear Elasticity 2D Problem first.")
        return self._f_func_vec(x_vec, y_vec)

    def neumann_bc_func(self, x_vec, y_vec):
        """
        The vectorized prescribed traction function for the Neumann boundary

        Parameters
        ----------
        x_vec : np.array
            array of x-values.
        y_vec : np.array
            array of y-values.

        Raises
        ------
        MissingInputFunctionPointerError
            if we have no prescribed traction function for the Neumann boundary.

        Returns
        -------
        np.array
            function values in the (x,y)-values.

        """
        if self._neumann_bc_func_vec is None:
            raise MissingInputFunctionPointerError(
                "neumann_bc_func is not given, set it in the Linear Elasticity 2D Problem first.")
        return self._neumann_bc_func_vec(x_vec, y_vec)

    def dirichlet_bc_func(self, x_vec, y_vec):
        """
        The vectorized prescribed displacement function for the Dirichlet boundary

        Parameters
        ----------
        x_vec : np.array
            array of x-values.
        y_vec : np.array
            array of y-values.

        Raises
        ------
        MissingInputFunctionPointerError
            if we have no prescribed displacement function for the Dirichlet boundary.

        Returns
        -------
        np.array
            function values in the (x,y)-values.

        """
        if self._dirichlet_bc_func_vec is None:
            raise MissingInputFunctionPointerError(
                "dirichlet_bc_func is not given, set it in the Linear Elasticity 2D Problem first.")
        return self._dirichlet_bc_func_vec(x_vec, y_vec)

    def hf_nodal_stress(self, print_info=True):
        """
        Compute the high-fidelity nodal stress

        Parameters
        ----------
        print_info : bool, optional
            give user info. The default is False.

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the high-fidelity system has not been solved by hfsolve.

        Returns
        -------
        None.

        """
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved("High fidelity Linear Elasticity 2D Problem has not been solved.")
        get_nodal_stress(self._uh, self._hf_data.p, self._hf_data.tri)
        if print_info:
            print("Get nodal stress by the property uh.nodal_stress of the class.")

    def nodal_stress(self, print_info=True):
        """
        Default compute the stress, computes the high-fidelity nodal stress

        Parameters
        ----------
        print_info : bool, optional
            give user info. The default is False.

        Returns
        -------
        None.

        """
        self.hf_nodal_stress(print_info=print_info)

    def rb_nodal_stress(self, print_info=True):
        """
        Compute the reduced-order nodal stress

        Parameters
        ----------
        print_info : bool, optional
            give user info. The default is False.

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the reduced-order system has not been solved by rbsolve.

        Returns
        -------
        None.

        """
        if self._uh_rom.values is None:
            raise LinearElasticity2DProblemNotSolved("Reduced order Linear Elasticity 2D Problem has not been solved.")
        get_nodal_stress(self._uh_rom, self._hf_data.p, self._hf_data.tri)
        if print_info:
            print("Get nodal stress by the property uh_rom.nodal_stress of the class.")

    def hf_von_mises_stress(self, print_info=True):
        """
        Compute the high-fidelity von Mises stress

        Parameters
        ----------
        print_info : bool, optional
            give user info. The default is False.

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the high-fidelity system has not been solved by hfsolve.

        Returns
        -------
        None.

        """
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved("High fidelity Linear Elasticity 2D Problem has not been solved.")
        if self._uh.nodal_stress is None:
            self.hf_nodal_stress(print_info=False)
        get_von_mises_stress(self._uh)
        if print_info:
            print("Get von Mises yield by the property uh.von_mises of the class.")

    def von_mises_stress(self, print_info=True):
        """
        Default compute the von Mises stress, computes the high-fidelity von Mises stress

        Parameters
        ----------
        print_info : bool, optional
            give user info. The default is False.

        Returns
        -------
        None.

        """
        self.hf_von_mises_stress(print_info=print_info)

    def rb_von_mises_stress(self, print_info=True):
        """
        Compute the reduced-order von Mises stress

        Parameters
        ----------
        print_info : bool, optional
            give user info. The default is False.

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the reduced-order system has not been solved by rbsolve.

        Returns
        -------
        None.

        """
        if self._uh_rom.values is None:
            raise LinearElasticity2DProblemNotSolved("Reduced order Linear Elasticity 2D Problem has not been solved.")
        if self._uh_rom.nodal_stress is None:
            self.rb_nodal_stress(print_info=False)
        get_von_mises_stress(self._uh_rom)
        if print_info:
            print("Get von Mises yield by the property uh_rom.von_mises of the class.")

    def plot_mesh(self):
        """
        Plot the used mesh

        Raises
        ------
        IsNotAssembledError
            if the high-fidelity matrices and vectors are not assembled.

        Returns
        -------
        None.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled, do not have the triangulation data.")
        plot_mesh(self._hf_data.n, self._hf_data.p, self._hf_data.tri)

    def hf_plot_displacement(self, _solve_mode="hf"):
        """
        Plot the high-fidelity displacement

        Parameters
        ----------
        _solve_mode : str, optional
            high-fidelity ("hf"), reduced-order ("rb") or nothing ("") displacement text. The default is "hf".

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the high-fidelity system has not been solved by hfsolve.

        Returns
        -------
        None.

        """
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved("High fidelity Linear Elasticity 2D Problem has not been solved.")
        plot_displacement(self._uh, self._hf_data.n, self._hf_data.p, self._hf_data.tri, solve_mode=_solve_mode)

    def plot_displacement(self):
        """
        Default plot displacement, plots high-fidelity displacement

        Returns
        -------
        None.

        """
        self.hf_plot_displacement(_solve_mode="")

    def rb_plot_displacement(self, _solve_mode="rb"):
        """
        Plot the reduced-order displacement

        Parameters
        ----------
        _solve_mode : str, optional
            high-fidelity ("hf"), reduced-order ("rb") or nothing ("") displacement text. The default is "rb".

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the reduced-order system has not been solved by rbsolve.

        Returns
        -------
        None.

        """
        if self._uh_rom.values is None:
            raise LinearElasticity2DProblemNotSolved("Reduced order Linear Elasticity 2D Problem has not been solved.")
        plot_displacement(self._uh_rom, self._hf_data.n, self._hf_data.p, self._hf_data.tri, solve_mode=_solve_mode)

    def hf_plot_von_mises(self, _solve_mode="hf", levels=None):
        """
        Plot the high-fidelity von Mises stress

        Parameters
        ----------
        _solve_mode : str, optional
            high-fidelity ("hf"), reduced-order ("rb") or nothing ("") von Mises text. The default is "hf".
        levels : np.array, optional
            array for the color levels, default 25 levels between 0 and max(von_mises). The default is None.

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the high-fidelity system has not been solved by hfsolve.

        Returns
        -------
        None.

        """
        if self._uh.von_mises is None:
            self.hf_von_mises_stress(print_info=False)
        plot_von_mises(self._uh, self._hf_data.n, self._hf_data.p, self._hf_data.tri,
                       solve_mode=_solve_mode, levels=levels)

    def plot_von_mises(self, levels=None):
        """
        Default plot von Mises stress, plots high-fidelity von Mises stress

        Parameters
        ----------
        levels : np.array, optional
            array for the color levels, default 25 levels between 0 and max(von_mises). The default is None.

        Returns
        -------
        None.

        """
        self.hf_plot_von_mises(_solve_mode="", levels=levels)

    def rb_plot_von_mises(self, _solve_mode="rb", levels=None):
        """
        Plot the reduced-order von Mises stress

        Parameters
        ----------
        _solve_mode : str, optional
            high-fidelity ("hf"), reduced-order ("rb") or nothing ("") von Mises text. The default is "rb".
        levels : np.array, optional
            array for the color levels, default 25 levels between 0 and max(von_mises). The default is None.

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the reduced-order system has not been solved by rbsolve.

        Returns
        -------
        None.

        """
        if self._uh_rom.von_mises is None:
            self.rb_von_mises_stress(print_info=False)
        plot_von_mises(self._uh_rom, self._hf_data.n, self._hf_data.p, self._hf_data.tri,
                       solve_mode=_solve_mode, levels=levels)

    def plot_pod_singular_values(self):
        """
        Plot the singular values of the from the pod algorithm

        Raises
        ------
        PodNotComputedError
            if the reduced-order matrices and vectors are not computed, by the pod algorithm.

        Returns
        -------
        None.

        """
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not return.")
        plot_singular_values(self._rb_data.sigma2_vec)

    def plot_pod_relative_information_content(self):
        """
        Plot the relative information content of the from the pod algorithm

        Raises
        ------
        PodNotComputedError
            if the reduced-order matrices and vectors are not computed, by the pod algorithm.

        Returns
        -------
        None.

        """
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not return.")
        plot_relative_information_content(self._rb_data.sigma2_vec, self._rb_data.n_rom)

    def save(self, directory_path):
        """
        Save the high-fidelity data and the reduced-order data (if it exists)

        Parameters
        ----------
        directory_path : str
            path to directory to save in.

        Raises
        ------
        DirectoryDoesNotExistsError
            If directory does not exit.

        Returns
        -------
        None.

        """
        if not os.path.isdir(directory_path):
            raise DirectoryDoesNotExistsError(f"Directory {directory_path} does not exist")

        hf_save(self._hf_data, directory_path, self._has_neumann, self._has_non_homo_dirichlet,
                self._has_non_homo_neumann, default_file_names_dict=self.DEFAULT_FILE_NAMES_DICT)

        if self._is_pod_computed:
            if self._rb_data.last_n_rom != self._rb_data.n_rom:
                self._rb_data.compute_rb_matrices_and_vectors(self._rb_data.n_rom, self._hf_data, self._has_neumann,
                                                              self._has_non_homo_dirichlet, self._has_non_homo_neumann)
            rb_save(self._hf_data.n, self._rb_data, directory_path, self._has_neumann, self._has_non_homo_dirichlet,
                    self._has_non_homo_neumann, default_file_names_dict=self.DEFAULT_FILE_NAMES_DICT)

    def _set_vec_functions(self):
        """
        Set the vectorized body force function, 
        the vectorized prescribed traction function for the Neumann boundary and 
        the vectorized prescribed displacement function for the Dirichlet boundary

        Returns
        -------
        None.

        """
        # the body force function
        self._f_func_vec = VectorizedFunction2D(self._f_func_non_vec)
        # the prescribed traction function for the Neumann boundary
        if self._neumann_bc_func_non_vec is not None:
            self._neumann_bc_func_vec = VectorizedFunction2D(self._neumann_bc_func_non_vec)
            self._has_neumann = True
        # the prescribed displacment function for the Dirichlet boundary
        if self._dirichlet_bc_func_non_vec is not None:
            self._dirichlet_bc_func_vec = VectorizedFunction2D(self._dirichlet_bc_func_non_vec)
            self._has_non_homo_dirichlet = True

    @classmethod
    def from_saves(cls, n, directory_path, rb_warnings=True, print_info=True):
        """
        Get a problem from saved files

        Parameters
        ----------
        n : int
            number of elements along the axes.
        directory_path : str
            path to directory to get files from.
        rb_warnings : bool, optional
            reduced-order warnings. The default is True.
        print_info : bool, optional
            user info. The default is True.

        Raises
        ------
        DirectoryDoesNotExistsError
            DESCRIPTION.

        Returns
        -------
        problem : LinearElasticity2DProblem
            LinearElasticity2DProblem from saved files.

        """

        if not os.path.isdir(directory_path):
            raise DirectoryDoesNotExistsError(f"Directory {directory_path} does not exist")

        problem = cls()
        # number of nodes along the axes
        problem._hf_data.n = n + 1

        start_time = perf_counter()
        # get the high-fidelity data
        problem._has_neumann, problem._has_non_homo_dirichlet, problem._has_non_homo_neumann \
            = hf_from_files(problem._hf_data, directory_path)
        if print_info:
            print("Loaded the high fidelity data in {:.6f} sec".format(perf_counter() - start_time))
        problem._is_assembled_and_free = True

        start_time = perf_counter()
        # get the reduced-order data if it exists
        problem._is_pod_computed = rb_from_files(problem._hf_data.n, problem._rb_data, directory_path, rb_warnings)
        if problem._is_pod_computed:
            if print_info:
                print("Loaded the reduced order data in {:.6f} sec".format(perf_counter() - start_time))
        problem._is_form_files = True
        return problem

    @classmethod
    def from_functions(cls, n, f_func, dirichlet_bc_func=None, get_dirichlet_edge_func=None,
                       neumann_bc_func=None, plate_limits=None, print_info=True):
        """
        Get a problem from functions

        Parameters
        ----------
        n : int
            number of elements along the axes.
        f_func : function, int
            body force function, 0 means no body force.
        dirichlet_bc_func : function, optional
            prescribed displacement function for the Dirichlet boundary. The default is None.
        get_dirichlet_edge_func : function, optional
            function to get the Dirichlet boundary, 
            if None assume that we only have a Dirichlet boundary. The default is None.
        neumann_bc_func : function, optional
            prescribed traction function for the Neumann boundary. The default is None.
        plate_limits : tuple, optional
            the plate limits. The default is None.
        print_info : TYPE, optional
            user info. The default is True.

        Raises
        ------
        MissingInputFunctionPointerError
            if we have neumann and non homo. dirichlet conditions,
            but no function giving the neumann and dirichlet edges.

        Returns
        -------
        problem : LinearElasticity2DProblem
            LinearElasticity2DProblem from functions.

        """
        def default_func(x, y):
            return 0, 0

        problem = cls()
        # number of nodes along the axes
        problem._hf_data.n = n + 1
        # set the functions
        problem._f_func_non_vec = f_func
        problem._neumann_bc_func_non_vec = neumann_bc_func
        problem._dirichlet_bc_func_non_vec = dirichlet_bc_func
        # check the functions
        if f_func == 0:
            problem._f_func_non_vec = default_func
            problem._f_func_is_not_zeros = False
        # check get_dirichlet_edge function
        if get_dirichlet_edge_func is not None and neumann_bc_func is None:
            problem._neumann_bc_func_non_vec = default_func
            problem._has_non_homo_neumann = False
        # set it
        problem._hf_data.get_dirichlet_edge_func = get_dirichlet_edge_func
        # set the vectorized functions
        problem._set_vec_functions()
        # check Neumann and Dirichlet
        if problem._has_neumann and problem._hf_data.get_dirichlet_edge_func is None \
                and problem._has_non_homo_dirichlet:
            error_text = "Have neumann and non homo. dirichlet conditions, " \
                         + "but no function giving the neumann and dirichlet edges. "
            raise MissingInputFunctionPointerError(error_text)
        # set plate limits, if None use default of high-fidelity data
        if plate_limits is not None:
            problem._hf_data.plate_limits = plate_limits

        start_time = perf_counter()
        # assemble the high-fidelity system
        problem._hf_assemble()
        if print_info:
            print("Assembled matrices and load-vectors in {:.6f} sec".format(perf_counter() - start_time))
        return problem

    @property
    def uh(self):
        """
        High-fidelity solution

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the high-fidelity system has not been solved.

        Returns
        -------
        SolutionFunctionValues2D
            High-fidelity solution.

        """
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved(
                "High fidelity Linear Elasticity 2D Problem has not been solved, can not return uh.")
        return self._uh

    @property
    def uh_free(self):
        """
        the free flattened part of the high-fidelity solution

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the high-fidelity system has not been solved.

        Returns
        -------
        np.array
            the free flattened part of the high-fidelity solution.

        """
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved(
                "High fidelity Linear Elasticity 2D Problem has not been solved, can not return uh_free.")
        return self._uh.flatt_values[self._hf_data.expanded_free_index]

    @property
    def uh_full(self):
        """
        the full flattened of the high-fidelity solution

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the high-fidelity system has not been solved.

        Returns
        -------
        np.array
            the full flattened of the high-fidelity solution.

        """
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved(
                "High fidelity Linear Elasticity 2D Problem has not been solved, can not return uh_full.")
        return self._uh.flatt_values

    @property
    def uh_rom(self):
        """
        Recovered reduced-order solution

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the reduced-order system has not been solved.

        Returns
        -------
        SolutionFunctionValues2D
            High-fidelity solution.

        """
        if self._uh_rom.values is None:
            raise LinearElasticity2DProblemNotSolved(
                "Reduced order Linear Elasticity 2D Problem has not been solved, can not return uh_free.")
        return self._uh_rom

    @property
    def uh_rom_free(self):
        """
        the free flattened part of the recovered reduced-order solution

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the reduced-order system has not been solved.

        Returns
        -------
        np.array
            the free flattened part of the recovered reduced-order solution.

        """
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved(
                "Reduced order Linear Elasticity 2D Problem has not been solved, can not return uh_free.")
        return self._uh_rom.flatt_values[self._hf_data.expanded_free_index]

    @property
    def uh_rom_full(self):
        """
        the full flattened of the recovered reduced-order solution

        Raises
        ------
        LinearElasticity2DProblemNotSolved
            if the reduced-order system has not been solved.

        Returns
        -------
        np.array
            the full flattened of the recovered reduced-order solution.

        """
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved(
                "Reduced order Linear Elasticity 2D Problem has not been solved, can not return uh_free.")
        return self._uh_rom.flatt_values

    @property
    def default_file_names(self):
        """
        get the default file names dictionary for saving the files

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.DEFAULT_FILE_NAMES_DICT

    @property
    def dirichlet_edge(self):
        """
        Get the Dirichlet edge

        Raises
        ------
        IsNotAssembledError
            if the high-fidelity system is not assembled.

        Returns
        -------
        np.array
            the Dirichlet edge.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.dirichlet_edge

    @property
    def neumann_edge(self):
        """
        Get the Neumann edge

        Raises
        ------
        IsNotAssembledError
            if the high-fidelity system is not assembled.

        Returns
        -------
        np.array
            the Neumann edge.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.neumann_edge

    @property
    def free_index(self):
        """
        Get the free index (expanded)

        Raises
        ------
        IsNotAssembledError
            if the high-fidelity system is not assembled.

        Returns
        -------
        np.array
            the free index (expanded).

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.expanded_free_index

    @property
    def dirichlet_edge_index(self):
        """
        Get the Dirichlet edge index (expanded)

        Raises
        ------
        IsNotAssembledError
            if the high-fidelity system is not assembled.

        Returns
        -------
        np.array
            the Dirichlet edge index (expanded).

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.expanded_dirichlet_edge_index

    @property
    def p(self):
        """
        Get the mesh points p, 
        Nodal points, (x,y)-coordinates for point i given in row i.
       

        Raises
        ------
        IsNotAssembledError
            if the high-fidelity system is not assembled.

        Returns
        -------
        np.array
            p.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.p

    @property
    def x(self):
        """
        Get the mesh x-values of the points p
       

        Raises
        ------
        IsNotAssembledError
            if the high-fidelity system is not assembled.

        Returns
        -------
        np.array
            x.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.p[:, 0]

    @property
    def y(self):
        """
        Get the mesh y-values of the points p
       

        Raises
        ------
        IsNotAssembledError
            if the high-fidelity system is not assembled.

        Returns
        -------
        np.array
            y.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.p[:, 1]

    @property
    def tri(self):
        """
        Get the mesh triangulation tri, 
        Elements. Index to the three corners of element i given in row i.
       

        Raises
        ------
        IsNotAssembledError
            if the high-fidelity system is not assembled.

        Returns
        -------
        np.array
            tri.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.tri

    @property
    def edge(self):
        """
        Get the mesh edge, 
        Index list of all nodal points on the outer edge.
       

        Raises
        ------
        IsNotAssembledError
            if the high-fidelity system is not assembled.

        Returns
        -------
        np.array
            edge.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.edge

    @property
    def n(self):
        """
        Get the number of elements along the axes
       

        Raises
        ------
        IsNotAssembledError
            if the high-fidelity system is not assembled.

        Returns
        -------
        np.array
            number of elements along the axes.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.n - 1

    @property
    def n_full(self):
        """
        Get the total number of of degrees in the high-fidelity system
       

        Raises
        ------
        IsNotAssembledError
            if the high-fidelity system is not assembled.

        Returns
        -------
        np.array
            the total number of of degrees in the high-fidelity system.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.n_full

    @property
    def n_free(self):
        """
        Get the number of of degrees of freedom in the high-fidelity system
       

        Raises
        ------
        IsNotAssembledError
            if the high-fidelity system is not assembled.

        Returns
        -------
        np.array
            number of of degrees of freedom in the high-fidelity system.

        """
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.n_free

    @property
    def plate_limits(self):
        """
        Get the plate_limits

        Raises
        ------
        PlateLimitsNotFoundError
            if the plate limits are None.

        Returns
        -------
        tuple
            the plate limits.

        """
        if self._hf_data.plate_limits is None:
            text = "Plate limits are None, get the Linear Elasticity 2D Problem from functions or " \
                   "saved matrices and vectors first."
            raise PlateLimitsNotFoundError(text)
        return self._hf_data.plate_limits

    @property
    def v(self):
        """
        Get the transformation matrix V

        Raises
        ------
        PodNotComputedError
            if the reduced-order matrices and load vectors are not computed, by pod algorithm.

        Returns
        -------
        np.array
            the transformation matrix V.

        """
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not return.")
        return self._rb_data.v

    @property
    def ns_rom(self):
        """
        Get the number of snapshots

        Raises
        ------
        PodNotComputedError
            if the reduced-order matrices and load vectors are not computed, by pod algorithm.

        Returns
        -------
        np.array
            the number of snapshots

        """
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not return.")
        return self._rb_data.ns_rom

    @property
    def n_rom(self):
        """
        Get the number of degrees of freedom in the reduced-order system

        Raises
        ------
        PodNotComputedError
            if the reduced-order matrices and load vectors are not computed, by pod algorithm.

        Returns
        -------
        np.array
            the number of degrees of freedom in the reduced-order system.

        """
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not return.")
        return self._rb_data.n_rom

    @property
    def e_young_range(self):
        """
        Get the Young's module range used, 
        if pod is not computed return reduced-order data default

        Returns
        -------
        tuple
            the Young's module range used.

        """
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return self._rb_data.e_young_range

    @property
    def nu_poisson_range(self):
        """
        Get the poisson ratio range used, 
        if pod is not computed return reduced-order data default

        Returns
        -------
        tuple
            the poisson ratio range used.

        """
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return self._rb_data.nu_poisson_range

    @property
    def eps_pod(self):
        """
        Get the POD tolerance used, 
        if pod is not computed return reduced-order data default

        Returns
        -------
        tuple
            the POD tolerance range used.

        """
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return self._rb_data.eps_pod

    @property
    def rb_grid(self):
        """
        Get the POD sampling grid used, 
        if pod is not computed return reduced-order data default

        Returns
        -------
        tuple
            the POD sampling grid range used.

        """
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return self._rb_data.rb_grid

    @property
    def pod_sampling_mode(self):
        """
        Get the POD sampling mode used, 
        if pod is not computed return reduced-order data default

        Returns
        -------
        str
            the POD sampling mode range used.

        """
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return self._rb_data.pod_sampling_mode

    @property
    def e_young_nu_poisson_mat(self):
        """
        Get the Young's module and poisson ratio matrix of combinations used, 
        if pod is not computed return reduced-order data default

        Returns
        -------
        np.array
            the Young's module and poisson ratio matrix of combinations used.

        """
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return get_e_young_nu_poisson_mat(self._rb_data)

    @property
    def n_rom_max(self):
        """
        Get the maximum pod dept used, 
        if pod is not computed return reduced-order data default

        Returns
        -------
        int, np.inf
            the maximum pod dept used.

        """
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return self._rb_data.n_rom_max

    @property
    def singular_values_squared_pod(self):
        """
        Get the singular values squared from the pod algorithm.

        Raises
        ------
        PodNotComputedError
            if the reduced-order matrices and load vectors are not computed, by pod algorithm.

        Returns
        -------
        np.array
            the singular values squared from the pod algorithm.

        """
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not return.")
        return self._rb_data.sigma2_vec

    @property
    def singular_values_pod(self):
        """
        Get the positive singular values from the pod algorithm

        Raises
        ------
        PodNotComputedError
            if the reduced-order matrices and load vectors are not computed, by pod algorithm.

        Returns
        -------
        np.array
            the positive singular values from the pod algorithm.

        """
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not return.")
        arg0 = np.argwhere(self._rb_data.sigma2_vec >= 0)
        return np.sqrt(self._rb_data.sigma2_vec[arg0])

    @property
    def solution_matrix_rank(self):
        """
        Get the rank of the solution matrix

        Returns
        -------
        np.array
            the rank of the solution matrix.

        """
        if self._rb_data.s_mat is None:
            raise CanNotComputeSolutionMatrixRankError("Solution matrix is not computed, can not compute its rank.")
        else:
            return self._rb_data.solution_matrix_rank()
