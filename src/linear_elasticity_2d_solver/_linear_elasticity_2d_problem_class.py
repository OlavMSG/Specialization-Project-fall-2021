# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import os
import warnings
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import spsolve

from ._default_constants import FILE_NAMES_DICT, EPS_POD, E_YOUNG_RANGE, NU_POISSON_RANGE, RB_GRID, POD_MODE, N_ROM_CUT
from ._helpers import compute_a, VectorizedFunction2D
from ._hf_data_class import HighFidelityData
from ._plotting import plot_singular_values, plot_relative_information_content, plot_mesh, plot_displacement, \
    plot_von_mises
from ._pod import pod, get_e_young_nu_poisson_mat
from ._rb_data_class import ReducedOrderData
from ._save_and_load import hf_save, rb_save, rb_from_files, hf_from_files
from ._solution_function_class import SolutionFunctionValues2D
from ._stress_recovery import von_mises_yield
from .exceptions import IsNotAssembledError, PodNotComputedError, CanNotForceNromError, DirectoryDoesNotExistsError, \
    MissingInputFunctionPointerError, LinearElasticity2DProblemNotSolved, PlateLimitsNotFoundError


class LinearElasticity2DProblem:
    DEFAULT_FILE_NAMES_DICT = FILE_NAMES_DICT

    def __init__(self):
        self._f_func_non_vec = None
        self._neumann_bc_func_non_vec = None
        self._dirichlet_bc_func_non_vec = None
        self._f_func_vec = None
        self._neumann_bc_func_vec = None
        self._dirichlet_bc_func_vec = None

        self._has_non_homo_dirichlet = False
        self._has_neumann = False

        self._is_assembled_and_free = False

        self._hf_data = HighFidelityData()
        self._rb_data = ReducedOrderData()
        self._uh = SolutionFunctionValues2D()
        self._uh_rom = SolutionFunctionValues2D()

        self._is_pod_computed = False
        self._is_from_files = False

    def _hf_assemble(self):
        self._hf_data.hf_assemble_a1_a2_f(self._f_func_vec)

        if self._has_neumann:
            self._hf_data.hf_assemble_f_neumann(self._neumann_bc_func_vec)
        else:
            self._hf_data.dirichlet_edge = self._hf_data.edge
        # set indexes
        self._hf_data.compute_free_and_expanded_edges(self._has_neumann)

        if self._has_non_homo_dirichlet:
            self._hf_data.compute_lifting_function_on_edge(self._dirichlet_bc_func_vec)

        self._is_assembled_and_free = True

    def compute_a_free(self, e_young, nu_poisson):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return compute_a(e_young, nu_poisson, self._hf_data.a1_free, self._hf_data.a2_free)

    def compute_a_full(self, e_young, nu_poisson):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return compute_a(e_young, nu_poisson, self._hf_data.a1_full, self._hf_data.a2_full)

    def _compute_a_dirichlet(self, e_young, nu_poisson):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return compute_a(e_young, nu_poisson, self._hf_data.a1_dirichlet, self._hf_data.a2_dirichlet)

    def compute_f_load_free(self, e_young=None, nu_poisson=None):
        f_load = self._hf_data.f_load_lv_free.copy()
        if self._has_neumann:
            f_load += self._hf_data.f_load_neumann_free
        if self._has_non_homo_dirichlet:
            if e_young is None or nu_poisson is None:
                raise ValueError("e_young and/or nu_poisson are not given, needed for non homo. dirichlet conditions.")
            f_load -= self._compute_a_dirichlet(e_young, nu_poisson) @ self._hf_data.rg
        return f_load

    def compute_a_free_rom(self, e_young, nu_poisson):
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not solve.")
        return compute_a(e_young, nu_poisson, self._rb_data.a1_free_rom, self._rb_data.a2_free_rom)

    def _compute_a_dirichlet_rom(self, e_young, nu_poisson):
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not solve.")
        return compute_a(e_young, nu_poisson, self._rb_data.a1_dirichlet_rom, self._rb_data.a2_dirichlet_rom)

    def compute_f_load_rom(self, e_young=None, nu_poisson=None):
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not solve.")
        f_load_rom = self._rb_data.f_load_lv_free_rom.copy()
        if self._has_neumann:
            f_load_rom += self._rb_data.f_load_neumann_free_rom
        if self._has_non_homo_dirichlet:
            if e_young is None or nu_poisson is None:
                raise ValueError("e_young and/or nu_poisson are not given, needed for non homo. dirichlet conditions.")
            f_load_rom -= self._compute_a_dirichlet_rom(e_young, nu_poisson) @ self._hf_data.rg
        return f_load_rom

    def build_rb_model(self, grid=RB_GRID, mode=POD_MODE, e_young_range=E_YOUNG_RANGE,
                       nu_poisson_range=NU_POISSON_RANGE, eps_pod=EPS_POD, n_rom_cut=N_ROM_CUT):

        self._rb_data.set_rb_model_params(grid, e_young_range, nu_poisson_range, eps_pod, mode, n_rom_cut)

        start_time = perf_counter()
        pod(self, self._rb_data)
        # do not allow computation of v for larger n_rom
        self._rb_data.compute_rb_matrices_and_vectors(self._rb_data.n_rom, self._hf_data, self._has_neumann,
                                                      self._has_non_homo_dirichlet)
        self._rb_data.last_n_rom = self._rb_data.n_rom
        self._rb_data.set_n_rom_max()
        print("Built reduced order model in {:.6f} sec".format(perf_counter() - start_time))
        # reset uh
        self._uh = self._uh = SolutionFunctionValues2D()
        self._is_pod_computed = True

    def hfsolve(self, e_young, nu_poisson, print_info=True):
        # compute a and convert to csr
        a = self.compute_a_free(e_young, nu_poisson).tocsr()
        f_load = self.compute_f_load_free(e_young, nu_poisson)  # e_young, nu_poisson needed if non homo. dirichlet
        # initialize uh
        uh = np.zeros(self._hf_data.n_full)
        start_time = perf_counter()
        uh[self._hf_data.expanded_free_index] = spsolve(a, f_load)
        if print_info:
            print("Solved a @ uh = f_load in {:.6f} sec".format(perf_counter() - start_time))
        if self._has_non_homo_dirichlet:
            uh[self._hf_data.expanded_dirichlet_edge_index] = self._hf_data.rg
        # set uh
        self._uh = SolutionFunctionValues2D.from_1xn2(uh)
        self._uh.set_e_young_and_nu_poisson(e_young, nu_poisson)
        if print_info:
            print("Get solution by the property uh, uh_free or uh_full of the class.\n" +
                  "The property uh, extra properties values, x and y are available.")

    def solve(self, e_young, nu_poisson, print_info=True):
        # set normal solver as the high_fidelity solver
        self.hfsolve(e_young, nu_poisson, print_info)

    def rbsolve(self, e_young, nu_poisson, n_rom=None, print_info=True):
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not solve.")
        if self._is_from_files and n_rom is not None:
            raise CanNotForceNromError("Can not force n_rom. Not implemented for LinearElasticity2DProblem for files.")

        if n_rom is None:
            n_rom = self._rb_data.n_rom
        if n_rom != self._rb_data.last_n_rom and not self._is_from_files:
            self._rb_data.compute_rb_matrices_and_vectors(n_rom, self._hf_data, self._has_neumann,
                                                          self._has_non_homo_dirichlet)
        self._rb_data.last_n_rom = self._rb_data.n_rom
        a_rom = self.compute_a_free_rom(e_young, nu_poisson)
        f_load_rom = self.compute_f_load_rom(e_young, nu_poisson)  # e_young, nu_poisson needed if non homo. dirichlet
        # initialize uh
        uh_rom = np.zeros(self._hf_data.n_full)
        start_time = perf_counter()
        uh_rom[self._hf_data.expanded_free_index] = self._rb_data.v @ np.linalg.solve(a_rom, f_load_rom)
        if print_info:
            print("Solved a_rom @ uh_rom = f_load_rom in {:.6f} sec".format(perf_counter() - start_time))
        if self._has_non_homo_dirichlet:
            uh_rom[self._hf_data.expanded_dirichlet_edge_index] = self._hf_data.rg
        # set uh
        self._uh_rom = SolutionFunctionValues2D.from_1xn2(uh_rom)
        self._uh_rom.set_e_young_and_nu_poisson(e_young, nu_poisson)
        if print_info:
            print("Get solution by the property uh_rom, uh_rom_free or uh_rom_full of the class.\n" +
                  "The property uh_rom, extra properties values, x and y are available.")

    def error_a_rb(self, e_young, nu_poisson, n_rom=None, compute_again=False, print_info=False):
        if self._is_from_files and n_rom is not None:
            raise CanNotForceNromError("Can not force n_rom. Not implemented for LinearElasticity2DProblem for files.")
        if n_rom is None:
            n_rom = self._rb_data.n_rom
        if not self._uh.check_e_young_and_nu_poisson(e_young, nu_poisson) \
                or compute_again:
            if self._rb_data.s_mat is None:
                # solve new
                self.hfsolve(e_young, nu_poisson, print_info=print_info)
            else:
                # get solution from s_mat
                e_nu_mat = self.e_young_nu_poisson_mat
                index = np.argwhere((e_nu_mat[:, 0] == e_young) & (e_nu_mat[:, 1] == nu_poisson)).ravel()
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
                    self._uh = SolutionFunctionValues2D.from_1xn2(uh)
                    self._uh.set_e_young_and_nu_poisson(e_young, nu_poisson)
                    if print_info:
                        print("Loaded a @ uh = f_load from s_mat in {:.6f} sec".format(perf_counter() - start_time))

        if not self._uh_rom.check_e_young_and_nu_poisson(e_young, nu_poisson) \
                or n_rom != self._rb_data.last_n_rom \
                or compute_again:
            self.rbsolve(e_young, nu_poisson, n_rom=n_rom, print_info=print_info)

        err = self.uh.flatt_values - self.uh_rom.flatt_values
        error_a = np.sqrt(err.T @ self.compute_a_full(e_young, nu_poisson) @ err)
        return error_a

    def f_func(self, x_vec, y_vec):
        if self._f_func_vec is None:
            raise MissingInputFunctionPointerError(
                "f_func is not given, set it in the Linear Elasticity 2D Problem first.")
        return self._f_func_vec(x_vec, y_vec)

    def neumann_bc_func(self, x_vec, y_vec):
        if self._neumann_bc_func_vec is None:
            raise MissingInputFunctionPointerError(
                "neumann_bc_func is not given, set it in the Linear Elasticity 2D Problem first.")
        return self._neumann_bc_func_vec(x_vec, y_vec)

    def dirichlet_bc_func(self, x_vec, y_vec):
        if self._dirichlet_bc_func_vec is None:
            raise MissingInputFunctionPointerError(
                "dirichlet_bc_func is not given, set it in the Linear Elasticity 2D Problem first.")
        return self._dirichlet_bc_func_vec(x_vec, y_vec)

    def hf_von_mises_yield(self, print_info=True):
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved("High fidelity Linear Elasticity 2D Problem has not been solved.")
        von_mises_yield(self._uh, self._hf_data.p, self._hf_data.tri)
        if print_info:
            print("Get von Mises yield by the property uh.von_mises of the class.")

    def von_mises_yield(self, print_info=True):
        self.hf_von_mises_yield(print_info=print_info)

    def rb_von_mises_yield(self, print_info=True):
        if self._uh_rom.values is None:
            raise LinearElasticity2DProblemNotSolved("Reduced order Linear Elasticity 2D Problem has not been solved.")
        von_mises_yield(self._uh_rom, self._hf_data.p, self._hf_data.tri)
        if print_info:
            print("Get von Mises yield by the property uh_rom.von_mises of the class.")

    def plot_mesh(self):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        plot_mesh(self._hf_data.n, self._hf_data.p, self._hf_data.tri)

    def hf_plot_displacement(self, _solve_mode="hf"):
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved("High fidelity Linear Elasticity 2D Problem has not been solved.")
        plot_displacement(self._uh, self._hf_data.p, self._hf_data.tri, solve_mode=_solve_mode)

    def plot_displacement(self):
        self.hf_plot_displacement(_solve_mode="")

    def rb_plot_displacement(self, _solve_mode="rb"):
        if self._uh_rom.values is None:
            raise LinearElasticity2DProblemNotSolved("Reduced order Linear Elasticity 2D Problem has not been solved.")
        plot_displacement(self._uh_rom, self._hf_data.p, self._hf_data.tri, solve_mode=_solve_mode)

    def hf_plot_von_mises(self, _solve_mode="hf"):
        if self._uh.von_mises is None:
            self.hf_von_mises_yield(print_info=False)
        plot_von_mises(self._uh, self._hf_data.p, self._hf_data.tri, solve_mode=_solve_mode)

    def plot_von_mises(self):
        self.hf_plot_von_mises(_solve_mode="")

    def rb_plot_von_mises(self, _solve_mode="rb"):
        if self._uh_rom.von_mises is None:
            self.rb_von_mises_yield(print_info=False)
        plot_von_mises(self._uh_rom, self._hf_data.p, self._hf_data.tri, solve_mode=_solve_mode)

    def plot_pod_singular_values(self):
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not return.")
        plot_singular_values(self._rb_data.sigma2_vec)

    def plot_pod_relative_information_content(self):
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not return.")
        plot_relative_information_content(self._rb_data.sigma2_vec, self._rb_data.n_rom)

    def save(self, directory_path):
        if not os.path.isdir(directory_path):
            raise DirectoryDoesNotExistsError(f"Directory {directory_path} does not exist")

        hf_save(self._hf_data, directory_path, self._has_neumann, self._has_non_homo_dirichlet,
                default_file_names_dict=self.DEFAULT_FILE_NAMES_DICT)

        if self._is_pod_computed:
            if self._rb_data.last_n_rom != self._rb_data.n_rom:
                self._rb_data.compute_rb_matrices_and_vectors(self._rb_data.n_rom, self._hf_data, self._has_neumann,
                                                              self._has_non_homo_dirichlet)
            rb_save(self._hf_data.n, self._rb_data, directory_path, self._has_neumann, self._has_non_homo_dirichlet,
                    default_file_names_dict=self.DEFAULT_FILE_NAMES_DICT)

    @classmethod
    def from_saves(cls, n, directory_path, rb_warnings=True):

        if not os.path.isdir(directory_path):
            raise DirectoryDoesNotExistsError(f"Directory {directory_path} does not exist")

        problem = cls()
        problem._hf_data.n = n

        start_time = perf_counter()
        problem._has_neumann, problem._has_non_homo_dirichlet = hf_from_files(problem._hf_data, directory_path)
        print("Loaded the high fidelity data in {:.6f} sec".format(perf_counter() - start_time))
        problem._is_assembled_and_free = True

        start_time = perf_counter()
        rb_from_files(problem._hf_data.n, problem._rb_data, directory_path, rb_warnings)
        if problem._is_pod_computed:
            print("Loaded the reduced order data in {:.6f} sec".format(perf_counter() - start_time))
            problem._is_pod_computed = True
        problem._is_from_files = True
        return problem

    def _set_vec_functions(self):
        self._f_func_vec = VectorizedFunction2D(self._f_func_non_vec)
        if self._neumann_bc_func_non_vec is not None:
            self._neumann_bc_func_vec = VectorizedFunction2D(self._neumann_bc_func_non_vec)
            self._has_neumann = True
        if self._dirichlet_bc_func_non_vec is not None:
            self._dirichlet_bc_func_vec = VectorizedFunction2D(self._dirichlet_bc_func_non_vec)
            self._has_non_homo_dirichlet = True

    @classmethod
    def from_functions(cls, n, f_func, neumann_bc_func=None, get_dirichlet_edge_func=None,
                       dirichlet_bc_func=None, plate_limits=None):
        problem = cls()
        problem._hf_data.n = n
        problem._f_func_non_vec = f_func
        problem._neumann_bc_func_non_vec = neumann_bc_func
        problem._dirichlet_bc_func_non_vec = dirichlet_bc_func

        if get_dirichlet_edge_func is not None and neumann_bc_func is None:
            def default_neumann_bc_func(x, y):
                return 0, 0
            problem._neumann_bc_func_non_vec = default_neumann_bc_func

        problem._hf_data.get_dirichlet_edge_func = get_dirichlet_edge_func
        problem._set_vec_functions()

        if problem._has_neumann and problem._hf_data.get_dirichlet_edge_func is None \
                and problem._has_non_homo_dirichlet:
                error_text = "Have neumann and non homo. dirichlet conditions, " \
                             + "but no function giving the neumann and dirichlet edges. "
                raise MissingInputFunctionPointerError(error_text)

        if plate_limits is not None:
            problem._hf_data.plate_limits = plate_limits

        start_time = perf_counter()
        problem._hf_assemble()
        print("Assembled matrices and load-vectors in {:.6f} sec".format(perf_counter() - start_time))

        return problem

    @property
    def uh(self):
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved(
                "High fidelity Linear Elasticity 2D Problem has not been solved, can not return uh.")
        return self._uh

    @property
    def uh_free(self):
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved(
                "High fidelity Linear Elasticity 2D Problem has not been solved, can not return uh_free.")
        return self._uh.flatt_values[self._hf_data.expanded_free_index]

    @property
    def uh_full(self):
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved(
                "High fidelity Linear Elasticity 2D Problem has not been solved, can not return uh_full.")
        return self._uh.flatt_values

    @property
    def uh_rom(self):
        if self._uh_rom.values is None:
            raise LinearElasticity2DProblemNotSolved(
                "Reduced order Linear Elasticity 2D Problem has not been solved, can not return uh_free.")
        return self._uh_rom

    @property
    def uh_rom_free(self):
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved(
                "Reduced order Linear Elasticity 2D Problem has not been solved, can not return uh_free.")
        return self._uh_rom.flatt_values[self._hf_data.expanded_free_index]

    @property
    def uh_rom_full(self):
        if self._uh.values is None:
            raise LinearElasticity2DProblemNotSolved(
                "Reduced order Linear Elasticity 2D Problem has not been solved, can not return uh_free.")
        return self._uh_rom.flatt_values

    @property
    def default_file_names(self):
        return self.DEFAULT_FILE_NAMES_DICT

    @property
    def dirichlet_edge(self):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.dirichlet_edge

    @property
    def neumann_edge(self):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.neumann_edge

    @property
    def p(self):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.p

    @property
    def x(self):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.p[:, 0]

    @property
    def y(self):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.p[:, 1]

    @property
    def tri(self):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.tri

    @property
    def edge(self):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.edge

    @property
    def n(self):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.n

    @property
    def n_full(self):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.n_full

    @property
    def n_free(self):
        if not self._is_assembled_and_free:
            raise IsNotAssembledError("Matrices and vectors are not assembled.")
        return self._hf_data.n_free

    @property
    def plate_limits(self):
        if self._hf_data.plate_limits is None:
            text = "Plate limits are None, get the Linear Elasticity 2D Problem from functions or " \
                   "saved matrices and vectors first."
            raise PlateLimitsNotFoundError(text)
        return self._hf_data.plate_limits

    @property
    def v(self):
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not return.")
        return self._rb_data.v

    @property
    def ns_rom(self):
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not return.")
        return self._rb_data.ns_rom

    @property
    def n_rom(self):
        if not self._is_pod_computed:
            raise PodNotComputedError("Pod is not computed. Can not return.")
        return self._rb_data.n_rom

    @property
    def e_young_range(self):
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return self._rb_data.e_young_range

    @property
    def nu_poisson_range(self):
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return self._rb_data.nu_poisson_range

    @property
    def eps_pod(self):
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return self._rb_data.eps_pod

    @property
    def rb_grid(self):
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return self._rb_data.rb_grid

    @property
    def pod_mode(self):
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return self._rb_data.pod_mode

    @property
    def e_young_nu_poisson_mat(self):
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return get_e_young_nu_poisson_mat(self._rb_data)

    @property
    def n_rom_max(self):
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return self._rb_data.n_rom_max

    @property
    def singular_values_pod(self):
        if not self._is_pod_computed:
            warnings.warn("Pod is not computed. Returning default value.")
        return self._rb_data.sigma2_vec
