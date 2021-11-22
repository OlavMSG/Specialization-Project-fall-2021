# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

# Default tolerance for checking
DEFAULT_TOL = 1e-14
# Default plate limits
# we will use the plate [self._plate_limits[0], self._plate_limits[1]]^2
PLATE_LIMITS = (0, 1)
# Default rb grid
RB_GRID = (5, 5)
# default pod mode
POD_MODE = "uniform"
# Ranges for parameters
E_YOUNG_RANGE = (10e3, 310e3)  # MPa
NU_POISSON_RANGE = (0, 0.4)
# epsilon for pod
EPS_POD = 1e-2
# max cut value for n_rom
N_ROM_CUT = 1e-12
# default names for saved files
FILE_NAMES_DICT = {"a1": "a1_mat.npz",
                   "a2": "a2_mat.npz",
                   "f_load_lv": "f_load_lv.npy",
                   "f_load_neumann": "f_load_neumann.npy",
                   "dirichlet_edge": "dirichlet_edge.npy",
                   "p_tri_edge": "p_tri_edge.npz",
                   "rg": "rg_lifting_func.npy",
                   "a1_rom": "a1_rom_mat.npy",
                   "a2_rom": "a2_rom_mat.npy",
                   "a1_dir_rom": "a1_dir_rom_mat.npy",
                   "a2_dir_rom": "a2_dir_rom_mat.npy",
                   "f_load_lv_rom": "f_load_lv_rom.npy",
                   "f_load_neumann_rom": "f_load_neumann_rom.npy",
                   "v": "v_mat.npy",
                   "sigma2": "singular_values_vec.npy",
                   "pod_parameters": "pod_parameters.npy"}
