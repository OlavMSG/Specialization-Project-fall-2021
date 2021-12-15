# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import os
import warnings

import numpy as np
import scipy.sparse as sparse

from .default_constants import file_names_dict
from .exceptions import DirectoryDoesNotExistsError, FileDoesNotExistError
from .helpers import check_and_make_folder


# hf_data is HighFidelityData, not imported due to circular import
# rb_data is ReducedOrderData, not imported due to circular import
# from ._hf_data_class import HighFidelityData
# from ._rb_data_class import ReducedOrderData


def hf_save(hf_data, directory_path, has_neumann, has_non_homo_dirichlet, has_non_homo_neumann,
            default_file_names_dict=file_names_dict):
    """
    Save the high-fidelity data

    Parameters
    ----------
    hf_data :
        high-fidelity data.
    directory_path : str
        path to directory to save in.
    has_neumann : bool
        Does the problem have Neumann boundary conditions.
    has_non_homo_dirichlet : bool
        Does the problem have non homogeneous Dirichlet boundary conditions.
    has_non_homo_neumann : bool
        Does the problem have non homogeneous Neumann boundary conditions.
    default_file_names_dict : dict, optional
        Dictionary of names for the files, see e.g. file_names_dict in default_constants.
        The default is file_names_dict.

    Returns
    -------
    None.

    """

    hf_folder_path = os.path.join(directory_path, "high_fidelity")
    hf_folder_path = check_and_make_folder(hf_data.n, hf_folder_path, n_counts_nodes=True)
    print(f"Saving in directory {hf_folder_path}")
    # matrices a1 and a2
    a1_file_path = os.path.join(hf_folder_path, default_file_names_dict["a1"])
    a2_file_path = os.path.join(hf_folder_path, default_file_names_dict["a2"])
    sparse.save_npz(a1_file_path, hf_data.a1_full.tocsr())
    sparse.save_npz(a2_file_path, hf_data.a2_full.tocsr())
    # body force load vector
    f_load_lv_file_path = os.path.join(hf_folder_path, default_file_names_dict["f_load_lv"])
    np.save(f_load_lv_file_path, hf_data.f_load_lv_full, allow_pickle=False)
    # Dirichlet edge
    dirichlet_edge_file_path = os.path.join(hf_folder_path, default_file_names_dict["dirichlet_edge"])
    np.save(dirichlet_edge_file_path, hf_data.dirichlet_edge, allow_pickle=False)
    # p, tri and edge
    p_tri_edge_file_path = os.path.join(hf_folder_path, default_file_names_dict["p_tri_edge"])
    np.savez(p_tri_edge_file_path, p=hf_data.p, tri=hf_data.tri, edge=hf_data.edge,
             allow_pickle=False)
    # Neumann load vector
    if has_neumann:
        f_load_neumann_file_path = os.path.join(hf_folder_path, default_file_names_dict["f_load_neumann"])
        if has_non_homo_neumann:
            np.save(f_load_neumann_file_path, hf_data.f_load_neumann_full, allow_pickle=False)
        else:
            np.save(f_load_neumann_file_path, np.array(["has homo neumann"]), allow_pickle=False)
    # Lifting Dirichlet load vector
    if has_non_homo_dirichlet:
        rg_lifting_func_file_path = os.path.join(hf_folder_path, default_file_names_dict["rg"])
        np.save(rg_lifting_func_file_path, hf_data.rg, allow_pickle=False)

    print(f"Saved the high fidelity data in {hf_folder_path}")


def rb_save(n, rb_data, directory_path, has_neumann, has_non_homo_dirichlet, has_non_homo_neumann,
            default_file_names_dict=file_names_dict):
    """
    Save the reduced-order data

    Parameters
    ----------
    n : int
        The number of nodes along the axes.
    rb_data :
        reduced-order data.
    directory_path : str
        path to directory to save in.
    has_neumann : bool
        Does the problem have Neumann boundary conditions.
    has_non_homo_dirichlet : bool
        Does the problem have non homogeneous Dirichlet boundary conditions.
    has_non_homo_neumann : bool
        Does the problem have non homogeneous Neumann boundary conditions.
    default_file_names_dict : dict, optional
        Dictionary of names for the files, see e.g. file_names_dict in default_constants.
        The default is file_names_dict.

    Returns
    -------
    None.

    """
    rb_folder_path = os.path.join(directory_path, "reduced_order")
    rb_folder_path = check_and_make_folder(n, rb_folder_path, n_counts_nodes=True)
    print(f"Saving in directory {rb_folder_path}")
    # matrices a1 and a2
    a1_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["a1_rom"])
    a2_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["a2_rom"])
    np.save(a1_rom_file_path, rb_data.a1_free_rom, allow_pickle=False)
    np.save(a2_rom_file_path, rb_data.a2_free_rom, allow_pickle=False)
    # body force load vector
    f_load_lv_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f_load_lv_rom"])
    np.save(f_load_lv_rom_file_path, rb_data.f_load_lv_free_rom, allow_pickle=False)
    # Neumann load vector
    if has_neumann:
        f_load_neumann_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f_load_neumann_rom"])
        if has_non_homo_neumann:
            np.save(f_load_neumann_rom_file_path, rb_data.f_load_neumann_free_rom, allow_pickle=False)
    # Lifting Dirichlet load vector
    if has_non_homo_dirichlet:
        f1_dir_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f1_dir_rom"])
        f2_dir_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f2_dir_rom"])
        np.save(f1_dir_rom_file_path, rb_data.f1_dirichlet_rom, allow_pickle=False)
        np.save(f2_dir_rom_file_path, rb_data.f2_dirichlet_rom, allow_pickle=False)
    # the matrix v
    v_mat_file_path = os.path.join(rb_folder_path, default_file_names_dict["v"])
    np.save(v_mat_file_path, rb_data.v, allow_pickle=False)
    # The singular values squared
    sigma2_file_path = os.path.join(rb_folder_path, default_file_names_dict["sigma2"])
    np.save(sigma2_file_path, rb_data.sigma2_vec, allow_pickle=False)
    # Pod parameters
    pod_parameters = np.array([rb_data.e_young_range,
                               rb_data.nu_poisson_range,
                               rb_data.rb_grid,
                               (rb_data.eps_pod, rb_data.pod_sampling_mode),
                               (rb_data.n_rom_max, rb_data.n_rom_cut)])
    pod_parameters_file_path = os.path.join(rb_folder_path, default_file_names_dict["pod_parameters"])
    np.save(pod_parameters_file_path, pod_parameters, allow_pickle=False)

    print(f"Saved the reduced order data in {rb_folder_path}")


def hf_from_files(hf_data, directory_path, default_file_names_dict=file_names_dict):
    """
    Get the high-fidelity data from saved files

    Parameters
    ----------
    hf_data :
        high-fidelity data.
    directory_path : str
        path to directory to save in.
    default_file_names_dict : dict, optional
        Dictionary of names for the files, see e.g. file_names_dict in default_constants.
        The default is file_names_dict.

    Raises
    ------
    DirectoryDoesNotExistsError
        If directory does not exit.

    Returns
    -------
    has_neumann : bool
        Does the problem have Neumann boundary conditions.
    has_non_homo_dirichlet : bool
        Does the problem have non homogeneous Dirichlet boundary conditions.
    has_non_homo_neumann : bool
        Does the problem have non homogeneous Neumann boundary conditions.
    

    """
    hf_folder_path = os.path.join(directory_path, "high_fidelity", f"n{hf_data.n - 1}")
    if not os.path.isdir(hf_folder_path):
        text = f"Directory {hf_folder_path} does not exist, can not load the high_fidelity data."
        raise DirectoryDoesNotExistsError(text)
    # matrices a1 and a2
    a1_file_path = os.path.join(hf_folder_path, default_file_names_dict["a1"])
    a2_file_path = os.path.join(hf_folder_path, default_file_names_dict["a2"])
    hf_data.a1_full = sparse.load_npz(a1_file_path)
    hf_data.a2_full = sparse.load_npz(a2_file_path)
    # body force load vector
    f_load_lv_file_path = os.path.join(hf_folder_path, default_file_names_dict["f_load_lv"])
    hf_data.f_load_lv_full = np.load(f_load_lv_file_path, allow_pickle=False)
    # Dirichlet edge
    dirichlet_edge_file_path = os.path.join(hf_folder_path, default_file_names_dict["dirichlet_edge"])
    hf_data.dirichlet_edge = np.load(dirichlet_edge_file_path, allow_pickle=False)
    # p, tri and edge
    p_tri_edge_file_path = os.path.join(hf_folder_path, default_file_names_dict["p_tri_edge"])
    p_tri_edge = np.load(p_tri_edge_file_path, allow_pickle=False)
    hf_data.p = p_tri_edge['p']
    hf_data.tri = p_tri_edge['tri']
    hf_data.edge = p_tri_edge['edge']
    # Neumann load vector
    f_load_neumann_file_path = os.path.join(hf_folder_path, default_file_names_dict["f_load_neumann"])
    has_neumann = os.path.isfile(f_load_neumann_file_path)
    has_non_homo_neumann = True
    if has_neumann:
        hf_data.f_load_neumann_full = np.load(f_load_neumann_file_path, allow_pickle=False)
        if hf_data.f_load_neumann_full[0] == "has homo neumann":
            hf_data.f_load_neumann_full = None
            has_non_homo_neumann = False
    # Get edge data
    hf_data.get_neumann_edge()
    hf_data.compute_free_and_expanded_edges(has_neumann, has_non_homo_neumann)
    hf_data.plate_limits = (np.min(hf_data.p), np.max(hf_data.p))
    # Lifting Dirichlet load vector
    rg_lifting_func_file_path = os.path.join(hf_folder_path, default_file_names_dict["rg"])
    has_non_homo_dirichlet = os.path.isfile(rg_lifting_func_file_path)
    if has_non_homo_dirichlet:
        hf_data.rg = np.load(rg_lifting_func_file_path, allow_pickle=False)

    return has_neumann, has_non_homo_dirichlet, has_non_homo_neumann


def rb_from_files(n, rb_data, directory_path, warn=True, default_file_names_dict=file_names_dict):
    """
    Load the reduced -order data if it exists.

    Parameters
    ----------
    n : int
        The number of nodes along the axes.
    rb_data :
        reduced-order data.
    directory_path : str
        path to directory to save in.
    warn : bool, optional
        Do warn the user. The default is True.
    default_file_names_dict : dict, optional
        Dictionary of names for the files, see e.g. file_names_dict in default_constants.
        The default is file_names_dict.

    Raises
    ------
    FileDoesNotExistError
        if one lifting Dirichlet load vector save file exists and the other does not exist.

    Returns
    -------
    is_pod_computed : bool
        True if the reduced-order data exists.

    """
    rb_folder_path = os.path.join(directory_path, "reduced_order", f"n{n - 1}")
    is_pod_computed = True
    if not os.path.isdir(rb_folder_path):
        # assume the data does not exist.
        if warn:
            warning_text = f"Directory {rb_folder_path}" \
                           + " does not exist, can not load the reduced order data." \
                           + "\nBuild it with build_rb_model of the class."
            warnings.warn(warning_text)
        is_pod_computed = False
    else:
        # matrices a1 and a2
        a1_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["a1_rom"])
        a2_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["a2_rom"])
        rb_data.a1_free_rom = np.load(a1_rom_file_path, allow_pickle=False)
        rb_data.a2_free_rom = np.load(a2_rom_file_path, allow_pickle=False)
        # body force load vector
        f_load_lv_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f_load_lv_rom"])
        rb_data.f_load_lv_free_rom = np.load(f_load_lv_rom_file_path, allow_pickle=False)
        # neumann load vector
        f_load_neumann_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f_load_neumann_rom"])
        if os.path.isfile(f_load_neumann_rom_file_path):
            rb_data.f_load_neumann_free_rom = np.load(f_load_neumann_rom_file_path, allow_pickle=False)
            if rb_data.f_load_neumann_free_rom[0] == "has homo neumann":
                rb_data.f_load_neumann_free_rom = None
        # Lifting Dirichlet load vector
        f1_dir_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f1_dir_rom"])
        f2_dir_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f2_dir_rom"])
        f1_dir_rom_isfile = os.path.isfile(f1_dir_rom_file_path)
        f2_dir_rom_isfile = os.path.isfile(f2_dir_rom_file_path)
        if (f1_dir_rom_isfile and not f2_dir_rom_isfile) or (not f1_dir_rom_isfile and f2_dir_rom_isfile):
            text = "One of the files {}".format(default_file_names_dict["f1_dir_rom"]) \
                   + " and {} does not exist.".format(default_file_names_dict["f2_dir_rom"])
            raise FileDoesNotExistError(text)
        elif f2_dir_rom_isfile and f2_dir_rom_isfile:
            rb_data.f1_dirichlet_rom = np.load(f1_dir_rom_file_path, allow_pickle=False)
            rb_data.f2_dirichlet_rom = np.load(f2_dir_rom_file_path, allow_pickle=False)
            # has_non_homo_dirichlet = True
        # matrix v 
        v_mat_file_path = os.path.join(rb_folder_path, default_file_names_dict["v"])
        rb_data.v = np.load(v_mat_file_path, allow_pickle=False)
        rb_data.n_rom = rb_data.v.shape[1]
        # singular values squared
        sigma2_file_path = os.path.join(rb_folder_path, default_file_names_dict["sigma2"])
        rb_data.sigma2_vec = np.load(sigma2_file_path, allow_pickle=False)
        # pod parameters
        pod_parameters_file_path = os.path.join(rb_folder_path, default_file_names_dict["pod_parameters"])
        pod_parameters = np.load(pod_parameters_file_path, allow_pickle=False)
        rb_data.e_young_range = tuple(pod_parameters[0].astype(float))
        rb_data.nu_poisson_range = tuple(pod_parameters[1].astype(float))
        rb_data.rb_grid = tuple(pod_parameters[2].astype(int))
        rb_data.eps_pod = pod_parameters[3, 0].astype(float)
        rb_data.pod_sampling_mode = pod_parameters[3, 1].astype(str)
        rb_data.n_rom_max = pod_parameters[4, 0].astype(int)
        rb_data.n_rom_cut = pod_parameters[4, 1]
        if rb_data.n_rom_cut != "rank":
            rb_data.n_rom_cut.astype(float)

    return is_pod_computed
