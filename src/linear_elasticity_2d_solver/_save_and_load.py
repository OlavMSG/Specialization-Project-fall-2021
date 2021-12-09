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


def _check_and_make_folder(n, folder_path):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    folder_path = os.path.join(folder_path, f"n{n}")
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    print(f"Saving in directory {folder_path}")
    return folder_path


def hf_save(hf_data, directory_path, has_neumann, has_non_homo_dirichlet, has_non_homo_neumann,
            default_file_names_dict=file_names_dict):

    hf_folder_path = os.path.join(directory_path, "high_fidelity")
    hf_folder_path = _check_and_make_folder(hf_data.n, hf_folder_path)

    a1_file_path = os.path.join(hf_folder_path, default_file_names_dict["a1"])
    a2_file_path = os.path.join(hf_folder_path, default_file_names_dict["a2"])
    sparse.save_npz(a1_file_path, hf_data.a1_full.tocsr())
    sparse.save_npz(a2_file_path, hf_data.a2_full.tocsr())

    f_load_lv_file_path = os.path.join(hf_folder_path, default_file_names_dict["f_load_lv"])
    np.save(f_load_lv_file_path, hf_data.f_load_lv_full, allow_pickle=False)

    dirichlet_edge_file_path = os.path.join(hf_folder_path, default_file_names_dict["dirichlet_edge"])
    np.save(dirichlet_edge_file_path, hf_data.dirichlet_edge, allow_pickle=False)

    p_tri_edge_file_path = os.path.join(hf_folder_path, default_file_names_dict["p_tri_edge"])
    np.savez(p_tri_edge_file_path, p=hf_data.p, tri=hf_data.tri, edge=hf_data.edge,
             allow_pickle=False)

    if has_neumann:
        f_load_neumann_file_path = os.path.join(hf_folder_path, default_file_names_dict["f_load_neumann"])
        if has_non_homo_neumann:
            np.save(f_load_neumann_file_path, hf_data.f_load_neumann_full, allow_pickle=False)
        else:
            np.save(f_load_neumann_file_path, np.array(["has homo neumann"]), allow_pickle=False)
    if has_non_homo_dirichlet:
        rg_lifting_func_file_path = os.path.join(hf_folder_path, default_file_names_dict["rg"])
        np.save(rg_lifting_func_file_path, hf_data.rg, allow_pickle=False)

    print(f"Saved the high fidelity data in {hf_folder_path}")


def rb_save(n, rb_data, directory_path, has_neumann, has_non_homo_dirichlet, has_non_homo_neumann,
            default_file_names_dict=file_names_dict):

    rb_folder_path = os.path.join(directory_path, "reduced_order")
    rb_folder_path = _check_and_make_folder(n, rb_folder_path)

    a1_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["a1_rom"])
    a2_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["a2_rom"])
    np.save(a1_rom_file_path, rb_data.a1_free_rom, allow_pickle=False)
    np.save(a2_rom_file_path, rb_data.a2_free_rom, allow_pickle=False)

    f_load_lv_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f_load_lv_rom"])
    np.save(f_load_lv_rom_file_path, rb_data.f_load_lv_free_rom, allow_pickle=False)

    if has_neumann:
        f_load_neumann_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f_load_neumann_rom"])
        if has_non_homo_neumann:
            np.save(f_load_neumann_rom_file_path, rb_data.f_load_neumann_free_rom, allow_pickle=False)
    if has_non_homo_dirichlet:
        f1_dir_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f1_dir_rom"])
        f2_dir_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f2_dir_rom"])
        np.save(f1_dir_rom_file_path, rb_data.f1_dirichlet_rom, allow_pickle=False)
        np.save(f2_dir_rom_file_path, rb_data.f2_dirichlet_rom, allow_pickle=False)

    v_mat_file_path = os.path.join(rb_folder_path, default_file_names_dict["v"])
    np.save(v_mat_file_path, rb_data.v, allow_pickle=False)

    sigma2_file_path = os.path.join(rb_folder_path, default_file_names_dict["sigma2"])
    np.save(sigma2_file_path, rb_data.sigma2_vec, allow_pickle=False)

    pod_parameters = np.array([rb_data.e_young_range,
                               rb_data.nu_poisson_range,
                               rb_data.rb_grid,
                               (rb_data.eps_pod, rb_data.pod_mode),
                               (rb_data.n_rom_max, rb_data.n_rom_cut)])
    pod_parameters_file_path = os.path.join(rb_folder_path, default_file_names_dict["pod_parameters"])
    np.save(pod_parameters_file_path, pod_parameters, allow_pickle=False)

    print(f"Saved the reduced order data in {rb_folder_path}")


def hf_from_files(hf_data, directory_path, default_file_names_dict=file_names_dict):
    hf_folder_path = os.path.join(directory_path, "high_fidelity", f"n{hf_data.n}")
    if not os.path.isdir(hf_folder_path):
        text = f"Directory {hf_folder_path} does not exist, can not load the high_fidelity data."
        raise DirectoryDoesNotExistsError(text)

    a1_file_path = os.path.join(hf_folder_path, default_file_names_dict["a1"])
    a2_file_path = os.path.join(hf_folder_path, default_file_names_dict["a2"])
    hf_data.a1_full = sparse.load_npz(a1_file_path)
    hf_data.a2_full = sparse.load_npz(a2_file_path)

    f_load_lv_file_path = os.path.join(hf_folder_path, default_file_names_dict["f_load_lv"])
    hf_data.f_load_lv_full = np.load(f_load_lv_file_path, allow_pickle=False)

    dirichlet_edge_file_path = os.path.join(hf_folder_path, default_file_names_dict["dirichlet_edge"])
    hf_data.dirichlet_edge = np.load(dirichlet_edge_file_path, allow_pickle=False)

    p_tri_edge_file_path = os.path.join(hf_folder_path, default_file_names_dict["p_tri_edge"])
    p_tri_edge = np.load(p_tri_edge_file_path, allow_pickle=False)
    hf_data.p = p_tri_edge['p']
    hf_data.tri = p_tri_edge['tri']
    hf_data.edge = p_tri_edge['edge']

    f_load_neumann_file_path = os.path.join(hf_folder_path, default_file_names_dict["f_load_neumann"])
    has_neumann = os.path.isfile(f_load_neumann_file_path)
    has_non_homo_neumann = True
    if has_neumann:
        hf_data.f_load_neumann_full = np.load(f_load_neumann_file_path, allow_pickle=False)
        if hf_data.f_load_neumann_full[0] == "has homo neumann":
            hf_data.f_load_neumann_full = None
            has_non_homo_neumann = False

    hf_data.get_neumann_edge()
    hf_data.compute_free_and_expanded_edges(has_neumann, has_non_homo_neumann)
    hf_data.plate_limits = (np.min(hf_data.p), np.max(hf_data.p))

    rg_lifting_func_file_path = os.path.join(hf_folder_path, default_file_names_dict["rg"])
    has_non_homo_dirichlet = os.path.isfile(rg_lifting_func_file_path)
    if has_non_homo_dirichlet:
        hf_data.rg = np.load(rg_lifting_func_file_path, allow_pickle=False)

    return has_neumann, has_non_homo_dirichlet, has_non_homo_neumann


def rb_from_files(n, rb_data, directory_path, warn=True, default_file_names_dict=file_names_dict):
    rb_folder_path = os.path.join(directory_path, "reduced_order", f"n{n}")
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
        a1_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["a1_rom"])
        a2_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["a2_rom"])
        rb_data.a1_free_rom = np.load(a1_rom_file_path, allow_pickle=False)
        rb_data.a2_free_rom = np.load(a2_rom_file_path, allow_pickle=False)

        f_load_lv_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f_load_lv_rom"])
        rb_data.f_load_lv_free_rom = np.load(f_load_lv_rom_file_path, allow_pickle=False)

        f_load_neumann_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f_load_neumann_rom"])
        if os.path.isfile(f_load_neumann_rom_file_path):
            rb_data.f_load_neumann_free_rom = np.load(f_load_neumann_rom_file_path, allow_pickle=False)
            if rb_data.f_load_neumann_free_rom[0] == "has homo neumann":
                rb_data.f_load_neumann_free_rom = None

        a1_dir_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f1_dir_rom"])
        a2_dir_rom_file_path = os.path.join(rb_folder_path, default_file_names_dict["f2_dir_rom"])
        a1_dir_rom_isfile = os.path.isfile(a1_dir_rom_file_path)
        a2_dir_rom_isfile = os.path.isfile(a2_dir_rom_file_path)
        if (a1_dir_rom_isfile and not a2_dir_rom_isfile) or (not a1_dir_rom_isfile and a2_dir_rom_isfile):
            text = "One of the files {}".format(default_file_names_dict["f1_dir_rom"]) \
                   + " and {} does not exist.".format(default_file_names_dict["f2_dir_rom"])
            raise FileDoesNotExistError(text)
        elif a2_dir_rom_isfile and a2_dir_rom_isfile:
            rb_data.f1_dirichlet_rom = np.load(a1_dir_rom_file_path, allow_pickle=False)
            rb_data.f2_dirichlet_rom = np.load(a2_dir_rom_file_path, allow_pickle=False)
            # has_non_homo_dirichlet = True

        v_mat_file_path = os.path.join(rb_folder_path, default_file_names_dict["v"])
        rb_data.v = np.load(v_mat_file_path, allow_pickle=False)
        rb_data.n_rom = rb_data.v.shape[1]

        sigma2_file_path = os.path.join(rb_folder_path, default_file_names_dict["sigma2"])
        rb_data.sigma2_vec = np.load(sigma2_file_path, allow_pickle=False)

        pod_parameters_file_path = os.path.join(rb_folder_path, default_file_names_dict["pod_parameters"])
        pod_parameters = np.load(pod_parameters_file_path, allow_pickle=False)
        rb_data.e_young_range = tuple(pod_parameters[0].astype(float))
        rb_data.nu_poisson_range = tuple(pod_parameters[1].astype(float))
        rb_data.rb_grid = tuple(pod_parameters[2].astype(int))
        rb_data.eps_pod = pod_parameters[3, 0].astype(float)
        rb_data.pod_mode = pod_parameters[3, 1].astype(str)
        rb_data.n_rom_max = pod_parameters[4, 0].astype(int)
        rb_data.n_rom_cut = pod_parameters[4, 1]
        if rb_data.n_rom_cut != "rank":
            rb_data.n_rom_cut.astype(float)


    return is_pod_computed


