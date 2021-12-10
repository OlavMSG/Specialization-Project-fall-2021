# -*- coding: utf-8 -*-
"""
@author: Olav Milian Gran
"""
import datetime
import multiprocessing as mp
import os
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

from linear_elasticity_2d_solver import LinearElasticity2DProblem
from linear_elasticity_2d_solver.default_constants import default_tol

"""for nice representation of plots"""

sym.init_printing()
fontsize = 20
new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
              'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize, 'figure.titlesize': fontsize,
              'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(new_params)


# rho_steal = 8e3  # kg/m^3

# Example 2: Gravity in 2D
def f(x, y):
    alpha = 8e3 * 9.81  # Newton/m^2...?
    return alpha, 0


def dirichlet_bc_func(x, y):
    return 0, 0


def clamped_bc(x, y):
    return abs(x) <= default_tol


def check_and_make_folder(n, folder_path):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    folder_path = os.path.join(folder_path, f"n{n}")
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    return folder_path


def make_plots(n, save, q=None, do_errors=True):
    now1 = datetime.datetime.now().time()  # time object
    txt = f"start time: {now1}, n={n}, save={save} \n"
    newline = "\n"
    line = "-" * 20
    save_dict = r"reduced_order_plots"
    save_dict = check_and_make_folder(n, save_dict)

    # define problem, can not get from saves, here because we want to set n_rom
    s = perf_counter()
    le2d = LinearElasticity2DProblem.from_functions(n, f,
                                                    get_dirichlet_edge_func=clamped_bc,
                                                    print_info=False)
    txt += f"Assembled HF system in {perf_counter() - s} s" + newline
    sigma2_dict = {}
    mean_err_dict = {}
    max_err_dict = {}
    n_rom_dict = {}
    for mode in ("uniform", "gauss lobatto"):
        sigma2_dict[mode] = {}
        mean_err_dict[mode] = {}
        max_err_dict[mode] = {}
        n_rom_dict[mode] = {}
        for grid in (5, 11):
            txt += line + newline + f"mode={mode}, gird={grid}" + newline
            s = perf_counter()
            le2d.build_rb_model(grid=grid, mode=mode, print_info=False)
            txt += f"Built RB model in {perf_counter() - s} s" + newline
            sigma2_dict[mode][grid] = le2d.singular_values_squared_pod
            n_rom_dict[mode][grid] = le2d.n_rom

            txt += f"Chosen n_rom={le2d.n_rom}, max use is n_rom_max={le2d.n_rom_max}, " \
                   + f"grid size is ns_rom={le2d.ns_rom}, Number of node on one axis is n={n}, " \
                   + f"Solution matrix rank: {le2d.solution_matrix_rank}" + newline

            txt += "Singular values squared:" + newline
            txt += str(le2d.singular_values_squared_pod) + newline
            if do_errors:
                max_err = np.zeros(le2d.n_rom_max)
                mean_err = np.zeros(le2d.n_rom_max)
                for n_rom in range(1, le2d.n_rom_max + 1):
                    errs = np.zeros(le2d.ns_rom)
                    for i, (e_young, nu_poisson) in enumerate(le2d.e_young_nu_poisson_mat):
                        # print(i, e_young, nu_poisson)
                        errs[i] = le2d.error_a_rb(e_young, nu_poisson, n_rom=n_rom)
                    max_err[n_rom - 1] = np.max(errs)
                    mean_err[n_rom - 1] = np.mean(errs)
                mean_err_dict[mode][grid] = mean_err
                max_err_dict[mode][grid] = max_err

    # make singular values plot
    plt.figure("Singular values")
    plt.title(f"Singular values, scaled to $\\sigma_1$, $n={n}$")
    for grid in (11, 5):
        for mode in ("gauss lobatto", "uniform"):
            sigma2_vec = sigma2_dict[mode][grid]
            arg0 = np.argwhere(sigma2_vec >= 0)
            sigma_vec = np.sqrt(sigma2_vec[arg0])
            rel_sigma_vec = sigma_vec / sigma_vec[0]
            plt.semilogy(np.arange(len(rel_sigma_vec)) + 1, rel_sigma_vec, "D-",
                         label=f"{mode} ${grid}\\times{grid}$", alpha=.8)
    plt.xlabel("$i$")
    plt.ylabel("$\\sigma_i$")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(save_dict + f"/singular_values_n{n}.pdf")
    plt.show()

    # make relative information plot
    plt.figure("Relative information content")
    plt.title(f"Relative information content, $I(N)$, $n={n}$")
    for grid in (11, 5):
        for mode in ("gauss lobatto", "uniform"):
            n_rom = n_rom_dict[mode][grid]
            sigma2_vec = sigma2_dict[mode][grid]
            arg0 = np.argwhere(sigma2_vec >= 0)
            i_n = np.cumsum(sigma2_vec[arg0]) / np.sum(sigma2_vec[arg0])
            plt.plot(np.arange(len(i_n)) + 1, i_n, "D-", label=f"{mode} ${grid}\\times{grid}$", alpha=.8)
            plt.plot(n_rom, i_n[n_rom - 1], "bo", alpha=.7)
    plt.xlabel("$N$")
    plt.ylabel("$I(N)$")
    plt.ylim(0.9996, 1.00005)
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(save_dict + f"/relative_information_content_mode_n{n}.pdf")
    plt.show()

    if do_errors:
        # make error plots
        for grid in (5, 11):
            fig, ax = plt.subplots(1, 1, num="Reduced order errors v. $n_{rom}$" + f"{grid}, $n={n}$", figsize=(12, 7))
            fig.suptitle("Reduced order errors v. $N$" + f", $n={n}$")
            for mode in ("gauss lobatto", "uniform"):
                mean_err = mean_err_dict[mode][grid]
                ax.semilogy(np.arange(len(mean_err)) + 1, mean_err, "D-",
                            label=f"mean: {mode} ${grid}\\times{grid}$", alpha=.8)
            for mode in ("gauss lobatto", "uniform"):
                max_err = max_err_dict[mode][grid]
                ax.semilogy(np.arange(len(max_err)) + 1, max_err, "D-",
                            label=f"max: {mode} ${grid}\\times{grid}$", alpha=.8)
            ax.set_xlabel("$N$")
            ax.set_ylabel("$\\|\\|u_h(\\mu) - Vu_N(\\mu)\\|\\|_a$")
            ax.grid()
            # adjust
            ax.legend(loc=9, bbox_to_anchor=(0.5, -0.13), ncol=2)
            if save:
                plt.savefig(save_dict + f"/reduced_order_errors_mode_grid{grid}_n{n}.pdf", bbox_inches='tight')
            plt.show()

    now2 = datetime.datetime.now().time()  # time object
    txt += f"end time: {now2}, n={n}, save={save} \n"
    if q is None:
        print(txt)
    else:
        res = "Process" + str(n), txt, now2
        q.put(res)


def listener(q, output_file):
    """listens for messages on the q, writes to file. """

    with open(output_file, 'w') as time_code_log:
        while True:
            m = q.get()
            if m == 'kill':
                time_code_log.write('killed')
                break

            time_code_log.write(m[0] + '\n')
            time_code_log.write(m[1] + '\n')
            time_code_log.write(str(m[2]) + '\n')
            time_code_log.flush()


def main():
    # took some time!!!! (20: 12 min, 40: 40 min, 80: 2 hours 42 min, total: 3 hours 34 min), without multiprocessing
    # took some time!!!! (20: 13 min, 40: 41 min, 80: 2 hours 41 min, total: 2 hours 34 min), with multiprocessing
    multi_process = True
    save = True
    n_vec = [20, 40, 80]
    text_n_vec = "_".join(str(n) for n in n_vec)
    output_file = "reduced_order_plots/time_log_n" + text_n_vec + ".txt"

    if multi_process:
        # must use Manager queue here, or will not work
        manager = mp.Manager()
        q = manager.Queue()
        pool = mp.Pool(mp.cpu_count())
        # put listener to work first
        watcher = pool.apply_async(listener, (q, output_file))
        jobs = []
        for n in n_vec:
            job = pool.apply_async(make_plots, (n, save, q))
            jobs.append(job)
        # collect results from the make_plots through the pool result queue
        for job in jobs:
            job.get()
        # now we are done, kill the listener
        q.put('kill')
        pool.close()
        pool.join()
    else:
        with open(output_file, "w") as time_code_log:
            sys.stdout = time_code_log
            for n in n_vec:
                make_plots(n, save)


if __name__ == '__main__':
    main()
