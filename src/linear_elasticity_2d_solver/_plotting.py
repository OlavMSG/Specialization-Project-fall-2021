# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

from ._default_constants import EPS_POD

"""for nice representation of plots"""

sym.init_printing()
fontsize = 20
new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
              'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize, 'figure.titlesize': fontsize,
              'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(new_params)


def plot_singular_values(sigma2_vec, n=None, eps_pod=EPS_POD):
    plt.figure("Singular values")
    plt.title("Singular values")
    plt.semilogy(np.arange(len(sigma2_vec)) + 1, sigma2_vec, "mx--", label="Singular Values, $\\sigma^2$")
    plt.hlines(eps_pod, xmin=-10, xmax=len(sigma2_vec) + 10, linestyles="dashed", colors="k",
               label="$\\epsilon_{POD}$")
    if n is not None:
        plt.plot(n, sigma2_vec[n - 1], "bo", label="$(n_{min}, \\sigma_{n_{min}}^2)$")
    plt.xlim(-1, len(sigma2_vec) + 1)
    plt.grid()
    plt.legend()


def plot_relative_information_content(sigma2_vec, n=None, eps_pod=EPS_POD):
    i_n = np.cumsum(sigma2_vec) / np.sum(sigma2_vec)
    plt.figure("Relative information content")
    plt.title("Relative information content")
    plt.plot(np.arange(len(i_n)) + 1, i_n, "mx--", label="$I(n_rom)$")
    plt.hlines(1 - eps_pod ** 2, xmin=-10, xmax=len(i_n) + 10, linestyles="dashed", colors="k",
               label="$1-\\epsilon_{POD}^2$")
    if n is not None:
        plt.plot(n, i_n[n - 1], "bo", label="$(n_{min}, I(n_{min}))$")
    plt.xlim(-1, len(i_n) + 1)
    plt.grid()
    plt.legend()


def plot_mesh(n, p, tri):
    plt.figure("Mesh plot", figsize=(7, 7))
    plt.title(f"Mesh for n={n}")
    plt.triplot(p[:, 0], p[:, 1], tri)
    plt.grid()


def plot_displacement(uh, p, tri, solve_mode=""):
    if solve_mode == "hf":
        title_text = "Displacement in high fidelity solution"
    elif solve_mode == "rb":
        title_text = "Displacement in reduced order solution"
    else:
        title_text = "Displacement"
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), num=title_text)
    fig.suptitle(title_text)
    ax1, ax2 = axs

    ax1.triplot(p[:, 0], p[:, 1], tri)
    ax1.grid()
    ax1.set_title("Initial position")

    ax2.triplot(p[:, 0] + uh.x, p[:, 1] + uh.y, tri)
    ax2.grid()
    ax2.set_title("Displaced position")

    # adjust
    plt.subplots_adjust(hspace=0.3, wspace=0.3)


def plot_von_mises(uh, p, tri, solve_mode=""):
    if solve_mode == "hf":
        title_text = "Stress intensity, von Mises stress, \n in high fidelity solution"
    elif solve_mode == "rb":
        title_text = "Stress intensity, von Mises stress, \n in reduced order solution"
    else:
        title_text = "Stress intensity, von Mises stress"
    plt.figure(title_text)
    plt.title(title_text)
    plt.gca().set_aspect('equal')
    levels = np.linspace(np.min(uh.von_mises), np.max(uh.von_mises), 25)
    plt.tricontourf(p[:, 0] + uh.x, p[:, 1] + uh.y, tri, uh.von_mises, extend='both', levels=levels,
                    cmap=plt.cm.get_cmap("jet"))
    plt.colorbar()
    plt.grid()
