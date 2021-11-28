# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

"""for nice representation of plots"""

sym.init_printing()
fontsize = 20
new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'figure.figsize': (12, 7),
              'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize, 'figure.titlesize': fontsize,
              'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(new_params)


def plot_singular_values(sigma2_vec):
    plt.figure("Singular values")
    plt.title("Singular values, scaled to $\\sigma_1$")
    arg0 = np.argwhere(sigma2_vec >= 0)
    sigma_vec = np.sqrt(sigma2_vec[arg0])
    rel_sigma_vec = sigma_vec / sigma_vec[0]
    plt.semilogy(np.arange(len(rel_sigma_vec)) + 1, rel_sigma_vec, "mx--", label="Singular Values, $\\sigma$.")
    plt.grid()
    plt.legend()


def plot_relative_information_content(sigma2_vec, n=None):
    arg0 = np.argwhere(sigma2_vec >= 0)
    i_n = np.cumsum(sigma2_vec[arg0]) / np.sum(sigma2_vec[arg0])
    plt.figure("Relative information content")
    plt.title("Relative information content")
    plt.plot(np.arange(len(i_n)) + 1, i_n, "mx--", label="$I(n)$")
    if n is not None:
        plt.plot(n, i_n[n - 1], "bo", label="$(n_{rom}, I(n_{rom}))$")
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
    """fig, axs = plt.subplots(1, 2, figsize=(16, 7), num=title_text)
    fig.suptitle(title_text)
    ax1, ax2 = axs

    ax1.triplot(p[:, 0], p[:, 1], tri)
    ax1.grid()
    ax1.set_title("Initial position")

    ax2.triplot(p[:, 0] + uh.x, p[:, 1] + uh.y, tri)
    ax2.grid()
    ax2.set_title("Displaced position")

    # adjust
    plt.subplots_adjust(hspace=0.3, wspace=0.3)"""
    plt.figure(title_text)
    plt.title(title_text)
    plt.triplot(p[:, 0] + uh.x, p[:, 1] + uh.y, tri)
    plt.grid()


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
