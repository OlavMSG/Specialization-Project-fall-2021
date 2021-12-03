# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import matplotlib.colors as colors
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
    plt.semilogy(np.arange(len(rel_sigma_vec)) + 1, rel_sigma_vec, "mD-", label="Singular Values, $\\sigma_i$.")
    plt.xlabel("$i$")
    plt.ylabel("$\\sigma_i$")
    plt.grid()
    plt.legend()


def plot_relative_information_content(sigma2_vec, n=None):
    arg0 = np.argwhere(sigma2_vec >= 0)
    i_n = np.cumsum(sigma2_vec[arg0]) / np.sum(sigma2_vec[arg0])
    plt.figure("Relative information content")
    plt.title("Relative information content, $I(N)$")
    plt.plot(np.arange(len(i_n)) + 1, i_n, "gD-")
    if n is not None:
        plt.plot(n, i_n[n - 1], "bo", label="$(N_{rom}, I(N_{rom}))$")
    plt.xlabel("$N$")
    plt.ylabel("$I(N)$")
    plt.grid()
    plt.legend()


def plot_mesh(n, p, tri):
    plt.figure("Mesh plot", figsize=(7, 7))
    plt.title(f"Mesh for n={n - 1}x{n - 1}")
    plt.triplot(p[:, 0], p[:, 1], tri)
    plt.grid()


def plot_displacement(uh, p, tri, solve_mode=""):
    if solve_mode == "hf":
        title_text = "Displacement in high fidelity solution"
    elif solve_mode == "rb":
        title_text = "Displacement in reduced order solution"
    else:
        title_text = "Displacement"

    plt.figure(title_text)
    plt.title(title_text)
    colors1 = np.ones(tri.shape[0])
    cmap1 = colors.ListedColormap("red")
    cmap2 = colors.ListedColormap("gray")
    plt.tripcolor(p[:, 0] + uh.x, p[:, 1] + uh.y, tri, facecolors=colors1, cmap=cmap1)
    plt.tripcolor(p[:, 0], p[:, 1], tri, facecolors=colors1, cmap=cmap2, alpha=0.5)
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

    plt.xlim(np.min(p[:, 0] + uh.x) - 0.05, np.max(p[:, 0] + uh.x) + 0.05)
    plt.ylim(np.min(p[:, 1] + uh.y) - 0.05, np.max(p[:, 1] + uh.y) + 0.05)
