# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

from ._default_constants import EPS_POD

"""for nice representation of plots"""

sym.init_printing()
fontsize = 20
new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
              'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
              'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(new_params)


def plot_singular_values(sigma2_vec, n=None, eps_pod=EPS_POD):
    plt.figure("Singular values", figsize=(12, 7))
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
    plt.figure("Relative information content", figsize=(12, 7))
    plt.title("Relative information content")
    plt.plot(np.arange(len(i_n)) + 1, i_n, "mx--", label="$I(n_rom)$")
    plt.hlines(1 - eps_pod ** 2, xmin=-10, xmax=len(i_n) + 10, linestyles="dashed", colors="k",
               label="$1-\\epsilon_{POD}^2$")
    if n is not None:
        plt.plot(n, i_n[n - 1], "bo", label="$(n_{min}, I(n_{min}))$")
    plt.xlim(-1, len(i_n) + 1)
    plt.grid()
    plt.legend()
