# -*- coding: utf-8 -*-
"""
Created on 07.10.2020

@author: Olav Gran
based in old code by Ruben Mustad
"""
import numpy as np
import matplotlib.pyplot as plt
from linear_elasticity_2d_solver import getPlatev2, getPlatev3

import sympy as sym

"""for nice representation of plots"""

sym.init_printing()
fontsize = 20
new_params = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
              'lines.linewidth': 2, 'lines.markersize': 7, 'ytick.labelsize': fontsize,
              'xtick.labelsize': fontsize, 'legend.fontsize': fontsize, 'legend.handlelength': 1.5}
plt.rcParams.update(new_params)


def meshplot(n_list, n_cols=3, save=False, deculany=False):
    """
    Function to make plots of meshes for N in n_list

    Parameters
    ----------
    n_list : list
        a list containing N's to plot a mesh for.
    n_cols : int, optional
        Number of columns in the final plot, meaning subplots per row. The default is 3.
    save : bool, optional
        Will the plot be saved in the plot folder. The default is False.
        Note: the plot folder must exist!

    Returns
    -------
    None.

    """
    # if n_list is just an int
    if isinstance(n_list, int):
        n_list = [n_list]

    # create a figure
    numPlots = len(n_list)
    nRows = numPlots // n_cols + numPlots % n_cols
    # make n_list's length to nRows*n_cols by appending -1
    while len(n_list) < nRows * n_cols:
        n_list.append(-1)
    # reshape n_list
    n_list = np.array(n_list).reshape((nRows, n_cols))
    # create the main figure and the axs as the subplots,
    # the figsize (21, 6) is good for nRows = 1
    # but figsize (21, 7 * nRows) is better for nRows = 2, 3, ...
    if nRows == 1:
        c = 6
    else:
        c = 7 * nRows
    fig, axs = plt.subplots(nRows, n_cols, figsize=(21, c))
    if n_cols == 1:
        axs = np.array([axs, ])
    if nRows == 1:
        axs = np.array([axs, ])

    for i in range(nRows):
        for j in range(n_cols):
            n = n_list[i, j]
            if n == -1:
                # don't show plot
                axs[i, j].set_axis_off()
            else:
                ax = axs[i, j]
                # get the nodes, elements and _edge lines
                if deculany:
                    p, tri, edge = getPlatev2(n, 0, 1)
                else:
                    p, tri, edge = getPlatev3(n, 0, 1)
                # plot them with triplot
                ax.triplot(p[:, 0], p[:, 1], tri)
                # label the axes
                ax.set_xlabel('$x$')
                ax.set_ylabel('$y$')
                # give the plot a title
                ax.set_title(F'Mesh for $N={n}$')
    # adjust
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # save the plot?
    if save:
        plt.savefig("other_plots/plot_mesh.pdf")
    plt.show()


if __name__ == "__main__":
    # save the plot as pdf?
    save = False
    deculany = False
    # list of N to plot for
    n_list = [3, 4, 5]
    # make a meshplot
    meshplot(n_list, save=save, deculany=deculany)
