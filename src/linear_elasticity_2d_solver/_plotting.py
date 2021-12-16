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
    """
    Plot the singular values

    Parameters
    ----------
    sigma2_vec : np.array
        singular values squared.

    Returns
    -------
    None.

    """
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
    """
    Plot the relative information content

    Parameters
    ----------
    sigma2_vec : np.array
        singular values squared.
    n : int, optional
        The number N for plotting the point (N, I(N)). The default is None.

    Returns
    -------
    None.

    """
    arg0 = np.argwhere(sigma2_vec >= 0)
    i_n = np.cumsum(sigma2_vec[arg0]) / np.sum(sigma2_vec[arg0])
    plt.figure("Relative information content")
    plt.title("Relative information content, $I(N)$")
    plt.plot(np.arange(len(i_n)) + 1, i_n, "gD-")
    if n is not None:
        plt.plot(n, i_n[n - 1], "bo", label="$(N, I(N))$")
    plt.xlabel("$N$")
    plt.ylabel("$I(N)$")
    plt.grid()
    plt.legend()


def plot_mesh(n, p, tri):
    """
    Plot the mesh

    Parameters
    ----------
    n : int
        number of nodes along the axes.
    p : np.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    tri : np.array
        Elements. Index to the three corners of element i given in row i.

    Returns
    -------
    None.

    """
    plt.figure("Mesh plot", figsize=(7, 7))
    plt.title(f"Mesh {n - 1}x{n - 1}")
    plt.triplot(p[:, 0], p[:, 1], tri)
    plt.grid()


def plot_displacement(uh, n, p, tri, solve_mode=""):
    """
    Plot the displacement

    Parameters
    ----------
    uh : SolutionFunctionValues2D
        Numerical solution.
    n : int
        number of nodes along the axes.
    p : np.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    tri : np.array
        Elements. Index to the three corners of element i given in row i..
    solve_mode : str, optional
        high-fidelity ("hf"), reduced-order ("rb") or nothing ("") displacement text. The default is "".

    Returns
    -------
    None.

    """
    if solve_mode == "hf":
        title_text = f"Displacement in high-fidelity solution, $n={n-1}$"
    elif solve_mode == "rb":
        title_text = f"Displacement in reduced-order solution, $n={n-1}$"
    else:
        title_text = f"Displacement, $n={n-1}$"

    plt.figure(title_text)
    plt.title(title_text)
    colors1 = np.ones(tri.shape[0])
    cmap1 = colors.ListedColormap("red")
    cmap2 = colors.ListedColormap("gray")
    plt.tripcolor(p[:, 0] + uh.x, p[:, 1] + uh.y, tri, facecolors=colors1, cmap=cmap1)
    plt.tripcolor(p[:, 0], p[:, 1], tri, facecolors=colors1, cmap=cmap2, alpha=0.5)
    plt.grid()


def plot_von_mises(uh, n, p, tri, solve_mode="", levels=None):
    """
    Plot von Mises stress

    Parameters
    ----------
    uh : SolutionFunctionValues2D
        Numerical solution.
    n : int
        number of nodes along the axes.
    p : np.array
        Nodal points, (x,y)-coordinates for point i given in row i.
    tri : np.array
        Elements. Index to the three corners of element i given in row i..
    solve_mode : str, optional
        high-fidelity ("hf"), reduced-order ("rb") or nothing ("") von Mises text. The default is "".
    levels : np.array, optional
        array for the color levels, default 25 levels between 0 and max(von_mises). The default is None.

    Returns
    -------
    None.

    """
    if solve_mode == "hf":
        title_text = f"Von Mises stress in high-fidelity solution, $n={n-1}$"
    elif solve_mode == "rb":
        title_text = f"Von Mises stress in reduced-order solution, $n={n-1}$"
    else:
        title_text = f"Von Mises stress, $n={n-1}$"
    if levels is None:
        levels = np.linspace(0, np.max(uh.von_mises), 25)
        
    plt.figure(title_text)
    plt.title(title_text)
    plt.gca().set_aspect('equal')
    plt.tricontourf(p[:, 0] + uh.x, p[:, 1] + uh.y, tri, uh.von_mises, extend='both', levels=levels,
                    cmap=plt.cm.get_cmap("jet"))
    plt.colorbar()
    plt.grid()

    plt.xlim(np.min(p[:, 0] + uh.x) - 0.05, np.max(p[:, 0] + uh.x) + 0.05)
    plt.ylim(np.min(p[:, 1] + uh.y) - 0.05, np.max(p[:, 1] + uh.y) + 0.05)
