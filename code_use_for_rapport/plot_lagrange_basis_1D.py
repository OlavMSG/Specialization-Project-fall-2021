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


def phi(n, i):
    phi_vec = np.zeros(n + 1)
    phi_vec[i] = 1
    return phi_vec


def main():
    n = 1
    save = True
    save_dict = r"other_plots"
    plt.figure("1D Linear Lagrange basis")
    plt.title("1D Linear Lagrange basis")
    x_vec = np.linspace(0, 1, n + 1)
    for i in range(n + 1):
        plt.plot(x_vec, phi(n, i), "-")
        if i % 2 == 0:
            y = 1.02
        else:
            y = 1.12
        if n == 1:
            y = 1.02
        plt.annotate("$\\varphi_{" + str(i) + "}(x)$", (x_vec[i], y), fontsize=30,
                     ha='center')
    plt.grid()
    plt.xlim(-0.08, 1.08)
    plt.ylim(-0.05, 1.22)

    if save:
        plt.savefig(save_dict + f"/1D_linear_lagrange_basis_n{n}.pdf")
    plt.show()


if __name__ == '__main__':
    main()
