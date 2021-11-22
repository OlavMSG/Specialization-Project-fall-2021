# -*- coding: utf-8 -*-
# Description:
#   Generate a mesh triangulation of the reference square (a,b)^2.
#
# Arguments:
#    n      Number of nodes in each spatial direction (n^2 total nodes).
#    a      Lower limit for x and y
#    b      Upper limit for x and y
#
# Returns:
#   p		Nodal points, (x,y)-coordinates for point i given in row i.
#   tri   	Elements. Index to the three corners of element i given in row i.
#   edge  	Index list of all nodal points on the outer _edge.
#
#   Author: Olav M. S. Gran
#   Boiler code by: Kjetil a. Johannessen, Abdullah Abdulhaque (October 2019)
#   Last edit: October 2021


import numpy as np
import scipy.spatial as spsa


def getPlatev2(n, a=0, b=1):
    # By: Olav Milian
    # Defining auxiliary variables.
    l = np.linspace(a, b, n)
    y, x = np.meshgrid(l, l)

    # Generating nodal points.
    n2 = n * n
    p = np.zeros((n2, 2))
    p[:, 0] = x.T.ravel()
    p[:, 1] = y.T.ravel()

    # Generating delaunay elements.
    mesh = spsa.Delaunay(p)
    tri = mesh.simplices

    # Generating nodal points on outer _edge.
    south_edge = np.array([np.arange(1, n), np.arange(2, n + 1)]).T
    east_edge = np.array([np.arange(n, n2 - n + 1, n), np.arange(2 * n, n2 + 1, n)]).T
    north_edge = np.array([np.arange(n2, n2 - n + 1, -1), np.arange(n2 - 1, n2 - n, -1)]).T
    west_edge = np.array([np.arange(n2 - n + 1, n - 1, -n), np.arange(n2 - 2 * n + 1, 0, -n)]).T
    edge = np.vstack((south_edge, east_edge, north_edge, west_edge))

    # Added this to get this script too work.
    edge -= 1

    return p, tri, edge


def getPlatev3(n, a=0, b=1):
    # By: Olav Milian
    # Defining auxiliary variables.
    l = np.linspace(a, b, n)
    y, x = np.meshgrid(l, l)

    # Generating nodal points.
    n2 = n * n
    p = np.zeros((n2, 2))
    p[:, 0] = x.T.ravel()
    p[:, 1] = y.T.ravel()

    # Generating elements.

    n12 = (n - 1) * (n - 1) * 2
    tri = np.zeros((n12, 3), dtype=int)

    def index_map(i, j):
        return i + n * j

    k = 0
    for i in range(n - 1):
        for j in range(n - 1):
            tri[k, 0] = index_map(i, j)
            tri[k, 1] = index_map(i + 1, j)
            tri[k, 2] = index_map(i + 1, j + 1)
            k += 1
            tri[k, 0] = index_map(i, j)
            tri[k, 1] = index_map(i + 1, j + 1)
            tri[k, 2] = index_map(i, j + 1)
            k += 1

    arg = np.argsort(tri[:, 0])
    tri = tri[arg]

    # Generating nodal points on outer _edge.
    south_edge = np.array([np.arange(1, n), np.arange(2, n + 1)]).T
    east_edge = np.array([np.arange(n, n2 - n + 1, n), np.arange(2 * n, n2 + 1, n)]).T
    north_edge = np.array([np.arange(n2, n2 - n + 1, -1), np.arange(n2 - 1, n2 - n, -1)]).T
    west_edge = np.array([np.arange(n2 - n + 1, n - 1, -n), np.arange(n2 - 2 * n + 1, 0, -n)]).T
    edge = np.vstack((south_edge, east_edge, north_edge, west_edge))

    # Added this to get this script too work.
    edge -= 1

    return p, tri, edge