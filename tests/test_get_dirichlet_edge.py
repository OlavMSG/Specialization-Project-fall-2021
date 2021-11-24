# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
import numpy as np

from linear_elasticity_2d_solver import LinearElasticity2DProblem, DEFAULT_TOL
from linear_elasticity_2d_solver.exceptions import EdgesAreIllegalError


def f(x, y):
    return 1e5, 1e5

def dirichlet_bc_func(x, y):
    return 0, 0

def neumann_bc_func(x, y):
    return 0, 0


def test1():
    n = 3
    test_res = False
    try:
        le2d = LinearElasticity2DProblem.from_functions(n, f, neumann_bc_func=neumann_bc_func)
    except EdgesAreIllegalError:
        test_res = True
    assert test_res


def test2():
    n = 3
    le2d = LinearElasticity2DProblem.from_functions(n, f, dirichlet_bc_func=dirichlet_bc_func)

    dirichlet_edge = np.array([[0, 1], [1, 2], [2, 5], [5, 8], [8, 7], [7, 6], [6, 3], [3, 0]])

    test_res = (le2d.dirichlet_edge == dirichlet_edge).all() and (le2d.neumann_edge is None)
    assert test_res


def test3():
    n = 3

    def get_dirichlet_bc_func(x, y):
        return abs(x) < DEFAULT_TOL

    le2d = LinearElasticity2DProblem.from_functions(n, f, neumann_bc_func=neumann_bc_func,
                                                    dirichlet_bc_func=dirichlet_bc_func,
                                                    get_dirichlet_edge_func=get_dirichlet_bc_func)

    dirichlet_edge = np.array([[6, 3], [3, 0]])
    neumann_edge = np.array([[0, 1], [1, 2], [2, 5], [5, 8], [7, 6], [8, 7]])

    test_res = (le2d.dirichlet_edge == dirichlet_edge).all() and (le2d.neumann_edge == neumann_edge).all()
    assert test_res


def test4():
    n = 3

    def get_dirichlet_bc_func(x, y):
        return abs(x) < DEFAULT_TOL

    le2d = LinearElasticity2DProblem.from_functions(n, f, dirichlet_bc_func=dirichlet_bc_func,
                                                    get_dirichlet_edge_func=get_dirichlet_bc_func)

    dirichlet_edge = np.array([[6, 3], [3, 0]])
    neumann_edge = np.array([[0, 1], [1, 2], [2, 5], [5, 8], [7, 6], [8, 7]])

    test_res = (le2d.dirichlet_edge == dirichlet_edge).all() and (le2d.neumann_edge == neumann_edge).all()
    assert test_res


def test5():
    n = 3

    def get_dirichlet_bc_func(x, y):
        return abs(x) < DEFAULT_TOL

    le2d = LinearElasticity2DProblem.from_functions(n, f, neumann_bc_func=neumann_bc_func,
                                                    get_dirichlet_edge_func=get_dirichlet_bc_func)

    dirichlet_edge = np.array([[6, 3], [3, 0]])
    neumann_edge = np.array([[0, 1], [1, 2], [2, 5], [5, 8], [7, 6], [8, 7]])

    test_res = (le2d.dirichlet_edge == dirichlet_edge).all() and (le2d.neumann_edge == neumann_edge).all()
    assert test_res


"""expect fail from here"""
def test6():
    n = 3

    def get_dirichlet_bc_func(x, y):
        return True

    test_res = False
    try:
        le2d = LinearElasticity2DProblem.from_functions(n, f, neumann_bc_func=neumann_bc_func,
                                                    dirichlet_bc_func=dirichlet_bc_func,
                                                    get_dirichlet_edge_func=get_dirichlet_bc_func)
    except EdgesAreIllegalError:
        test_res = True
    assert test_res



def test7():
    n = 3

    def get_dirichlet_bc_func(x, y):
        return False

    test_res = False
    try:
        le2d = LinearElasticity2DProblem.from_functions(n, f, neumann_bc_func=neumann_bc_func,
                                                    dirichlet_bc_func=dirichlet_bc_func,
                                                    get_dirichlet_edge_func=get_dirichlet_bc_func)
    except EdgesAreIllegalError:
        test_res = True
    assert test_res



def test8():
    n = 3

    def get_dirichlet_bc_func(x, y):
        return True

    test_res = False
    try:
        le2d = LinearElasticity2DProblem.from_functions(n, f, dirichlet_bc_func=dirichlet_bc_func,
                                                    get_dirichlet_edge_func=get_dirichlet_bc_func)
    except EdgesAreIllegalError:
        test_res = True
    assert test_res


def test9():
    n = 3

    def get_dirichlet_bc_func(x, y):
        return False

    test_res = False
    try:
        le2d = LinearElasticity2DProblem.from_functions(n, f, neumann_bc_func=neumann_bc_func,
                                                    get_dirichlet_edge_func=get_dirichlet_bc_func)
    except EdgesAreIllegalError:
        test_res = True
    assert test_res


def main():
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()
    test9()


if __name__ == '__main__':
    main()
