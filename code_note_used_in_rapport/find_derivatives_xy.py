# -*- coding: utf-8 -*-
"""
Created on 07.11.2021

@author: Olav Milian Gran
"""
from sympy import sin
from sympy.abc import x, y, lamda, mu, pi
from sympy.matrices import Matrix, eye

differ = Matrix([x, y])


def epsilon(u):
    grad_u = u.jacobian(differ)
    return 0.5 * (grad_u + grad_u.T)


def div(u):
    return u.jacobian(differ).trace()


def sigma(u):
    return 2 * mu * epsilon(u) + lamda * div(u) * eye(2)


def div_sym_matrix_2d(mat):
    c1 = mat[:, 0]
    c2 = mat[:, 1]
    # print(c1.jacobian(differ))
    # print(c2.jacobian(differ))
    return Matrix([div(c1), div(c2)])


def psi_basis(phi, d):
    if d == 0:
        return Matrix([phi, 0.])
    else:
        return Matrix([0., phi])


def el_mult_2d(u, v):
    return Matrix([[u[0, 0] * v[0, 0], u[0, 1] * v[0, 1]],
                   [u[1, 0] * v[1, 0], u[1, 1] * v[1, 1]]])


def double_dot_2d(u, v):
    return sum(el_mult_2d(u, v))


def main():
    u = Matrix([sin(pi*x)*sin(pi*y), sin(pi*x)*sin(pi*y)])

    grad_u = u.jacobian(differ)
    # print(u)
    # print(grad_u)
    # epsilon_u = epsilon(u)
    # print(epsilon_u, "epsilon")
    # div_u = div(u)
    # print(div_u, "div")
    # print(eye(2))
    sigma_u = sigma(u)
    # print(sigma_u)
    print("f")
    div_sigma = div_sym_matrix_2d(sigma_u)
    print(-div_sigma)
    print("bc")
    print(u.evalf(subs={x: 0.}), "west")
    print(u.evalf(subs={x: 1.}), "east")
    print(u.evalf(subs={y: 0.}), "south")
    print(u.evalf(subs={y: 1.}), "north")


if __name__ == '__main__':
    main()
