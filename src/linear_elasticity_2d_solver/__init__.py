# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""
from ._getplate import getPlatev2, getPlatev3
from ._default_constants import TOL_EDGE, E_YOUNG_RANGE, NU_POISSON_RANGE, EPS_POD
from ._helpers import get_mu_lambda, get_e_young_nu_poisson, get_u_exact
from ._linear_elasticity_2d_problem_class import LinearElasticity2DProblem

__all__ = [
    "getPlatev2",
    "getPlatev3",
    "TOL_EDGE",
    "E_YOUNG_RANGE",
    "NU_POISSON_RANGE",
    "EPS_POD",
    "get_mu_lambda",
    "get_e_young_nu_poisson",
    "get_u_exact",
    "LinearElasticity2DProblem"
]