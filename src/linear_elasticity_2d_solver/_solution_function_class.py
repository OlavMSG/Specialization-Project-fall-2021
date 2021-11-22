# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""

from ._default_constants import DEFAULT_TOL
from ._helpers import FunctionValues2D


class SolutionFunctionValues2D(FunctionValues2D):

    def __init__(self):
        super().__init__()
        self._e_young = None
        self._nu_poisson = None
        self._von_mises = None

    def set_e_young_and_nu_poisson(self, e_young, nu_poisson):
        self._e_young = e_young
        self._nu_poisson = nu_poisson

    def check_e_young_and_nu_poisson(self, e_young, nu_poisson):
        if self._values is None:
            return False
        elif abs(self._e_young - e_young) <= DEFAULT_TOL and abs(self._nu_poisson - nu_poisson) <= DEFAULT_TOL:
            return True
        else:
            return False

    def set_von_mises(self, von_mises):
        self._von_mises = von_mises

    @property
    def e_young(self):
        return self._e_young

    @property
    def nu_poisson(self):
        return self._nu_poisson

    @property
    def von_mises(self):
        return self._von_mises

