# -*- coding: utf-8 -*-
"""
@author: Olav M.S. Gran
"""


class MissingInputFunctionPointerError(Exception):
    pass


class IsNotAssembledError(Exception):
    pass


class LinearElasticity2DProblemNotSolved(Exception):
    pass


class PlateLimitsNotFoundError(Exception):
    pass


class PodNotComputedError(Exception):
    pass


class CanNotForceNromError(Exception):
    pass


class DirectoryDoesNotExistsError(Exception):
    pass


class FileDoesNotExistError(Exception):
    pass

class EdgesAreIllegalError(Exception):
    pass
