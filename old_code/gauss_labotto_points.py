# -*- coding: utf-8 -*-
"""
Created on 12.11.2021

@author: Olav Milian Gran
"""
from quadpy.c1 import gauss_lobatto
import numpy as np
range_ = (10e3, 310e3)

def main():
    m = 5
    gl_points = gauss_lobatto(m).points
    print(gl_points)
    print(range_, np.mean(range_))
    u = np.linspace(range_[0], range_[1], m)
    print(u, np.mean(u))
    gl = 0.5 * ((range_[1] - range_[0]) * gl_points + (range_[1] + range_[0]))
    print(gl, np.mean(gl))


if __name__ == '__main__':
    main()
