import numpy as np
import sympy

def rk4(list_of_fs, initial_values, interval, iterations):
    '''
    Solves a system of the form
    y1'(s) = f(s, y1, y2, ..., ym)
    ...
    ym'(s)
    '''
    a, b = interval
    h = (b-a)/iterations
    matrix_of_ys = np.zeros((iterations, len(initial_values)))
    matrix_of_ys[:, 0] = initial_values.T
    for k in range(iterations):
        for i in range(initial_values):
            k1 = h*list_of_fs[i].subs([()])
