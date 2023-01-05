#!/usr/bin/env python3
import numpy as np
from scipy.integrate import solve_ivp


def solve_Schrodinger(calc_Hamiltonian, ket_initial, time_arr, rtol=1e-8, atol=1e-12):
    def Schrodinger(t, ket):
        Ham = calc_Hamiltonian(t)
        return -1.j * Ham @ ket

    time_span = (time_arr[0], time_arr[-1])
    N_time = len(time_arr)
    ket_shape = ket_initial.shape

    # Set rtol and atol to keep relatively good normalization.
    sol = solve_ivp(Schrodinger, time_span, ket_initial, t_eval=time_arr, rtol=rtol, atol=atol)

    # SAVING SOLUTION INTO 2D ARRAY.
    ket = np.zeros((N_time, *ket_shape), dtype=np.complex128)
    for j in range(2):
        ket[:, j] = sol.y[j]
    # END OF SAVING SOLUTION INTO 2D ARRAY.

    return ket
