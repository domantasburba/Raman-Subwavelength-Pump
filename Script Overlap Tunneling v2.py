import math
import numpy as np

from QOPack.Utility import *
from QOPack.RamanTimeDepDetuning.OverlapTunneling import routine_band_flatness, routine_overlap_tunneling, routine_spectrum_tunneling, routine_overlap_tunneling_v2


@fun_time
# @profile
def main():
    # PHYSICAL PARAMETERS
    N_state = 5  # Number of dressed states (the same as the number of bare states)

    # Omega = 2.5  # Primary lattice strength
    Omega_span = (0.1, 1.2)
    # Omega_span = (0.1, 2.5)

    Delta = np.zeros(N_state)  # Detuning
    Delta[-1] = -0.0
    Delta[0] = 0.0
    Delta[1] = 0.0

    a_0 = 1.0
    # END OF PHYSICAL PARAMETERS

    # COMPUTATIONAL PARAMETERS
    K = 10000  # 20000  # Direct space divisions
    # Note that K being even or odd is important.
    band_num = 2  # Number of energy bands

    # R = 20  # Number of Omega points
    R = round(10*(Omega_span[1] - Omega_span[0]) + 1)  # Number of Omega points

    max_sep = 3.0  # Farthest tunneling elements and overlap integrals that are calculated correspond to wavefunctions separated by max_sep

    x_max = 10.0 * a_0  # 20.0 * a_0
    # END OF COMPUTATIONAL PARAMETERS

    # PLOTTING PARAMETERS
    colors = ["purple", "green", "blue", "red", "black"]
    # END OF PLOTTING PARAMETERS

    # routine_band_flatness(Omega_span, band_num, R, colors)
    # routine_overlap_tunneling(N_state, Omega_span, a_0, Delta, K, R, band_num, max_sep, x_max, colors)
    # routine_spectrum_tunneling(Omega_span, R, band_num, max_sep, colors)
    routine_overlap_tunneling_v2(N_state, Omega_span, Delta, K, R, band_num, max_sep, x_max)


if __name__ == "__main__":
    main()
