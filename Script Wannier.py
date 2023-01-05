from math import pi
import numpy as np
import matplotlib.pyplot as plt

from QOPack.Math import *
from QOPack.Utility import *
from QOPack.Wannier import full_calc_Wannier
from QOPack.WannierV2 import *
from QOPack.RamanTimeDepDetuning.OverlapTunneling import create_chi_HO, plot_Wannier_HO

# There are subtle choices in this code (parameters being even or odd, k_arr having or not having endpoints).
# Slight changes will probably break the resulting Wannier functions.


def plot_EDR(k_arr, E_k, Omega, N_band=3):
    fig = plt.figure(r"EDR", figsize=(8, 5))
    ax = plt.axes()

    plt.title(r"$\Omega = %.2fE_R$" % Omega)
    plt.xlabel(r"$k$ / rad")
    plt.ylabel(r"$E$ / $E_R$")
    for i in range(N_band):
        # # plt.hlines(2*n*np.sqrt(Omega) - (n**2 + n + 1/2)/8, k_arr[0], k_arr[-1], color="blue", linestyles="--")
        # plt.hlines(-2*Omega + 2*(i+1/2)*np.sqrt(Omega) - (i**2 + i + 1/2)/8, k_arr[0], k_arr[-1], color="blue", linestyles="--")
        # print(-2*Omega + 2*(i+1/2)*np.sqrt(Omega) - (i**2 + i + 1/2)/8)
        plt.plot(k_arr, E_k[:, i], color="blue")

    set_plot_defaults(fig, ax)
    save_plot("EDR")
    plt.show()


@fun_time
# @profile
def main():
    # # PARAMETERS
    # Omega = 2.0
    # x_max = 10.
    # band_num = 2
    # K = 10000

    # a_0 = 1.
    # k_0 = 2. * np.pi / a_0

    # sen = 0

    # x_arr = np.linspace(-x_max, x_max, K)
    # # END OF PARAMETERS

    # W_R, D2_W_R = full_calc_Wannier(Omega, x_max, band_num, K)
    # chi_HO = create_chi_HO(x_arr, Omega, band_num, K, k_0)
    # plot_Wannier_HO(W_R, chi_HO, x_arr, K, a_0, sen)

    # PARAMETERS
    Omega = 0.1

    x_span = (-10, 10)

    N_Fourier = 50  # Small N_Fourier only good for lowest band Wannier.
    N_band = 3
    N_x = 10000
    # Note that N_x and N_k being even or odd is important.

    S_band = 1
    S_x = x2index(x_span, N_x, 1.)

    S_band_arr = np.arange(9)
    # END OF PARAMETERS

    N2_Fourier = 2*N_Fourier + 1

    # k_arr = np.linspace(-pi, pi, N_k, endpoint=False)
    x_arr = np.linspace(*x_span, N_x)

    N_k = 250
    S_k = 200
    k_arr = np.linspace(-pi, pi, N_k)

    W_R, D2_W_R = full_calc_Wannier_Kohn(Omega, x_span, N_band, N_x, useCached=False)
    # abs2_W_R = abs2(W_R)
    plot_Wannier_vs_x(x_arr, W_R, S_band, Omega)

    # E_k, Bloch_k = solve_sin_Schrodinger(Omega, x_span, N_band, N_Fourier, N_k, N_x)
    # Bloch_k, W_R = calc_Wannier_Kohn(Bloch_k, k_arr, x_arr, Omega, N_k, N_x, N2_Fourier)
    # plot_EDR(k_arr, E_k, Omega)

    # plot_log_abs2_Wannier_vs_x(x_arr, abs2_W_R, S_band, Omega)

    # plot_u_k_vs_x(x_arr, u_k, S_band, S_k, Omega)
    # plot_u_k_vs_k(k_arr, u_k, S_band, S_x, Omega)
    # plot_Bloch_k_vs_k(k_arr, Bloch_k, S_band, S_x, Omega)
    # plot_Bloch_k_vs_x(x_arr, Bloch_k, S_band, S_k, Omega)
    # plotN_Wannier_vs_x(x_arr, W_R, S_band_arr, Omega)

    # Note that Mantas uses alternate normalization for Wannier functions.

    # # Ideally, antisymmetic wavefunctions should be zero at x=0.
    # # plotN_Wannier_vs_x(x_arr, W_R, S_band_arr, Omega)
    # # plotN_log_abs2_Wannier_vs_x(x_arr, abs2_W_R, S_band_arr, Omega)
    # plotN_Wannier_and_log_abs2_Wannier_vs_x(x_arr, W_R, abs2_W_R, Omega)

    # plt.plot(x_arr, D2_W_R[S_Fourier], color='black')
    # plt.show()

    # for i in range(3):
    #     print(E_k[N_k//2, i+1] - E_k[N_k//2, i],
    #           2*np.sqrt(Omega) - 1/4*(i+1))


if __name__ == "__main__":
    main()
