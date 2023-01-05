from datetime import datetime
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from QOPack.Utility import get_main_dir, set_plot_defaults, save_plot
from QOPack.Wannier import full_diag_cos_lattice


# --------------- GENERAL CALCULATIONS ---------------
@njit(cache=True)
def calc_indirect_gaps(E):
    """E = E(k); E.shape = (N_band, N_k)."""
    N_band_m1 = E.shape[0] - 1

    indirect_gaps = np.zeros(N_band_m1, dtype=np.float64)
    for i in range(N_band_m1):
        indirect_gaps[i] = E[i + 1].min() - E[i].max()
    
    return indirect_gaps


@njit(cache=True)
def calc_direct_gaps(E):
    """E = E(k); E.shape = (N_band, N_k)."""
    N_band_m1 = E.shape[0] - 1

    direct_gaps = np.zeros(N_band_m1, dtype=np.float64)
    for i in range(N_band_m1):
        direct_gaps[i] = (E[i+1] - E[i]).min()

    return direct_gaps


@njit(cache=True)
def calc_bandwidths(E):
    N_band = E.shape[0]

    bandwidths = np.zeros(N_band, dtype=np.float64)
    for i in range(N_band):
        bandwidths[i] = E[i].max() - E[i].min()

    return bandwidths
# --------------- END OF GENERAL CALCULATIONS ---------------


def calc_shifted_E(E):
    band_num = np.shape(E)[0]
    band_gaps = calc_indirect_gaps(E)
    shifted_E = E - (band_gaps[0] * np.arange(band_num))[:, np.newaxis]

    return shifted_E


def full_cos_EDR(Omega, band_num, N_k, singlePlot):
    k_arr = np.linspace(-np.pi, np.pi, N_k)
    E, g_Fourier = full_diag_cos_lattice(Omega, band_num, N_k)
    print("Band gaps:", calc_indirect_gaps(E))
    print("Bandwidths:", calc_bandwidths(E))
    if singlePlot:
        plot_EDR(k_arr, E, Omega)
    else:
        subplot_EDR(k_arr, E, Omega)


def full_cos_shifted_E(Omega, band_num, N_k, colors):
    k_span = np.array([-np.pi, np.pi])
    k_arr = np.linspace(k_span[0], k_span[1], N_k)
    E, g_Fourier = full_diag_cos_lattice(Omega, band_num, N_k)
    shifted_E = calc_shifted_E(E)
    plot_shifted_EDR(k_arr, shifted_E, Omega, colors)


def plot_EDR(q_arr, E, Omega):
    fig = plt.figure(r"EDR", figsize=(6, 5))
    ax = plt.axes()

    plt.title(r"$\Omega = %.2fE_R$" % Omega)
    plt.xlabel(r"$k$ / $k_0$")
    plt.ylabel(r"$E$ / $E_R$")
    for i in range(np.shape(E)[0]):
        plt.plot(q_arr, E[i], color="blue")

    set_plot_defaults(fig, ax)
    save_plot("EDR")
    plt.show()


def plot_shifted_EDR(q_arr, shifted_E, Omega, colors):
    fig = plt.figure(r"Harmonic EDR", figsize=(6, 5))
    ax = plt.axes()

    plt.title(r"$\Omega = %.3fE_R$" % Omega)
    plt.tick_params(axis="both", direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    plt.xlabel(r"$k$ / $k_0$")
    plt.ylabel(r"$E$ / $E_R$")
    for i in range(np.shape(shifted_E)[0]):
        plt.plot(q_arr, shifted_E[i], color=colors[i % len(colors)])
    plt.grid()

    datetime_now = datetime.now()
    path_string = "%s/Graphs/%s" % (get_main_dir(), datetime_now.strftime("%Y-%m-%d"))
    pathlib.Path(path_string).mkdir(parents=True, exist_ok=True)
    name_string = "Harmonic_EDR!%s.png" % datetime_now.strftime("%Y-%m-%d!%H.%M.%S")
    plt.savefig("%s/%s" % (path_string, name_string))
    plt.show()


def subplot_EDR(q_arr, E, Omega):
    band_num = np.shape(E)[0]
    fig, axs = plt.subplots(band_num, sharex=True)
    fig.suptitle(r"$\Omega = %.3fE_R$" % Omega)
    for i in range(band_num):
        axs[i].tick_params(axis="both", direction="in")
        axs[i].xaxis.set_ticks_position("both")
        axs[i].yaxis.set_ticks_position("both")
        axs[i].set_ylabel(r"$E$ / $E_R$")
        axs[i].plot(q_arr, E[band_num - 1 - i], color="blue")
        axs[i].grid()
    axs[band_num - 1].set_xlabel(r"$k$ / $k_0$")
    plt.tight_layout()

    datetime_now = datetime.now()
    path_string = "%s/Graphs/%s" % (get_main_dir(), datetime_now.strftime("%Y-%m-%d"))
    pathlib.Path(path_string).mkdir(parents=True, exist_ok=True)
    name_string = "Subplot_EDR!%s.png" % datetime_now.strftime("%Y-%m-%d!%H.%M.%S")
    plt.savefig("%s/%s" % (path_string, name_string))
    plt.show()
