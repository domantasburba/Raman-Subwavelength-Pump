import functools
from math import pi
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from QOPack.Math import *
from QOPack.Utility import *
from QOPack.RamanTimeDepDetuning.ZakPhase import *
import QOPack.RamanTimeDepDetuning.TimeEvolutionFull as Full
import QOPack.RamanTimeDepDetuning.MomentumFull as Full_k


# @njit(cache=True)
# def smooth_Zak_arr(Zak_arr):
#     """Shift from [-pi, pi) to [0, 2pi)."""
#     # Zak_arr[0, 0] -= 2 * pi
#     for i in range(Zak_arr.shape[0]):
#         for j in range(Zak_arr.shape[1]):
#             if Zak_arr[i, j] < 0.:
#                 Zak_arr[i, j] += 2 * pi

#     return Zak_arr


# --------------- MODULATION SCHEMES ---------------
@njit(cache=True)
def modulate_RM_book_generic(t, u_bar, v_bar, w_0, T_pump):
    """Janos k. Asboth, Laszlo Oroszlany, Andras Palyi, "A Short Course on
    Topological Insulators: Band Structure and Edge States in One and Two
    Dimensions", Springer.  Taken from SFN.pdf, page 60."""
    # u = np.sin(2*pi*t/T_pump)
    u = u_bar + np.sin(2*pi*t/T_pump)
    v = v_bar + np.cos(2*pi*t/T_pump)
    w = w_0

    return u, v, w


@njit(cache=True)
def modulate_RM_Ian_alternating(t, J_0, T_pump):
    """From Spielman2022.2202.05033.pdf"""
    u = 0.
    if np.remainder(t, T_pump) < 0.5*T_pump:
        v = J_0
        w = 0.
    else:
        v = 0.
        w = J_0

    return u, v, w


@njit(cache=True)
def set_modulation_RM(t, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme):
    if modulation_scheme == "Book":
        return modulate_RM_book_generic(t, u_bar, v_bar, w_0, T_pump)  # [u, v, w]
    elif modulation_scheme == "Ian":
        return modulate_RM_Ian_alternating(t, J_0, T_pump)  # [u, v, w]
    else:
        raise ValueError(r"Invalid modulation scheme.")
# --------------- END OF MODULATION SCHEMES ---------------


# --------------- SPECIFIC CALCULATIONS ---------------
@njit(cache=True)
def calc_RM_Hamiltonian(t, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_Ham):
    # if adiabaticLaunching:
    #     alpha_t = Full.calc_alpha_t(t, tau_adiabatic)
    # else:
    #     alpha_t = 1.
    alpha_t = 1.  # DISABLED

    u, v, w = set_modulation_RM(t, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme)

    Ham = np.zeros((N_Ham, N_Ham), dtype=np.float64)

    # v ASSIGNMENT
    for i in range(0, N_Ham - 1, 2):
        Ham[i, i + 1] = v
        Ham[i + 1, i] = v
    # END OF v ASSIGNMENT

    # w ASSIGNMENT
    for i in range(1, N_Ham - 1, 2):
        Ham[i, i + 1] = w
        Ham[i + 1, i] = w
    # END OF w ASSIGNMENT

    # u ASSIGNMENT
    for i in range(0, N_Ham, 2):
        Ham[i, i] = u
    for i in range(1, N_Ham, 2):
        Ham[i, i] = -u
    # END OF u ASSIGNMENT

    return Ham


@njit(cache=True)
def calc_RM_k_Hamiltonian(t, k, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme):
    u, v, w = set_modulation_RM(t, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme)

    H_12 = v + w * np.exp(-1j * k)

    Ham_k = np.empty((2, 2), dtype=np.complex128)
    Ham_k[0, 0] = u
    Ham_k[0, 1] = H_12
    Ham_k[1, 0] = np.conj(H_12)
    Ham_k[1, 1] = -u

    return Ham_k


def time_evolution_RM(ket_initial, time_arr, time_span, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_Ham, N_time):
    calc_Hamiltonian = functools.partial(calc_RM_Hamiltonian, u_bar=u_bar, v_bar=v_bar, w_0=w_0, J_0=J_0, T_pump=T_pump, modulation_scheme=modulation_scheme, adiabaticLaunching=adiabaticLaunching, tau_adiabatic=tau_adiabatic, N_Ham=N_Ham)

    def Schrodinger(t, ket):
        Ham = calc_Hamiltonian(t)
        return -1.j * Ham @ ket

    sol = solve_ivp(Schrodinger, time_span, ket_initial, t_eval=time_arr, atol=1e-8, rtol=1e-10)
    # sol = solve_ivp(Schrodinger, time_span, ket_initial, t_eval=time_arr, atol=1E-5, rtol=1E-5)

    # SAVING SOLUTION INTO 2D ARRAY.
    ket = np.zeros((N_time, N_Ham), dtype=np.complex128)
    for j in range(N_Ham):
        ket[:, j] = sol.y[j]
    # END OF SAVING SOLUTION INTO 2D ARRAY.

    return ket


def diag_RM_time_spectrum(period_arr, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_period, N_Ham):
    # calc_Hamiltonian = functools.partial(calc_RM_Hamiltonian, u_bar=u_bar, v_bar=v_bar, w_0=w_0, J_0=J_0, T_pump=T_pump, modulation_scheme=modulation_scheme, N_Ham=N_Ham)
    calc_Hamiltonian = functools.partial(calc_RM_Hamiltonian, u_bar=u_bar, v_bar=v_bar, w_0=w_0, J_0=J_0, T_pump=T_pump, modulation_scheme=modulation_scheme, adiabaticLaunching=False, tau_adiabatic=0, N_Ham=N_Ham)

    E = np.empty((N_period, N_Ham), dtype=np.float64)
    ket_eigen = np.empty((N_period, N_Ham, N_Ham), dtype=np.float64)
    for i in range(N_period):
        Ham = calc_Hamiltonian(period_arr[i])
        # E[i, :] = np.linalg.eigvalsh(Ham)
        E[i, :], v = np.linalg.eigh(Ham)
        ket_eigen[i, :] = v.T

    return E, ket_eigen


def calc_RM_Wannier_center(period_arr, k_arr, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_lat, N_period):
    # CONSTRUCTING RICE-MELE HAMILTONIAN
    # Ham_k.shape = (N_period, N_lat, 2, 2)
    Ham_k = np.empty((N_period, N_lat, 2, 2), dtype=np.complex128)
    for idx_period in range(N_period):
        for idx_k in range(N_lat):
            Ham_k[idx_period, idx_k, :, :] = calc_RM_k_Hamiltonian(period_arr[idx_period], k_arr[idx_k], u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme)
    # END OF CONSTRUCTING RICE-MELE HAMILTONIAN

    # SOLVING EIGENVALUE EQUATION
    E_k = np.empty((N_period, N_lat, 2), dtype=np.float64)
    # u_k.shape = (N_period, N_lat, 2, 2)
    u_k = np.empty_like(Ham_k, dtype=np.complex128)
    for i in range(N_period):
        for j in range(N_lat):
            E_k[i, j, :], v = np.linalg.eigh(Ham_k[i, j, :, :])
            u_k[i, j, :, :] = v.T
    # END OF SOLVING EIGENVALUE EQUATION

    # CALCULATE ZAK PHASE
    Zak_arr = calc_general_Berry_phase(u_k, prod_axis=1)
    Zak_arr = smooth_Berry_phase(np.swapaxes(Zak_arr, 0, 1))
    Zak_arr = np.swapaxes(Zak_arr, 0, 1)
    # Zak_arr.shape: (N_period, 2) -> (2, N_period) -> (N_period, 2)
    # print(Zak_arr.shape)
    # END OF CALCULATE ZAK PHASE

    x_Wannier = Zak_arr / (2 * pi)

    return x_Wannier
# --------------- END OF SPECIFIC CALCULATIONS ---------------


# @@@@@@@@@@@@@@@ FULL ROUTINES @@@@@@@@@@@@@@@
def routine_RM_time_spectrum(u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_lat, N_period):
    # DERIVED PARAMETERS
    N_Ham = 2 * N_lat

    time_selected = 0.6*T_pump  # Must be between 0 and T_pump (in the first period).
    S_eigen = 0  # N_Ham // 2
    S_period = x2index((0., T_pump), N_period, time_selected)

    period_arr = np.linspace(0., T_pump, N_period)
    # END OF DERIVED PARAMETERS

    E, ket_eigen = diag_RM_time_spectrum(period_arr, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_period, N_Ham)

    # plot_period_spectrum(E, period_arr, N_Ham)
    # plot_RM_eigenstate(E, ket_eigen, period_arr, S_period, S_eigen, T_pump)
    plotN_RM_spectrum_eigen(E, ket_eigen, period_arr, S_period, S_eigen)


def routine_RM_Wannier_center(u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_lat, N_period):
    # DERIVED PARAMETERS
    k_arr = np.linspace(-pi, pi, N_lat)
    period_arr = np.linspace(0., T_pump, N_period)
    # END OF DERIVED PARAMETERS

    x_Wannier = calc_RM_Wannier_center(period_arr, k_arr, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_lat, N_period)

    plot_RM_x_Wannier(period_arr, x_Wannier)


def routine_RM_time_evolution(time_span, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, initial_condition, adiabaticLaunching, tau_adiabatic, time_selected, N_lat, N_time, N_period):
    # DERIVED PARAMETERS
    N_Ham = 2 * N_lat
    time_arr = np.linspace(*time_span, N_time)

    period_arr = np.linspace(0., T_pump, N_period)

    S_time = x2index(time_span, N_time, time_selected)
    # END OF DERIVED PARAMETERS

    E, ket_eigen = diag_RM_time_spectrum(period_arr, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_period, N_Ham)

    # INITIAL CONDITION
    ket_initial = Full.set_initial_condition(ket_eigen, initial_condition, 5., N_lat, N_Ham, 0, 2)
    # END OF INITIAL CONDITION

    # TIME EVOLUTION
    ket = time_evolution_RM(ket_initial, time_arr, time_span, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_Ham, N_time)
    # END OF TIME EVOLUTION

    # CALCULATION OF WAVEFUNCTION PARAMETERS
    abs2_ket = abs2(ket)
    ket_COM = Full.calc_ket_COM(abs2_ket, N_lat=N_lat, N_state=1, N_band=2)
    # END OF CALCULATION OF WAVEFUNCTION PARAMETERS

    # DEBUG
    # print("ket_COM(t=0) & ket_COM(t=1):\n",
    #       ket_COM[x2index((time_arr[0], time_arr[-1]), len(time_arr), 0.)], "\n",
    #       ket_COM[x2index((time_arr[0], time_arr[-1]), len(time_arr), 1.)])
    # END OF DEBUG

    # PLOTTING
    Full.plotN_ket2_snapshots(abs2_ket, time_arr, ["red", "black"], S_time, N_lat, 2)
    Full.plot_ket_pcolormesh(abs2_ket, N_Ham, time_arr, T_pump, file_name="RM_Ket_Pcolormesh")
    Full.plot_ket_COM(ket_COM, time_arr, T_pump, file_name="RM_Ket_COM")
    # END OF PLOTTING


def routine_RM_all(time_span, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, initial_condition, adiabaticLaunching, tau_adiabatic, time_selected, N_lat, N_time, N_period):
    # DERIVED PARAMETERS
    N_Ham = 2 * N_lat
    time_arr = np.linspace(*time_span, N_time)

    k_arr = np.linspace(-pi, pi, N_lat)
    period_arr = np.linspace(0., T_pump, N_period)

    S_time = x2index(time_span, N_time, time_selected)
    # END OF DERIVED PARAMETERS

    E, ket_eigen = diag_RM_time_spectrum(period_arr, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_period, N_Ham)

    # INITIAL CONDITION
    ket_initial = Full.set_initial_condition(ket_eigen, initial_condition, 5., N_lat, N_Ham, 0, 2)
    # END OF INITIAL CONDITION

    # TIME EVOLUTION
    ket = time_evolution_RM(ket_initial, time_arr, time_span, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_Ham, N_time)
    # END OF TIME EVOLUTION

    # CALCULATION OF WAVEFUNCTION PARAMETERS
    abs2_ket = abs2(ket)
    ket_COM = Full.calc_ket_COM(abs2_ket, N_lat=N_lat, N_state=1, N_band=2)
    # END OF CALCULATION OF WAVEFUNCTION PARAMETERS

    x_Wannier = calc_RM_Wannier_center(period_arr, k_arr, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_lat, N_period)

    E, ket_eigen = diag_RM_time_spectrum(period_arr, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_period, N_Ham)

    # PLOTTING
    plotN_RM_all(E, abs2_ket, x_Wannier, ket_COM, period_arr, time_arr, T_pump, N_Ham)
    # END OF PLOTTING


def Floquet_band_structure(u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_lat, N_period):
    # DERIVED PARAMETERS
    k_arr = np.linspace(-pi, pi, N_lat)
    period_arr = np.linspace(0., T_pump, N_period)
    # END OF DERIVED PARAMETERS

    # RM MODEL
    calc_momentum_Hamiltonian = functools.partial(calc_RM_k_Hamiltonian, u_bar=u_bar, v_bar=v_bar, w_0=w_0, J_0=J_0, T_pump=T_pump, modulation_scheme=modulation_scheme)
    E_Floquet, ket_Floquet = Full_k.diag_k_time_evolution_operator(calc_momentum_Hamiltonian, period_arr, k_arr, N_band=2)
    # E_Floquet.shape = (N_lat, 2)
    # ket_Floquet.shape = (N_lat, 2, 2)

    abs2_ket_Floquet = abs2(ket_Floquet)
    magnetization = abs2_ket_Floquet[..., 0] - abs2_ket_Floquet[..., 1]
    # magnetization.shape = (N_lat, 2)
    # END OF RM MODEL

    # PLOTTING
    Full_k.plot_k_Floquet_Quasienergy(E_Floquet, k_arr, magnetization, file_name="RM_k_Floquet_Quasienergy")
    # Full_k.plot_k_Floquet_Quasienergy(E_Floquet, k_arr, magnetization=None, file_name="RM_k_Floquet_Quasienergy")
    # END OF PLOTTING
# @@@@@@@@@@@@@@@ END OF FULL ROUTINES @@@@@@@@@@@@@@@


# --------------- PLOTTING ---------------
def plot_RM_eigenstate(E, ket_eigen, period_arr, S_period, S_eigen, T_pump, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    cell_arr = np.arange(ket_eigen.shape[-1]) / 2

    S_period = x2index((0., T_pump), ket_eigen.shape[0], period_arr[S_period])

    ax.set_title(r"$t$=%.2f, $E$=%.2f" % (period_arr[S_period], E[S_period, S_eigen]))
    ax.set_xlabel(r"Cell index $m$")

    ax.axhline(0, color="black", ls="--")
    # ax.plot(ket_eigen[S_period, S_eigen], color="red")
    ax.bar(cell_arr[::2], ket_eigen[S_period, S_eigen, ::2], color="red", width=0.5)
    ax.bar(cell_arr[1::2], ket_eigen[S_period, S_eigen, 1::2], color="black", width=0.5)

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("RM_Eigenstate")
        plt.show()


def plot_RM_period_spectrum(E, period_arr, N_Ham, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)
    if fig_ax is None:
        fig.set_size_inches(9, 6)

    T_pump = period_arr[-1]
    ax.set_xlabel(r"t / $T$")
    ax.set_ylabel(r"E / $E_R$")
    [ax.plot(period_arr/T_pump, E[:, i], color="black") for i in range(N_Ham)]
    # [ax.scatter(period_arr/T_pump, E[:, i], color="black", s=4.0) for i in range(N_Ham)]

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("RM_Period_Spectrum")
        plt.show()


def plot_RM_x_Wannier(period_arr, x_Wannier, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    T_pump = period_arr[-1]
    ax.set_xlabel(r"t / $T$")
    ax.set_ylabel(r"$x_W$ / $a_0$")
    ax.plot(period_arr/T_pump, x_Wannier[:, 0], color="black")
    ax.plot(period_arr/T_pump, x_Wannier[:, 1], color="red")

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("RM_x_Wannier")
        plt.show()


def plotN_RM_spectrum_eigen(E, ket_eigen, period_arr, S_period, S_eigen):
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    cell_arr = np.arange(ket_eigen.shape[-1]) / 2

    T_pump = period_arr[-1]

    ax[0].set_xlabel(r"t / $\tau$")
    ax[0].set_ylabel(r"E / $E_R$")
    [ax[0].plot(period_arr/T_pump, E[:, i], color="black") for i in range(E.shape[1])]
    ax[0].scatter(period_arr[S_period]/T_pump, E[S_period, S_eigen], s=30, color="magenta")

    ax[1].set_title(r"$t$=%.2f, $E$=%.2f" % (period_arr[S_period], E[S_period, S_eigen]))
    ax[1].set_xlabel(r"Cell index $m$")
    ax[1].set_ylabel(r"$\psi$")
    # ax[1].yaxis.set_ticks_position("right")
    # ax[1].yaxis.set_label_position("right")
    ax[1].axhline(0, color="black", ls="--")
    # ax[1].plot(ket_eigen[S_period, S_eigen], color="red")
    ax[1].bar(cell_arr[::2], ket_eigen[S_period, S_eigen, ::2], color="red", width=0.5)
    ax[1].bar(cell_arr[1::2], ket_eigen[S_period, S_eigen, 1::2], color="black", width=0.5)

    set_plot_defaults(fig, ax)
    save_plot("RM_Spectrum_Eigen")
    plt.show()


def plotN_RM_all(E, abs2_ket, x_Wannier, ket_COM, period_arr, time_arr, T_pump, N_Ham):
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)

    plot_RM_period_spectrum(E, period_arr, N_Ham, (fig, ax[0, 0]))
    Full.plot_ket_pcolormesh(abs2_ket, N_Ham, time_arr, T_pump, fig_ax=(fig, ax[0, 1]))
    plot_RM_x_Wannier(period_arr, x_Wannier, (fig, ax[1, 0]))
    Full.plot_ket_COM(ket_COM, time_arr, T_pump, fig_ax=(fig, ax[1, 1]))

    set_plot_defaults(fig, ax)
    save_plot("RM_All")
    plt.show()
# --------------- END OF PLOTTING ---------------
