import functools
from math import pi
import numpy as np
from numba import njit
import scipy.linalg
from scipy.integrate import solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt

from QOPack.Math import *
from QOPack.Utility import *
from QOPack.RamanTimeDepDetuning.ZakPhase import calc_general_Berry_phase, smooth_Berry_phase
from QOPack.RamanTimeDepDetuning.ReverseMapping import *
from QOPack.RamanTimeDepDetuning.OverlapTunneling import calc_1Omega_noHO_overlap_tunneling
import QOPack.RamanTimeDepDetuning.TimeEvolutionCRM as CRM
import QOPack.RamanTimeDepDetuning.AdiabaticPumping as AP
import QOPack.RamanTimeDepDetuning.TimeEvolutionFull as Full


# --------------- GENERAL CALCULATIONS ---------------
def diag_k_time_spectrum(calc_momentum_Hamiltonian, period_arr, k_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_period):
    # Diagonalization of momentum Hamiltonians
    # N_lat = N_k
    E_k = np.empty((N_period, N_lat, 2), dtype=np.float64)
    u_k = np.empty((N_period, N_lat, 2, 2), dtype=np.complex128)
    for i in range(N_period):
        for j in range(N_lat):
            Ham_k = calc_momentum_Hamiltonian(period_arr[i], k_arr[j])

            E_k[i, j, :], v = np.linalg.eigh(Ham_k)
            u_k[i, j, :, :] = v.T

    return E_k, u_k


def diag_Full_k_time_spectrum(W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_period, sep_len, k_arr, period_arr):
    calc_momentum_Hamiltonian = functools.partial(calc_Full_k_Hamiltonian, W_H_W_arr=W_H_W_arr, W_W_overlap_arr=W_W_overlap_arr, omega=omega, delta_p=delta_p, gamma=gamma, delta_pm_bar=delta_pm_bar, omega_bar=omega_bar, delta_0_bar=delta_0_bar, T_pump=T_pump, modulate_parameters=modulate_parameters, adiabaticLaunching=adiabaticLaunching, tau_adiabatic=tau_adiabatic, N_state=N_state, N_band=N_band, sep_len=sep_len)

    # Diagonalization of momentum Hamiltonians
    # N_lat = N_k
    E_k = np.empty((N_period, N_lat, N_band), dtype=np.float64)
    u_k = np.empty((N_period, N_lat, N_band, N_band), dtype=np.complex128)
    for i in range(N_period):
        for j in range(N_lat):
            Ham_k = calc_momentum_Hamiltonian(period_arr[i], k_arr[j])
            E_k[i, j, :], v = np.linalg.eigh(Ham_k)
            u_k[i, j, :, :] = v.T

    return E_k, u_k


# @njit(cache=True)
def diag_k_time_evolution_operator(calc_momentum_Hamiltonian, period_arr, k_arr, N_band):
    """By diagonalizing the time evolution operator, one obtains the Floquet energies and eigenstates.

    Time evolution operator is defined by:
    U(t <- t_0) \psi(t_0) = \psi(t);

    Time evolution operator is obtained by:
    U(t <- t_0) = \prod_n U(t_{n+1} <- t_n),
    where:
    1) U(t_{n+1} <- t_n) = \exp(-i * Ham(t_n) * \Delta t / \hbar);
    2) t_n := t_0 + n*\Delta t;
    3) \Delta t := (t_{N-1} - t_0) / (N - 1);"""

    # DERIVED PARAMETERS
    N_lat = len(k_arr)
    delta_time = (period_arr[-1] - period_arr[0]) / (len(period_arr) - 1)
    T_pump = period_arr[-1]
    # END OF DERIVED PARAMETERS

    lambda_Floquet = np.empty((N_lat, N_band), dtype=np.complex128)
    E_Floquet = np.empty((N_lat, N_band), dtype=np.float64)
    ket_Floquet = np.empty((N_lat, N_band, N_band), dtype=np.complex128)

    ########## FROM TIME EVOLUTION OPERATOR ##########
    # for idx_k, k in enumerate(k_arr):
    #     # SINGLE PERIOD TIME EVOLUTION OPERATOR
    #     U_time_evo_k = np.identity(N_band, dtype=np.complex128)
    #     for time in period_arr:
    #         Ham_k = calc_momentum_Hamiltonian(time, k)
    #         U_time_evo_k = scipy.linalg.expm(-1j * Ham_k * delta_time) @ U_time_evo_k
    #     # END OF SINGLE PERIOD TIME EVOLUTION OPERATOR

    #     lambda_Floquet[idx_k, :], v = np.linalg.eig(U_time_evo_k)
    #     ket_Floquet[idx_k, :, :] = v.T

    # # E_Floquet = 1j * np.log(lambda_Floquet)
    # E_Floquet = -np.imag(np.log(lambda_Floquet))
    ##################################################

    ########## FROM EFFECTIVE HAMILTONIAN ##########
    for idx_k, k in enumerate(k_arr):
        # SINGLE PERIOD TIME EVOLUTION OPERATOR
        U_time_evo_k = np.identity(N_band, dtype=np.complex128)
        for time in period_arr:
            Ham_k = calc_momentum_Hamiltonian(time, k)
            U_time_evo_k = scipy.linalg.expm(-1j * Ham_k * delta_time) @ U_time_evo_k
        # END OF SINGLE PERIOD TIME EVOLUTION OPERATOR

        H_eff = 1j * scipy.linalg.logm(U_time_evo_k)
        # check_if_Hermitian(H_eff, "H_eff")

        E_Floquet[idx_k, :], v = np.linalg.eigh(H_eff)
        ket_Floquet[idx_k, :, :] = v.T
    ################################################

    # E_Floquet.shape = (N_lat, N_band)
    # ket_Floquet.shape = (N_lat, N_band, N_band)
    return E_Floquet, ket_Floquet


def calc_initial_time_Floquet_abs2_overlaps(u_k, ket_Floquet):
    return abs2(np.sum(np.conj(u_k[0, :, :, np.newaxis, :]) * ket_Floquet[:, np.newaxis, :, :], axis=-1))


def calc_period_averaged_Floquet_abs2_overlaps(u_k, ket_Floquet):
    return abs2(np.sum(np.conj(u_k[:, :, :, np.newaxis, :]) * ket_Floquet[np.newaxis, :, np.newaxis, :, :], axis=(0, -1)))


@njit(cache=True)
def calc_magnetization(Floquet_abs2_overlaps):
    return Floquet_abs2_overlaps[:, 0, :] - Floquet_abs2_overlaps[:, 1, :]
# --------------- END OF GENERAL CALCULATIONS ---------------


# --------------- SPECIFIC CALCULATIONS ---------------
@njit(cache=True)
def calc_Full_k_Hamiltonian(t, k, W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, sep_len):
    if adiabaticLaunching:
        alpha_t = Full.calc_alpha_t(t, tau_adiabatic)
    else:
        alpha_t = 1.

    # PARAMETER MODULATION
    delta_pm_t, omega_t, delta_0_t = Full.modulate_Full_parameters(t, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, alpha_t)
    # END OF PARAMETER MODULATION

    # Set detuning values
    delta_p_t = np.empty(3, dtype=np.float64)
    delta_p_t[-1] = delta_pm_t[0]
    delta_p_t[0] = delta_0_t
    delta_p_t[1] = delta_pm_t[1]

    Ham_k = np.zeros((N_band, N_band), dtype=np.complex128)

    # --------------- H_k ASSIGNMENT ---------------
    # TODO: Only nearest neighbours considered.
    # H_0 ASSIGNMENT
    # On-site (no spatial shift) terms.
    for i in range(N_band):  # N_band -> [s, p, d, ...]
        Ham_k[i, i] += W_H_W_arr[i, i, 0] - i*omega_t

    # Natural tunneling terms.
    for i in range(N_band):  # N_band -> [s, p, d, ...]
        for j in range(1, sep_len):  # sep_len -> [0, a_0, 2*a_0, ...]
            Ham_k[i, i] += 2 * W_H_W_arr[i, i, j] * np.cos(j*N_state * k)
    # END OF H_0 ASSIGNMENT

    # U ASSIGNMENT
    F = Full.calc_F(t, omega_t, delta_p_t, gamma, alpha_t)

    for a in range(N_band):  # N_band -> [s, p, d, ...]
        for b in range(N_band):  # N_band -> [s, p, d, ...]
            U_coupling_term = F * np.exp(1j*k) * np.exp(1j*(b-a)*omega_t*t) * W_W_overlap_arr[b, a, 1]
            U_coupling_conj_term = np.conj(U_coupling_term)

            Ham_k[a, b] += U_coupling_term
            Ham_k[b, a] += U_coupling_conj_term
    # END OF U ASSIGNMENT
    # --------------- END OF H_k ASSIGNMENT ---------------

    return Ham_k  # (N_band, N_band)


def calc_general_x_Wannier(u_k, k_axis):
    """Wannier center x_Wannier is the Zak phase, divided by 2*pi and multiplied
    by the lattice period a_0, which is assumed to be 1."""
    Zak_arr = calc_general_Berry_phase(u_k, prod_axis=k_axis)  # (N_period, N_band)
    Zak_arr = smooth_Berry_phase(np.swapaxes(Zak_arr, 0, 1))
    Zak_arr = np.swapaxes(Zak_arr, 0, 1)
    # Zak_arr.shape: (N_period, N_band) -> (N_band, N_period) -> (N_period, N_band)

    x_Wannier = Zak_arr / (2*pi)  # (N_period, N_band)

    return x_Wannier
# --------------- END OF SPECIFIC CALCULATIONS ---------------


# @@@@@@@@@@@@@@@ FULL ROUTINES @@@@@@@@@@@@@@@
# TODO: Does not work.
def full_Full_k_Wannier_center(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_x, x_max, max_sep, N_time, time_span, N_period, period_selected, S_eigen, colors, params_selected):
    # DERIVED PARAMETERS
    N_Ham = N_band * N_lat

    sep_len = int(max_sep) + 1

    k_arr = np.linspace(-pi, pi, N_lat)
    period_arr = np.linspace(0., T_pump, N_period)

    S_period = x2index((0., T_pump), N_period, period_selected)

    delta_pm_sum_period, omega_period, delta_0_period = Full.calc_Full_parameters_period(omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, N_period, period_arr)
    # END OF DERIVED PARAMETERS

    # OVERLAPS AND HAMILTONIAN MATRIX ELEMENTS
    W_W_overlap_arr, W_H_W_arr = calc_1Omega_noHO_overlap_tunneling(Omega, max_sep, N_state, N_band)
    # END OF OVERLAPS AND HAMILTONIAN MATRIX ELEMENTS

    # MOMENTUM TIME DIAGONALIZATION
    E_k, u_k = diag_Full_k_time_spectrum(W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_period, sep_len, k_arr, period_arr)
    E_momentum = np.reshape(E_k, (N_period, N_Ham))
    # END OF MOMENTUM TIME DIAGONALIZATION

    # WANNIER CENTER
    x_Wannier = calc_general_x_Wannier(u_k, 1)
    # END OF WANNIER CENTER

    # PLOTTING
    plotN_Full_momentum_all(E_momentum, E_k, x_Wannier, delta_pm_sum_period, omega_period, delta_0_period, colors, period_arr, k_arr, T_pump, S_period, S_eigen)
    # END OF PLOTTING


def compare_Full_momentum_to_real(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_x, x_max, max_sep, N_time, time_span, N_period, period_selected, S_eigen, colors, params_selected):
    # DERIVED PARAMETERS
    N_Ham = N_band * N_lat

    sep_len = int(max_sep) + 1

    k_arr = np.linspace(-pi, pi, N_lat)
    time_arr = np.linspace(*time_span, N_time)
    period_arr = np.linspace(0., T_pump, N_period)

    S_period = x2index((0., T_pump), N_period, period_selected)

    delta_pm_sum_period, omega_period, delta_0_period = Full.calc_Full_parameters_period(omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, N_period, period_arr)
    # END OF DERIVED PARAMETERS

    # OVERLAPS AND HAMILTONIAN MATRIX ELEMENTS
    W_W_overlap_arr, W_H_W_arr = calc_1Omega_noHO_overlap_tunneling(Omega, max_sep, N_state, N_band)
    # END OF OVERLAPS AND HAMILTONIAN MATRIX ELEMENTS

    # *************** REAL SPACE CALCULATIONS ***************
    # INITIAL CONDITION
    # psi_initial = create_centered_single(N_band, N_lat, N_Ham)
    # END OF INITIAL CONDITION

    # TIME DIAGONALIZATION
    E_real, psi_eigen = Full.diag_Full_time_spectrum(W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, N_state, N_band, N_lat, N_period, N_Ham, sep_len, period_arr)
    # abs2_psi_eigen = abs2(psi_eigen)
    # END OF TIME DIAGONALIZATION

    # TIME EVOLUTION
    # psi = calc_Full_psi_t(psi_initial, W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, N_time, time_span, N_state, N_band, N_lat, N_Ham, sep_len, time_arr)
    # abs2_psi = abs2(psi)
    # END OF TIME EVOLUTION

    # ADDITIONAL CALCULATIONS
    # psi_COM = calc_psi_COM(abs2_psi, N_state, N_band, N_lat)
    # END OF ADDITIONAL CALCULATIONS
    # *************** END OF REAL SPACE CALCULATIONS ***************

    # *************** MOMENTUM SPACE CALCULATIONS ***************
    # MOMENTUM TIME DIAGONALIZATION
    E_k, u_k = diag_Full_k_time_spectrum(W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_period, sep_len, k_arr, period_arr)
    E_momentum = np.reshape(E_k, (N_period, N_Ham))
    # END OF MOMENTUM TIME DIAGONALIZATION
    # *************** END OF MOMENTUM SPACE CALCULATIONS ***************

    # PLOTTING
    plotN_Full_real_momentum_comparison(E_real, E_momentum, delta_pm_sum_period, omega_period, delta_0_period, period_arr, params_selected, S_eigen, S_period)
    # END OF PLOTTING


def compare_Full_k_to_CRM_k(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, max_sep, N_period, period_selected, colors):
    # DERIVED PARAMETERS
    sep_len = int(max_sep) + 1

    k_arr = np.linspace(-pi, pi, N_lat)
    period_arr = np.linspace(0., T_pump, N_period)

    S_period = x2index((0., T_pump), N_period, period_selected)
    # END OF DERIVED PARAMETERS

    # OVERLAPS AND HAMILTONIAN MATRIX ELEMENTS
    W_W_overlap_arr, W_H_W_arr = calc_1Omega_noHO_overlap_tunneling(Omega, max_sep, N_state, N_band)
    # END OF OVERLAPS AND HAMILTONIAN MATRIX ELEMENTS

    # FULL MOMENTUM SPACE MODEL
    E_k_Full, u_k = diag_Full_k_time_spectrum(W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_period, sep_len, k_arr, period_arr)
    # END OF FULL MOMENTUM SPACE MODEL

    # CRM MOMENTUM SPACE MODEL
    t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar = Full2CRM(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, N_state, printResult=True)

    # Note that these CRM results are derived analytically and therefore do not account for the first order Floquet correction.
    epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time = AP.full_modulate(epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, (0., 1.), modulate_parameters, N_period)
    E_k_CRM = AP.full_exact_E_kt(epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time, gamma, (-pi, pi), N_state, N_lat, N_period)
    # END OF CRM MOMENTUM SPACE MODEL

    # PLOTTING
    plotN_Full_k_CRM_k_comparison(E_k_Full, E_k_CRM, k_arr, period_arr, colors, S_period)
    # END OF PLOTTING


def Floquet_band_structure(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, max_sep, N_period, period_selected, colors):
    # DERIVED PARAMETERS
    sep_len = int(max_sep) + 1

    k_arr = np.linspace(-pi, pi, N_lat)
    period_arr = np.linspace(0., T_pump, N_period)
    # END OF DERIVED PARAMETERS

    # OVERLAPS AND HAMILTONIAN MATRIX ELEMENTS
    W_W_overlap_arr, W_H_W_arr = calc_1Omega_noHO_overlap_tunneling(Omega, max_sep, N_state, N_band)
    # END OF OVERLAPS AND HAMILTONIAN MATRIX ELEMENTS

    # FULL MODEL
    calc_momentum_Hamiltonian = functools.partial(calc_Full_k_Hamiltonian, W_H_W_arr=W_H_W_arr, W_W_overlap_arr=W_W_overlap_arr, omega=omega, delta_p=delta_p, gamma=gamma, delta_pm_bar=delta_pm_bar, omega_bar=omega_bar, delta_0_bar=delta_0_bar, T_pump=T_pump, modulate_parameters=modulate_parameters, adiabaticLaunching=adiabaticLaunching, tau_adiabatic=tau_adiabatic, N_state=N_state, N_band=N_band, sep_len=sep_len)
    E_Floquet, ket_Floquet = diag_k_time_evolution_operator(calc_momentum_Hamiltonian, period_arr, k_arr, N_band)
    # E_Floquet.shape = (N_lat, N_band)
    # ket_Floquet.shape = (N_lat, N_band, N_band)

    abs2_ket_Floquet = abs2(ket_Floquet)
    # Assumes ket_Floquet has the following structure: [A A B B A A ...];
    # magnetization = np.zeros((N_lat, N_band), dtype=np.float64)
    # for idx_lat in range(N_lat):
    #     for idx_sublat in range(N_band):

    # TODO: Probably incorrect, fix.
    # magnetization = abs2_ket_Floquet[..., 0] - abs2_ket_Floquet[..., 1]

    # FLOQUET ORDERING (HOLTHAUS)
    # u_k.shape = (N_period, N_lat, N_band, N_band)
    E_k, u_k = diag_Full_k_time_spectrum(W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_period, sep_len, k_arr, period_arr)
    # Based on ordering idea from Holthaus2015.
    # How much abs2_overlap the Floquet eigenstates have with each of the initial time eigenstates?
    # Floquet_abs2_overlaps.shape = (N_lat, N_band, N_band)
    #                               (idx_k, idx_u, idx_Floquet)

    # Floquet_abs2_overlaps = calc_initial_time_Floquet_abs2_overlaps(u_k, ket_Floquet)  # INITIAL TIME PROJECTION
    Floquet_abs2_overlaps = calc_period_averaged_Floquet_abs2_overlaps(u_k, ket_Floquet)  # PERIOD AVERAGED PROJECTION

    # magnetization.shape = (N_lat, N_band)
    magnetization = calc_magnetization(Floquet_abs2_overlaps)

    # # Which of the Floquet states has the most abs2_overlap with the initial time eigenstates?
    # # magnetization = np.empty((N_lat, N_band), dtype=np.float64)  # WRONG
    # Floquet_order = np.empty((N_lat, N_band), dtype=np.int64)
    # for idx_k in range(N_lat):
    #     for idx_u in range(N_band):
    #         Floquet_order[idx_k, idx_u] = np.argmax(Floquet_abs2_overlaps[idx_k, idx_u, :])
    #         # magnetization[idx_k, idx_u] = np.amax(Floquet_abs2_overlaps[idx_k, idx_u, :])

    # # The "first" Floquet state is that which has most overlap with the lowest band initial time eigenstate, the "second" - with the second lowest band and so on.
    # temp_E_Floquet = E_Floquet
    # for idx_k in range(N_lat):
    #     for idx_u in range(N_band):
    #         E_Floquet[idx_k, idx_u] = temp_E_Floquet[idx_k, Floquet_order[idx_k, idx_u]]
    # END OF FLOQUET ORDERING (HOLTHAUS)

    # FLOQUET ORDERING (ENERGY)
    # END OF FLOQUET ORDERING (ENERGY)
    # END OF FULL MODEL

    # PLOTTING
    # plot_k_Floquet_Quasienergy(E_Floquet, k_arr, magnetization)
    # plot_k_Floquet_Quasienergy(E_Floquet[:, 0:2], k_arr, magnetization)

    # plot_k_Floquet_Quasienergy(E_Floquet, k_arr)
    plot_k_Floquet_Quasienergy(E_Floquet, k_arr, magnetization)
    # plot_k_Floquet_Quasienergy(E_Floquet[:, 0:2], k_arr)
    # END OF PLOTTING


def Floquet_bands_Full_Full2_CRM(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, max_sep, N_period, period_selected, colors):
    # DERIVED PARAMETERS
    sep_len = int(max_sep) + 1

    k_arr = np.linspace(-pi, pi, N_lat)
    period_arr = np.linspace(0.0*T_pump, 1.0*T_pump, N_period)
    # END OF DERIVED PARAMETERS

    # OVERLAPS AND HAMILTONIAN MATRIX ELEMENTS
    W_W_overlap_arr, W_H_W_arr = calc_1Omega_noHO_overlap_tunneling(Omega, max_sep, N_state, N_band)
    # END OF OVERLAPS AND HAMILTONIAN MATRIX ELEMENTS

    # FULL MODEL
    calc_momentum_Hamiltonian = functools.partial(calc_Full_k_Hamiltonian, W_H_W_arr=W_H_W_arr, W_W_overlap_arr=W_W_overlap_arr, omega=omega, delta_p=delta_p, gamma=gamma, delta_pm_bar=delta_pm_bar, omega_bar=omega_bar, delta_0_bar=delta_0_bar, T_pump=T_pump, modulate_parameters=modulate_parameters, adiabaticLaunching=adiabaticLaunching, tau_adiabatic=tau_adiabatic, N_state=N_state, N_band=N_band, sep_len=sep_len)
    E_Floquet_Full, ket_Floquet_Full = diag_k_time_evolution_operator(calc_momentum_Hamiltonian, period_arr, k_arr, N_band)

    E_k_Full, u_k_Full = diag_Full_k_time_spectrum(W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_period, sep_len, k_arr, period_arr)
    # Floquet_abs2_overlaps_Full = calc_initial_time_Floquet_abs2_overlaps(u_k_Full, ket_Floquet_Full)
    Floquet_abs2_overlaps_Full = calc_period_averaged_Floquet_abs2_overlaps(u_k_Full, ket_Floquet_Full)
    magnetization_Full = calc_magnetization(Floquet_abs2_overlaps_Full)
    # END OF FULL MODEL

    # FULL MODEL (2 BANDS)
    calc_momentum_Hamiltonian = functools.partial(calc_Full_k_Hamiltonian, W_H_W_arr=W_H_W_arr, W_W_overlap_arr=W_W_overlap_arr, omega=omega, delta_p=delta_p, gamma=gamma, delta_pm_bar=delta_pm_bar, omega_bar=omega_bar, delta_0_bar=delta_0_bar, T_pump=T_pump, modulate_parameters=modulate_parameters, adiabaticLaunching=adiabaticLaunching, tau_adiabatic=tau_adiabatic, N_state=N_state, N_band=2, sep_len=sep_len)
    E_Floquet_Full2, ket_Floquet_Full2 = diag_k_time_evolution_operator(calc_momentum_Hamiltonian, period_arr, k_arr, 2)

    E_k_Full2, u_k_Full2 = diag_Full_k_time_spectrum(W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, 2, N_lat, N_period, sep_len, k_arr, period_arr)
    # Floquet_abs2_overlaps_Full2 = calc_initial_time_Floquet_abs2_overlaps(u_k_Full2, ket_Floquet_Full2)
    Floquet_abs2_overlaps_Full2 = calc_period_averaged_Floquet_abs2_overlaps(u_k_Full2, ket_Floquet_Full2)
    magnetization_Full2 = calc_magnetization(Floquet_abs2_overlaps_Full2)
    # END OF FULL MODEL (2 BANDS)

    # CRM MODEL
    t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar = Full2CRM(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, N_state, printResult=False)
    calc_momentum_Hamiltonian = functools.partial(CRM.calc_CRM_k_Hamiltonian, t_bar=t_bar, epsilon_bar=epsilon_bar, t_0_bar=t_0_bar, t_pm=t_pm, epsilon=epsilon, t_0alpha=t_0alpha, tn_alpha=tn_alpha, gamma=gamma, J_0=0, v_bar=0, w_0=0, T_pump=T_pump, modulate_parameters=modulate_parameters, modulation_scheme="Gediminas", adiabaticLaunching=adiabaticLaunching, tau_adiabatic=tau_adiabatic, N_state=N_state)
    E_Floquet_CRM, ket_Floquet_CRM = diag_k_time_evolution_operator(calc_momentum_Hamiltonian, period_arr, k_arr, N_band=2)

    E_k_CRM, u_k_CRM = diag_k_time_spectrum(calc_momentum_Hamiltonian, period_arr, k_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, 0, 0, 0, T_pump, modulate_parameters, "Gediminas", adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_period)
    # Floquet_abs2_overlaps_CRM = calc_initial_time_Floquet_abs2_overlaps(u_k_CRM, ket_Floquet_CRM)
    Floquet_abs2_overlaps_CRM = calc_period_averaged_Floquet_abs2_overlaps(u_k_CRM, ket_Floquet_CRM)
    magnetization_CRM = calc_magnetization(Floquet_abs2_overlaps_CRM)
    # END OF CRM MODEL

    # PLOTTING
    # plotN_Floquet_bands_Full_Full2_CRM(E_Floquet_Full, E_Floquet_Full2, E_Floquet_CRM, k_arr)
    plotN_Floquet_bands_Full_Full2_CRM(E_Floquet_Full, E_Floquet_Full2, E_Floquet_CRM, k_arr, magnetization_Full, magnetization_Full2, magnetization_CRM)
    # END OF PLOTTING
# @@@@@@@@@@@@@@@ END OF FULL ROUTINES @@@@@@@@@@@@@@@


# --------------- PLOTTING ---------------
# SINGLE PLOTS
def plot_u_k_components(k_arr, u_k, colors, S_period, S_band, N_band, fig_ax=None):
    """Numerically obtained eigenvector u_k will not have a smooth gauge, but if
    one uses a gauge-invariant formulas, this is irrelevant."""
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    ax[0].set_title(r"Re")
    # ax[1].set_xlabel(r"k / $k_0$")
    ax[0].set_xlabel(r"$k$ / rad")
    ax[0].set_ylabel(r"Re $\langle \alpha|u_k \rangle$")
    [ax[0].plot(k_arr, np.real(u_k[S_period, :, S_band, i]), color=colors[i]) for i in range(N_band)]

    ax[1].set_title(r"Im")
    ax[1].set_xlabel(r"$k$ / rad")
    ax[1].set_ylabel(r"Im $\langle \alpha|u_k \rangle$")
    [ax[1].plot(k_arr, np.imag(u_k[S_period, :, S_band, i]), color=colors[i]) for i in range(N_band)]

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("u_k_Components")
        plt.show()


def plot_Full_k_Wannier_center(period_arr, x_Wannier, S_band, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    T_pump = period_arr[-1]
    ax.set_xlabel(r"t / $T$")
    ax.set_ylabel(r"$x_W$ / $a$")
    ax.plot(period_arr/T_pump, x_Wannier[:, S_band], color='black')

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("Full_k_Wannier_center")
        plt.show()


def plot_Full_k_Energy(k_arr, E_k, colors, period_arr, S_period, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    ax.set_title(r"$t$=%.2f" % period_arr[S_period])
    ax.set_xlabel(r"$k$ / rad")
    ax.set_ylabel(r"E / $E_R$")
    [ax.plot(k_arr, E_k[S_period, :, i], color=colors[i]) for i in range(E_k.shape[-1])]

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("Full_k_Energy")
        plt.show()


def plot_k_Floquet_Quasienergy(E_Floquet, k_arr, magnetization=None, fig_ax=None, file_name="k_Floquet_Quasienergy"):
    """If magnetization=None, choose color by eigenstate number."""
    fig, ax = get_fig_ax(fig_ax)

    normed_E_Floquet = E_Floquet / (2*pi)
    size = 5

    ax.set_ylim(-0.5, 0.5)

    ax.set_xlabel(r"$k$ / rad")
    ax.set_ylabel(r"$ϵT$ / $2\pi$")
    if magnetization is None:
        # colors = ["red", "black", "blue", "green", "purple", "magenta", "lime"]
        # [ax.scatter(k_arr, normed_E_Floquet[:, i], color=colors[i], s=size) for i in range(normed_E_Floquet.shape[-1])]

        [ax.scatter(k_arr, normed_E_Floquet[:, i], color="black", s=size) for i in range(normed_E_Floquet.shape[-1])]
    else:
        ## scatter for colorbar settings only.
        scatter = ax.scatter(k_arr[0], normed_E_Floquet[0, 0], s=0., c=0., cmap="viridis",
                            norm=mpl.colors.Normalize(vmin=-1., vmax=1.))
                            # norm=mpl.colors.Normalize(vmin=magnetization.min(), vmax=magnetization.max()))
        [ax.scatter(k_arr, normed_E_Floquet[:, i], s=size, c=magnetization[:, i], cmap="viridis")  # "bwr"
        for i in range(normed_E_Floquet.shape[-1])]

        cbar = plt.colorbar(scatter, label=r"Magnetization $\langle \hat{F}_x \rangle$", location="right", aspect=30, ax=ax)
        cbar.ax.xaxis.label.set_size(12)
        cbar.ax.tick_params(which="both", length=2, direction="in")
        cbar.ax.xaxis.set_ticks_position("both")

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot(file_name)
        plt.show()
# END OF SINGLE PLOTS


# COMPOSITE PLOTS
def plotN_Full_momentum_all(E_momentum, E_k, x_Wannier, delta_pm_sum_period, omega_period, delta_0_period, colors, period_arr, k_arr, T_pump, S_period, S_eigen):
    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(22, 4)

    Full.plot_Full_modulation(delta_pm_sum_period, omega_period, delta_0_period, period_arr, T_pump, (fig, ax[0]))
    Full.plot_Full_spectrum(E_momentum, S_eigen, S_period, period_arr, T_pump, (fig, ax[1]))
    plot_Full_k_Energy(k_arr, E_k, colors, period_arr, S_period, (fig, ax[2]))
    plot_Full_k_Wannier_center(period_arr, x_Wannier, S_band=0, fig_ax=(fig, ax[3]))

    set_plot_defaults(fig, ax)
    save_plot("Full_Momentum_All")
    plt.show()


def plotN_Full_real_momentum_comparison(E_real, E_momentum, delta_pm_sum_period, omega_period, delta_0_period, period_arr, params_selected, S_eigen, S_period):
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)
    fig.suptitle('Top - real, bottom - momentum.')

    Full.plot_Full_modulation(delta_pm_sum_period, omega_period, delta_0_period, period_arr, (fig, ax[0, 0]))
    Full.plot_Full_spectrum(E_real, S_eigen, S_period, period_arr, (fig, ax[0, 1]))
    Full.plot_Full_parameter_path(delta_pm_sum_period, omega_period, delta_0_period, params_selected, (fig, ax[1, 0]))
    Full.plot_Full_spectrum(E_momentum, S_eigen, S_period, period_arr, (fig, ax[1, 1]))

    set_plot_defaults(fig, ax)
    save_plot("Full_Real_Momentum_Comparison")
    plt.show()


def plotN_Full_k_CRM_k_comparison(E_k_Full, E_k_CRM, k_arr, period_arr, colors, S_period):
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    plot_Full_k_Energy(k_arr, E_k_Full, colors, period_arr, S_period, (fig, ax[0]))
    AP.plot_CRM_Energy_kt(E_k_CRM, k_arr, period_arr, colors, S_period, (fig, ax[1]))

    set_plot_defaults(fig, ax)
    save_plot("Full_k_CRM_k_comparison")
    plt.show()


def plotN_Floquet_bands_Full_Full2_CRM(E_Floquet_Full, E_Floquet_Full2, E_Floquet_CRM, k_arr, magnetization_Full=None, magnetization_Full2=None, magnetization_CRM=None, file_name="Floquet_bands_Full_Full2_CRM"):
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(18, 5)

    # Taken from https://stackoverflow.com/questions/25239933/how-to-add-title-to-subplots-in-matplotlib
    ax[0].title.set_text(r"Full%i" % E_Floquet_Full.shape[-1])
    ax[1].title.set_text("Full2")
    ax[2].title.set_text("CRM")

    plot_k_Floquet_Quasienergy(E_Floquet_Full, k_arr, magnetization_Full, fig_ax=(fig, ax[0]))
    plot_k_Floquet_Quasienergy(E_Floquet_Full2, k_arr, magnetization_Full2, fig_ax=(fig, ax[1]))
    plot_k_Floquet_Quasienergy(E_Floquet_CRM, k_arr, magnetization_CRM, fig_ax=(fig, ax[2]))

    set_plot_defaults(fig, ax)
    save_plot(file_name)
    plt.show()
# END OF COMPOSITE PLOTS
# --------------- END OF PLOTTING ---------------
