import functools
from math import pi
from datetime import datetime
import numpy as np
from numba import njit
import scipy.linalg
from scipy.integrate import solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt

from QOPack.Math import *
from QOPack.Utility import *
from QOPack.RamanTimeDepDetuning.ZakPhase import create_Delta_points, full_energy_gaps, exact_Wannier_pm
from QOPack.RamanTimeDepDetuning.ReverseMapping import *
from QOPack.RamanTimeDepDetuning.OverlapTunneling import calc_1Omega_noHO_overlap_tunneling
import QOPack.RamanTimeDepDetuning.AdiabaticPumping as AP
# from QOPack.RamanTimeDepDetuning.AdiabaticPumping import *
import QOPack.RamanTimeDepDetuning.TimeEvolutionCRM as CRM

# plt.style.use("../matplotlibrc")
plt.style.use("default")

# This program solves the Full model in the Wannier basis (specifically, sinusoidal lattice Wannier
# functions centered at x=n, where n is an integer and the lattice period is assumed to be 1).

# This program calculates energies and eigenstates of the Full model Hamiltonian by
# exact diagonalization and time evolution of chosen initial conditions.

# There is also functionality to compare the results of this program with other programs
# (namely, 1) the coupled Rice-Mele (CRM) Wannier basis program; 2) the Full model in
# momentum space (Bloch basis) program).
# The results of this program (Full model in the Wannier basis) should perfectly agree
# with the results of the Full model in momentum space EXCLUDING boundary effects (e.g.,
# edge states).
# The results of this program (Full model in the Wannier basis) should be close to
# the results of the CRM Wannier basis program (when the necessary physical conditions
# are met) since the CRM model is an approximation of the Full model, taking into
# account only the two lowest bands (s and p) and a time averaged zero order Floquet
# picture (all Hamiltonian elements oscillating with a frequency of \omega or
# greater are discarded).


# --------------- GENERAL CALCULATIONS ---------------
def solve_Schrodinger(calc_Hamiltonian, ket_initial, time_span, N_time, N_Ham, time_arr):
    def Schrodinger(t, ket):
        Ham = calc_Hamiltonian(t)
        return -1.j * Ham @ ket

    ## Set rtol and atol to keep relatively good normalization.
    ## LSODA does not work for complex values.
    # sol = solve_ivp(Schrodinger, time_span, ket_initial, t_eval=time_arr)
    # sol = solve_ivp(Schrodinger, time_span, ket_initial, t_eval=time_arr, rtol=1e-5, atol=1e-8)
    sol = solve_ivp(Schrodinger, time_span, ket_initial, t_eval=time_arr, rtol=1e-8, atol=1e-12)
    # sol = solve_ivp(Schrodinger, time_span, ket_initial, method="BDF", t_eval=time_arr, rtol=1e-8, atol=1e-12)
    # sol = solve_ivp(Schrodinger, time_span, ket_initial, t_eval=time_arr, rtol=1e-10, atol=1e-14)
    # print(sol.success)
    # print(np.shape(ket_initial))
    # print(np.shape(sol.y))

    # SAVING SOLUTION INTO 2D ARRAY.
    ket = np.zeros((N_time, N_Ham), dtype=np.complex128)
    for j in range(N_Ham):
        ket[:, j] = sol.y[j]
    # END OF SAVING SOLUTION INTO 2D ARRAY.

    ## Checking normalization
    # check_ket_normalization(ket, 'ket')
    # print('ket_norm:', np.sum(abs2(ket[-10:]), axis=-1))

    return ket


@njit(cache=True)
def calc_band_pop(ket, ket_eigen, N_band, N_lat, N_time):
    """Assumes that bands have equal number (N_lat) of unique energy eigenvalues
    (and associated eigenstates). This assumption includes edge state populations
    into certain bands, but as long as edge state populations are small, it
    should not matter."""
    band_pop = np.zeros((N_band, N_time), dtype=np.float64)

    ## ket.shape = (N_time, N_Ham), where N_Ham = N_band*N_lat
    ## ket_eigen.shape = (N_time, N_Ham, N_Ham)
    # for i in range(N_band):
    #     for j in range(N_lat):
    #         band_pop[i, :] += np.sum(abs2(np.conj(ket_eigen[:, i*N_lat+j, :]) * ket), axis=-1)

    for i in range(N_band):
        for j in range(N_time):
            for k in range(N_lat):
                band_pop[i, j] += abs2(np.sum(np.conj(ket_eigen[j, i*N_lat+k, :]) * ket[j, :], axis=-1))

    ## DEBUG
    # print(abs2(np.sum(np.conj(ket_eigen[0, 0, :]) * ket_eigen[0, 0, :], axis=-1)))
    # print(abs2(np.sum(np.conj(ket_eigen[0, 0, :]) * ket[0, :], axis=-1)))

    ## Total population should remain close to 1.
    # print(np.sum(band_pop, axis=0))
    ## END OF DEBUG

    return band_pop  # (N_band, N_time)


@njit(cache=True)
def calc_ket_COM(abs2_ket, N_state, N_band, N_lat):
    x_arr = np.arange(N_lat) - N_lat // 2
    x_arr = np.repeat(x_arr, N_band)

    ## x_mul_ket2.shape = (N_time, N_Ham)
    x_mul_ket2 = np.expand_dims(x_arr, axis=0) * abs2_ket

    # ket_COM = np.sum(x_mul_ket2, axis=-1) / N_state  # In terms of a_0
    ket_COM = np.sum(x_mul_ket2, axis=-1)  # In terms of a

    return ket_COM


@njit(cache=True)
def calc_IPR(ket_eigen, component_axis=-1):
    ## DEBUG
    # print(abs2(ket_eigen)[30, 100, :])
    # print(np.sum(abs4(ket_eigen), axis=-1)[30, 100])
    ## END OF DEBUG

    # return np.sum(abs2(abs2(ket_eigen[..., ::2]) + abs2(ket_eigen[..., 1::2])), axis=-1)
    return np.sum(abs4(ket_eigen), axis=component_axis)
# --------------- END OF GENERAL CALCULATIONS ---------------


# --------------- INITIAL CONDITION FUNCTIONS ---------------
@njit(cache=True)
def create_centered_single_sublat(N_lat, N_Ham, N_band, idx_sublat=0):
    ket_initial = np.zeros(N_Ham, dtype=np.complex128)
    # idx_sublat=0 -> s band; idx_sublat=1 -> p band; ...
    ket_initial[N_band*(N_lat//2)+idx_sublat] = 1.0

    return ket_initial


@njit(cache=True)
def create_centered_single_lat(N_lat, N_Ham, N_band):
    ket_initial = np.zeros(N_Ham, dtype=np.complex128)
    for idx_sublat in range(N_band):
        ket_initial[N_band*(N_lat//2)+idx_sublat] = 1.0 / np.sqrt(N_band)

    return ket_initial


@njit(cache=True)
def create_centered_eigen(ket_eigen, N_lat, N_Ham, N_band):
    # Two equivalent options: N_band*(N_lat//2) and N_band*(N_lat//2)+1
    # Choose which one is better

    # If one Wannier function is not localized, try another localization center.
    for a in range(N_band):
        Re_ket0_eigen = np.real(ket_eigen[0, :, N_band*(N_lat//2)+a])
        Im_ket0_eigen = np.imag(ket_eigen[0, :, N_band*(N_lat//2)+a])

        phi = np.arctan2(Im_ket0_eigen, Re_ket0_eigen)

        ket_initial = np.zeros(N_Ham, dtype=np.complex128)
        for i in range(N_lat-2):
        # for i in range(N_lat):
        # for i in range(N_lat, 2*N_lat):
            ket_initial += np.exp(-1j * phi[i]) * ket_eigen[0, i, :]

        ## Normalization
        ket_initial /= np.sqrt(np.sum(abs2(ket_initial)))

        # If center third is not 75% of population, discard the initial state.
        center_third_pop = np.sum(abs2(ket_initial[N_Ham//3:2*N_Ham//3]))
        if center_third_pop > 0.75:
            break
            # print("a:", a)

    return ket_initial


@njit(cache=True)
def create_real_centered_eigen(ket_eigen, N_lat, N_Ham):
    ket_initial = np.zeros(N_Ham, dtype=np.complex128)
    for i in range(N_lat):
    # for i in range(N_lat, N_Ham):
        if np.real(ket_eigen[0, i, 2*(N_lat//2)]) > 0.:
            ket_initial += ket_eigen[0, i, :]
        else:
            ket_initial -= ket_eigen[0, i, :]

    # Normalization
    ket_initial /= np.sqrt(np.sum(abs2(ket_initial)))

    return ket_initial


@njit(cache=True)
def create_selected_eigen(ket_eigen, S_eigen):
    return ket_eigen[0, S_eigen, :]


@njit(cache=True)
def create_centered_Gaussian(sigma_Gaussian, N_lat, N_band):
    x_arr = np.arange(N_lat) - N_lat // 2
    x_arr = np.repeat(x_arr, N_band)

    ket_initial = np.exp(-0.5*(x_arr/sigma_Gaussian)**2)

    ## Normalization
    ket_initial /= np.sqrt(np.sum(abs2(ket_initial)))

    return ket_initial.astype(np.complex128)


def create_plane_constructive_ket(N_lat, N_band):
    x_arr = np.arange(N_lat) - N_lat//2
    k_arr = np.linspace(-pi, pi, N_lat)

    plane_waves = np.exp(1.j * k_arr[np.newaxis, :] * x_arr[:, np.newaxis])
    plane_waves = np.repeat(plane_waves, N_band, axis=-1)

    ket_initial = np.sum(plane_waves, axis=0)
    ket_initial = ket_initial / np.sqrt(np.sum(abs2(ket_initial)))

    return ket_initial


def create_Full_analytic_Wannier(epsilon, t_pm, t_0alpha, tn_alpha, gamma, N_state, N_band, N_k, withIdentity=True):
    ket_initial_CRM = exact_Wannier_pm(epsilon, t_pm, t_0alpha, tn_alpha, gamma, N_state, N_k, withIdentity)[0]

    # It is assumed that all higher bands (above s and p, e.g., d, f, g) are not
    # populated at the initial time.
    N_Ham = N_band * N_k  # N_k = N_lat
    if N_band > 2:
        ket_initial = np.zeros((N_Ham,), dtype=np.complex128)

        for i in range(N_k):  # N_k = N_lat
            for j in range(2):  # 2 -> [s, p]
                ket_initial[N_band*i+j] = ket_initial_CRM[2*i+j]
    else:
        ket_initial = ket_initial_CRM

    return ket_initial


def set_initial_condition(ket_eigen, initial_condition, sigma_Gaussian, N_lat, N_Ham, S_eigen, N_band):
    """Create initial condition based on the input string initial_condition."""
    if initial_condition == "Centered Single Sublattice":
        return create_centered_single_sublat(N_lat, N_Ham, N_band)
    elif initial_condition == "Centered Single Lattice":
        return create_centered_single_lat(N_lat, N_Ham, N_band)
    elif initial_condition == "Real Centered Eigen":
        return create_real_centered_eigen(ket_eigen, N_lat, N_Ham)
    elif initial_condition == "Centered Eigen":
        return create_centered_eigen(ket_eigen, N_lat, N_Ham, N_band)
    elif initial_condition == "Selected Eigen":
        return create_selected_eigen(ket_eigen, S_eigen)
    elif initial_condition == "Centered Gaussian":
        return create_centered_Gaussian(sigma_Gaussian, N_lat, N_band)
    elif initial_condition == "Plane Constructive":
        return create_plane_constructive_ket(N_lat, N_band)
    else:
        raise ValueError(r"Invalid initial_condition (from %s)" % set_initial_condition.__name__)
# --------------- END OF INITIAL CONDITION FUNCTIONS ---------------


# --------------- SPECIFIC CALCULATIONS ---------------
@njit(cache=True)
def modulate_Full_parameters(t, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, alpha_t):
    if modulate_parameters[0]:
        # delta_pm_t = delta_pm_bar[1] + np.array([-1, 1]) * delta_pm_bar[0] * np.sin(2*pi*t/T_pump)
        delta_pm_t = np.array([-1, 1]) * delta_pm_bar[0] + delta_pm_bar[1] * np.sin(2*pi*t/T_pump)
    else:
        delta_pm_t = np.array([delta_p[-1], delta_p[1]])
    if modulate_parameters[1]:
        omega_t = omega_bar[0] + alpha_t * omega_bar[1] * np.cos(2*pi*t/T_pump)
    else:
        omega_t = omega
    if modulate_parameters[2]:
        delta_0_t = delta_0_bar[0] + delta_0_bar[1] * np.cos(2*pi*t/T_pump)
    else:
        delta_0_t = delta_p[0]

    return delta_pm_t, omega_t, delta_0_t


@njit(cache=True)
def calc_alpha_t(t, tau_adiabatic):
    alpha_t = np.tanh(t / tau_adiabatic)
    # if t < tau_adiabatic:
    #     alpha_t = (t / tau_adiabatic)**1
    # else:
    #     alpha_t = 1

    return alpha_t


@njit(cache=True)
def calc_F(t, omega_t, delta_p_t, gamma, alpha_t):
    """F(t) is a function appearing in the calculation of the Hamiltonian matrix
    elements (in the Wannier basis). See notes for details."""
    F = 0.
    for p in range(-1, 2):
        F += alpha_t * delta_p_t[p] * np.exp(1j*(p*omega_t*t+gamma[p]))

    return F


@njit(cache=True)
def calc_Full_Hamiltonian(t, W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_Ham, sep_len):
    if adiabaticLaunching:
        alpha_t = calc_alpha_t(t, tau_adiabatic)
    else:
        alpha_t = 1.

    # PARAMETER MODULATION
    # parameter_t is the value of the parameter at time t.
    delta_pm_t, omega_t, delta_0_t = modulate_Full_parameters(t, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, alpha_t)
    # END OF PARAMETER MODULATION

    # SET DETUNING VALUES
    delta_p_t = np.empty(3, dtype=np.float64)
    delta_p_t[-1] = delta_pm_t[0]
    delta_p_t[0] = delta_0_t
    delta_p_t[1] = delta_pm_t[1]
    # END OF SET DETUNING VALUES

    Ham = np.zeros((N_Ham, N_Ham), dtype=np.complex128)

    # *************** H ASSIGNMENT *******************
    # ---------- H_0 ASSIGNMENT ----------
    # On-site (no spatial shift) terms.
    for j in range(N_band):  # N_band -> [s, p, d, ...]
        for i in range(N_lat):  # N_lat -> [..., r, r + 1, ...]
            Ham[N_band*i+j, N_band*i+j] += W_H_W_arr[j, j, 0] - j*omega_t

    # Natural tunneling terms.
    for k in range(1, sep_len):  # sep_len -> [0, a_0, 2*a_0, ...]
        for j in range(N_band):  # N_band -> [s, p, d, ...]
            for i in range(N_lat - k*N_state):  # N_lat -> [..., r, r + 1, ...]
                Ham[N_band*i+j, N_band*(i+k*N_state)+j] += W_H_W_arr[j, j, k]
                Ham[N_band*(i+k*N_state)+j, N_band*i+j] += W_H_W_arr[j, j, k]
    # ---------- END OF H_0 ASSIGNMENT ----------

    # TODO: Not fully general, does not include N*l+1 terms.
    # ---------- U ASSIGNMENT -----------------
    F = calc_F(t, omega_t, delta_p_t, gamma, alpha_t)

    for a in range(N_band):  # N_band -> [s, p, d, ...]
        for b in range(N_band):  # N_band -> [s, p, d, ...]
            # U_coupling_term = F * np.exp(1j*(b-a)*omega_t*t) * W_W_overlap_arr[a, b, 1]
            U_coupling_term = F * np.exp(1j*(b-a)*omega_t*t) * W_W_overlap_arr[b, a, 1]
            # if a-b == -1 or a-b == 0 or a-b == 1:
            #     U_coupling_term = delta_p_t[a-b] * np.exp(1j*gamma[a-b])* W_W_overlap_arr[b, a, 1]
            U_coupling_conj_term = np.conj(U_coupling_term)
            for i in range(N_lat - 1):  # N_lat -> [..., r, r + 1, ...]
                Ham[N_band*i+a, N_band*(i+1)+b] += U_coupling_term
                Ham[N_band*(i+1)+b, N_band*i+a] += U_coupling_conj_term
    # ---------- END OF U ASSIGNMENT ----------
    # *************** END OF H ASSIGNMENT ************

    return Ham


def diag_Full_time_spectrum(W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, N_state, N_band, N_lat, N_period, N_Ham, sep_len, period_arr):
    # Looking at long-term behaviour, where the way the driving was started
    # (instantly or adiabatically) does not matter. Therefore,
    # adiabaticLaunching is set to False and tau_adiabatic is irrelevant.
    calc_Hamiltonian = functools.partial(calc_Full_Hamiltonian, W_H_W_arr=W_H_W_arr, W_W_overlap_arr=W_W_overlap_arr, omega=omega, delta_p=delta_p, gamma=gamma, delta_pm_bar=delta_pm_bar, omega_bar=omega_bar, delta_0_bar=delta_0_bar, T_pump=T_pump, modulate_parameters=modulate_parameters, adiabaticLaunching=False, tau_adiabatic=0, N_state=N_state, N_band=N_band, N_lat=N_lat, N_Ham=N_Ham, sep_len=sep_len)

    E = np.empty((N_period, N_Ham), dtype=np.float64)
    ket_eigen = np.empty((N_period, N_Ham, N_Ham), dtype=np.complex128)
    for i in range(N_period):
        Ham = calc_Hamiltonian(period_arr[i])
        E[i, :], v = np.linalg.eigh(Ham)
        ket_eigen[i, :, :] = v.T

    return E, ket_eigen


def calc_Full_ket_t(ket_initial, W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_time, time_span, N_state, N_band, N_lat, N_Ham, sep_len, time_arr):
    calc_Hamiltonian = functools.partial(calc_Full_Hamiltonian, W_H_W_arr=W_H_W_arr, W_W_overlap_arr=W_W_overlap_arr, omega=omega, delta_p=delta_p, gamma=gamma, delta_pm_bar=delta_pm_bar, omega_bar=omega_bar, delta_0_bar=delta_0_bar, T_pump=T_pump, modulate_parameters=modulate_parameters, adiabaticLaunching=adiabaticLaunching, tau_adiabatic=tau_adiabatic, N_state=N_state, N_band=N_band, N_lat=N_lat, N_Ham=N_Ham, sep_len=sep_len)

    # (N_time, N_Ham)
    ket = solve_Schrodinger(calc_Hamiltonian, ket_initial, time_span, N_time, N_Ham, time_arr)

    return ket  # (N_time, N_Ham)


def calc_Full_all(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_x, x_max, max_sep, N_time, time_span, N_period, initial_condition, sigma_Gaussian, S_eigen, N_Ham, sep_len, period_arr, time_arr):
    # OVERLAPS AND HAMILTONIAN MATRIX ELEMENTS
    W_W_overlap_arr, W_H_W_arr = calc_1Omega_noHO_overlap_tunneling(Omega, max_sep, N_state, N_band)
    # END OF OVERLAPS AND HAMILTONIAN MATRIX ELEMENTS

    # TIME DIAGONALIZATION
    E, ket_eigen = diag_Full_time_spectrum(W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, N_state, N_band, N_lat, N_period, N_Ham, sep_len, period_arr)
    abs2_ket_eigen = abs2(ket_eigen)
    # END OF TIME DIAGONALIZATION

    ##### TIME EVOLUTION #####
    # INITIAL CONDITION
    # ket_initial = set_initial_condition(ket_eigen, initial_condition, sigma_Gaussian, N_lat, N_Ham, S_eigen, N_band)

    t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar = Full2CRM(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, N_state)
    t_pm_t0, epsilon_t0, t_0alpha_t0 = CRM.modulate_CRM_Gediminas(0, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, T_pump, modulate_parameters, alpha_t=1.)
    ket_initial = create_Full_analytic_Wannier(epsilon_t0, t_pm_t0, t_0alpha_t0, tn_alpha, gamma, N_state, N_band, N_lat, withIdentity=True)
    # END OF INITIAL CONDITION

    ket = calc_Full_ket_t(ket_initial, W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_time, time_span, N_state, N_band, N_lat, N_Ham, sep_len, time_arr)
    abs2_ket = abs2(ket)
    ##### END OF TIME EVOLUTION #####

    # ADDITIONAL CALCULATIONS
    # Instead of period_arr and N_period, use time_arr and N_time.
    E_t, ket_eigen_t = diag_Full_time_spectrum(W_H_W_arr, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, N_state, N_band, N_lat, N_time, N_Ham, sep_len, time_arr)
    band_pop = calc_band_pop(ket, ket_eigen_t, N_band, N_lat, N_time)
    ket_COM = calc_ket_COM(abs2_ket, N_state, N_band, N_lat)
    # END OF ADDITIONAL CALCULATIONS

    return E, abs2_ket_eigen, abs2_ket, ket_COM, band_pop, W_W_overlap_arr, W_H_W_arr
# --------------- END OF SPECIFIC CALCULATIONS ---------------


# @@@@@@@@@@@@@@@ FULL ROUTINES @@@@@@@@@@@@@@@
def full_Full_all(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_x, x_max, max_sep, N_time, time_span, N_period, initial_condition, sigma_Gaussian, period_selected, S_eigen, colors, params_selected):
    # DERIVED PARAMETERS
    N_Ham = N_band * N_lat

    sep_len = int(max_sep) + 1

    time_arr = np.linspace(*time_span, N_time)
    period_arr = np.linspace(0., T_pump, N_period)

    S_period = x2index((0., T_pump), N_period, period_selected)
    # END OF DERIVED PARAMETERS

    # FULL MODEL
    delta_pm_sum_period, omega_period, delta_0_period = calc_Full_parameters_period(omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, N_period, period_arr)
    E, abs2_ket_eigen, abs2_ket, ket_COM, band_pop, W_W_overlap_arr, W_H_W_arr = calc_Full_all(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_x, x_max, max_sep, N_time, time_span, N_period, initial_condition, sigma_Gaussian, S_eigen, N_Ham, sep_len, period_arr, time_arr)
    # END OF FULL MODEL

    # PLOTTING
    plotN_Full_all(E, abs2_ket_eigen, abs2_ket, ket_COM, band_pop, delta_pm_sum_period, omega_period, delta_0_period, params_selected, N_band, S_eigen, colors, N_Ham, S_period, time_arr, period_arr)
    # END OF PLOTTING


def compare_Full_to_CRM(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_x, x_max, max_sep, sigma_Gaussian, N_time, time_span, N_period, initial_condition, addFirstOrder, period_selected, time_selected, S_eigen, colors, params_selected):
    """For N_band=2, the top and bottom plots should be identical. For N_band > 2, the plots should be fairly close. May have unexpected behaviour when N_band \neq 2."""

    # DERIVED PARAMETERS
    N_Ham_Full = N_band * N_lat
    N_Ham_CRM = 2 * N_lat

    sep_len = int(max_sep) + 1

    time_arr = np.linspace(*time_span, N_time)
    period_arr = np.linspace(0., T_pump, N_period)

    S_period = x2index((0., T_pump), N_period, period_selected)
    S_time = x2index(time_span, N_time, time_selected)
    # END OF DERIVED PARAMETERS

    J_0 = 0.
    v_bar = 0.
    w_0 = 0.
    modulation_scheme = "Gediminas"

    # CRM MODEL (PART 1)
    # t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar = Full2CRM(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, N_state, printResult=True)
    t_pm, epsilon, t_0alpha, _, t_bar, epsilon_bar, t_0_bar = Full2CRM(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, N_state, printResult=True)
    tn_alpha = np.array([0., 0.])
    Chern_numbers = AP.calc_CRM_Chern_numbers(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, (-pi, pi), (-0.1, 0.9), modulate_parameters, 5000, N_time)
    print("Chern numbers:", Chern_numbers)

    Delta_points = create_Delta_points(epsilon, t_0alpha, params_selected)
    delta_pm_sum_period, omega_period, delta_0_period = calc_Full_parameters_period(omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, N_period, period_arr)
    t_pm_diff, epsilon_diff, t_0alpha_diff = CRM.calc_CRM_parameter_diffs(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, 0, 0, 0, T_pump, modulate_parameters, "Gediminas", N_period)
    x_gap_param_arr, x_gap_plot_arr, y_gap_param_arr, y_gap_plot_arr = CRM.calc_energy_gaps_plot_arrs(params_selected, t_bar, epsilon_bar, t_0_bar, Delta_points, 198, 200)
    # END OF CRM MODEL (PART 1)

    # FULL MODEL
    E_Full, abs2_ket_eigen_Full, abs2_ket_Full, ket_COM_Full, band_pop_Full, W_W_overlap_arr, W_H_W_arr = calc_Full_all(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_x, x_max, max_sep, N_time, time_span, N_period, initial_condition, sigma_Gaussian, S_eigen, N_Ham_Full, sep_len, period_arr, time_arr)
    print("*************** FULL MODEL DONE ***************")
    # END OF FULL MODEL

    # FULL MODEL (2 BANDS)
    # E_Full2, abs2_ket_eigen_Full2, abs2_ket_Full2, ket_COM_Full2, band_pop_Full2, W_W_overlap_arr, W_H_W_arr = calc_Full_all(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, 2, N_lat, N_x, x_max, max_sep, N_time, time_span, N_period, initial_condition, sigma_Gaussian, S_eigen, N_Ham_CRM, sep_len, period_arr, time_arr)
    # print("*************** FULL MODEL (2 BANDS) DONE ***************")
    # END OF FULL MODEL (2 BANDS)

    # CRM MODEL CALCULATIONS
    E_CRM, ket_eigen_CRM = CRM.diag_CRM_time_spectrum(W_W_overlap_arr, period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, N_state, N_lat, N_period, N_Ham_CRM, addFirstOrder, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar)
    abs2_ket_eigen_CRM = abs2(ket_eigen_CRM)
    # IPR_eigen_CRM = Full.calc_IPR(ket_eigen)

    abs2_ket_CRM, ket_COM_CRM, band_pop_CRM = CRM.calc_CRM_time_evolution(ket_eigen_CRM, initial_condition, sigma_Gaussian, S_eigen, time_arr, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_Ham_CRM, N_time, addFirstOrder, W_W_overlap_arr, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar)

    # energy_gaps_CRM = CRM.calc_energy_gaps(x_gap_param_arr, y_gap_param_arr, t_pm, epsilon, t_0alpha, tn_alpha, gamma, params_selected, N_state, 500, 198, 200,
    #                                        "Direct", withIdentity=True)

    print("*************** CRM MODEL DONE ***************")
    # END OF CRM MODEL CALCULATIONS

    # DEBUG
    print("ket_COM_CRM(t=0) & ket_COM_CRM(t=time_selected):\n",
          ket_COM_CRM[x2index(time_span, N_time, 0.)], "\n",
          ket_COM_CRM[S_time])
    # END OF DEBUG

    # SAVING ARRAYS
    save_array_NPY(abs2_ket_Full, "%s!%s" % ("abs2_ket_Full", datetime.now().strftime("%Y-%m-%d!%H.%M.%S")))
    # save_array_NPY(abs2_ket_Full2, "%s!%s" % ("abs2_ket_Full2", datetime.now().strftime("%Y-%m-%d!%H.%M.%S")))
    save_array_NPY(abs2_ket_CRM, "%s!%s" % ("abs2_ket_CRM", datetime.now().strftime("%Y-%m-%d!%H.%M.%S")))
    # END OF SAVING ARRAYS

    # save_parameter_TXT()

    # PLOTTING
    plotN_compare_Full2CRM(Chern_numbers[0], E_Full, E_CRM, abs2_ket_eigen_Full, abs2_ket_eigen_CRM, abs2_ket_Full, abs2_ket_CRM, ket_COM_Full, ket_COM_CRM, band_pop_Full, band_pop_CRM, delta_pm_sum_period, omega_period, delta_0_period, t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, time_arr, T_pump, period_arr, addFirstOrder, params_selected, colors, N_band, S_eigen, S_period, showPlot=True)
    # plot_ket_COM_Full_CRM(ket_COM_Full, ket_COM_CRM, time_arr, T_pump, colors)
    # plot_ket_COM_Full_Full2_CRM(ket_COM_Full, ket_COM_Full2, ket_COM_CRM, time_arr, T_pump, colors, showPlot=True)
    # END OF PLOTTING


def full_CRM_then_Full(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_x, x_max, max_sep, sigma_Gaussian, N_time, time_span, N_period, initial_condition, addFirstOrder, period_selected, time_selected, S_eigen, colors, params_selected):
    # DERIVED PARAMETERS
    N_Ham_Full = N_band * N_lat
    N_Ham_CRM = 2 * N_lat

    sep_len = int(max_sep) + 1

    time_arr = np.linspace(*time_span, N_time)
    period_arr = np.linspace(0., T_pump, N_period)

    S_period = x2index((0., T_pump), N_period, period_selected)
    S_time = x2index(time_span, N_time, time_selected)
    # END OF DERIVED PARAMETERS

    J_0 = 0.
    v_bar = 0.
    w_0 = 0.
    modulation_scheme = "Gediminas"

    # CRM MODEL CALCULATIONS
    # t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar = Full2CRM(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, N_state, printResult=True)
    t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar = Full2CRM(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, N_state, printResult=True)
    # tn_alpha = np.array([0., 0.])

    # if 100 * abs(tn_alpha[1]) >
    #     print("WARNING:")

    Chern_numbers = AP.calc_CRM_Chern_numbers(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, (-pi, pi), (-0.1, 0.9), modulate_parameters, 5000, N_time)
    print("Chern numbers:", Chern_numbers)

    W_W_overlap_arr, W_H_W_arr = calc_1Omega_noHO_overlap_tunneling(Omega, max_sep, N_state, N_band)

    Delta_points = create_Delta_points(epsilon, t_0alpha, params_selected)
    delta_pm_sum_period, omega_period, delta_0_period = calc_Full_parameters_period(omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, N_period, period_arr)
    t_pm_diff, epsilon_diff, t_0alpha_diff = CRM.calc_CRM_parameter_diffs(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, 0, 0, 0, T_pump, modulate_parameters, "Gediminas", N_period)
    x_gap_param_arr, x_gap_plot_arr, y_gap_param_arr, y_gap_plot_arr = CRM.calc_energy_gaps_plot_arrs(params_selected, t_bar, epsilon_bar, t_0_bar, Delta_points, 198, 200)

    E_CRM, ket_eigen_CRM = CRM.diag_CRM_time_spectrum(W_W_overlap_arr, period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, N_state, N_lat, N_period, N_Ham_CRM, addFirstOrder, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar)
    abs2_ket_eigen_CRM = abs2(ket_eigen_CRM)
    # IPR_eigen_CRM = Full.calc_IPR(ket_eigen)

    abs2_ket_CRM, ket_COM_CRM, band_pop_CRM = CRM.calc_CRM_time_evolution(ket_eigen_CRM, initial_condition, sigma_Gaussian, S_eigen, time_arr, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_Ham_CRM, N_time, addFirstOrder, W_W_overlap_arr, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar)

    x_Wannier = AP.full_exact_CRM_Zak_arr_t(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, T_pump, modulate_parameters, time_selected, colors, 250, N_period, plotResults=False) / (2*pi)

    # energy_gaps_CRM = CRM.calc_energy_gaps(x_gap_param_arr, y_gap_param_arr, t_pm, epsilon, t_0alpha, tn_alpha, gamma, params_selected, N_state, 500, 198, 200,
    #                                        "Direct", withIdentity=True)

    ############### ENERGY GAPS ###############
    # x_param_diff_span = np.array([-0.20, 0.20])
    # y_param_diff_span = np.array([-0.20, 0.20])

    # # t_avg should exceed some critical value. Otherwise, Zak phase will be zero everywhere (trivial case).
    # # epsilon_avg and t_0_avg should not affect the results since only the difference matters.
    # # x_param_avg = 0.20
    # x_param_avg = t_bar[0]
    # # y_param_avg = 0.20
    # y_param_avg = epsilon_bar[0] / 2

    # full_energy_gaps(x_param_diff_span, y_param_diff_span, x_param_avg, y_param_avg, t_pm, epsilon, t_0alpha, tn_alpha, gamma, params_selected, N_state, 1000, 200, 198, "Indirect")
    ############### END OF ENERGY GAPS ###############

    CRM.plotN_CRM_all(E_CRM, abs2_ket_eigen_CRM, abs2_ket_CRM, ket_COM_CRM, Chern_numbers[0], x_Wannier, band_pop_CRM, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, time_arr, T_pump, colors, Delta_points, S_period, S_eigen, params_selected, N_Ham_CRM,
                      withIPR=False, IPR_eigen=None,
                      withEnergyGaps=False, gap_string="Direct")

    print("*************** CRM MODEL DONE ***************")
    # END OF CRM MODEL CALCULATIONS

    # N_boundary = N_Ham_CRM//5  # Only used for boundary hit
    # if sum(abs2_ket_CRM[-1, -N_boundary:-1]) + sum(abs2_ket_CRM[-1, 0:N_boundary]) > 0.05:
    #     print("*************** BOUNDARY HIT ***************")

    x_displacement = ket_COM_CRM[-1] - ket_COM_CRM[0]
    print("x_displacement:", x_displacement)
    print("******************************")
    if -0.05 < x_displacement - Chern_numbers[0] < 0.05:
        # FULL MODEL
        E_Full, abs2_ket_eigen_Full, abs2_ket_Full, ket_COM_Full, band_pop_Full, W_W_overlap_arr, W_H_W_arr = calc_Full_all(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_x, x_max, max_sep, N_time, time_span, N_period, initial_condition, sigma_Gaussian, S_eigen, N_Ham_Full, sep_len, period_arr, time_arr)
        print("*************** FULL MODEL DONE ***************")
        # END OF FULL MODEL

        # FULL MODEL (2 BANDS)
        # E_Full2, abs2_ket_eigen_Full2, abs2_ket_Full2, ket_COM_Full2, band_pop_Full2, W_W_overlap_arr, W_H_W_arr = calc_Full_all(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, 2, N_lat, N_x, x_max, max_sep, N_time, time_span, N_period, initial_condition, sigma_Gaussian, S_eigen, N_Ham_CRM, sep_len, period_arr, time_arr)
        # print("*************** FULL MODEL (2 BANDS) DONE ***************")
        # END OF FULL MODEL (2 BANDS)

        # DEBUG
        print("ket_COM_CRM(t=0) & ket_COM_CRM(t=time_selected):\n",
              ket_COM_CRM[x2index(time_span, N_time, 0.)], "\n",
              ket_COM_CRM[S_time])
        # END OF DEBUG

        # SAVING ARRAYS
        save_array_NPY(abs2_ket_Full, "%s!%s" % ("abs2_ket_Full", datetime.now().strftime("%Y-%m-%d!%H.%M.%S")))
        # save_array_NPY(abs2_ket_Full2, "%s!%s" % ("abs2_ket_Full2", datetime.now().strftime("%Y-%m-%d!%H.%M.%S")))
        save_array_NPY(abs2_ket_CRM, "%s!%s" % ("abs2_ket_CRM", datetime.now().strftime("%Y-%m-%d!%H.%M.%S")))
        # END OF SAVING ARRAYS

        # save_parameter_TXT()

        # PLOTTING
        plotN_compare_Full2CRM(Chern_numbers[0], E_Full, E_CRM, abs2_ket_eigen_Full, abs2_ket_eigen_CRM, abs2_ket_Full, abs2_ket_CRM, ket_COM_Full, ket_COM_CRM, band_pop_Full, band_pop_CRM, delta_pm_sum_period, omega_period, delta_0_period, t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, time_arr, T_pump, period_arr, addFirstOrder, params_selected, colors, N_band, S_eigen, S_period, showPlot=False)
        # plot_ket_COM_Full_CRM(ket_COM_Full, ket_COM_CRM, time_arr, T_pump, colors)
        # plot_ket_COM_Full_Full2_CRM(ket_COM_Full, ket_COM_Full2, ket_COM_CRM, time_arr, T_pump, colors, showPlot=False)
        # END OF PLOTTING
    else:
        print("*************** FAILED PARAMETERS ***************")
# @@@@@@@@@@@@@@@ END OF FULL ROUTINES @@@@@@@@@@@@@@@


# --------------- PLOTTING ---------------
def calc_Full_parameters_period(omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, N_period, period_arr):
    delta_pm_period = np.empty((2, N_period), dtype=np.float64)
    omega_period = np.empty(N_period, dtype=np.float64)
    delta_0_period = np.empty(N_period, dtype=np.float64)

    for i, time in enumerate(period_arr):
        delta_pm_period[:, i], omega_period[i], delta_0_period[i] = modulate_Full_parameters(time, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, alpha_t=1)

    # delta_pm_diff_period = delta_pm_period[0, :] - delta_pm_period[1, :]  # (N_period,)
    delta_pm_sum_period = delta_pm_period[0, :] + delta_pm_period[1, :]  # (N_period,)

    return delta_pm_sum_period, omega_period, delta_0_period


def set_selected_axis_label(ax, param_selected, axis):
    """axis argument should be either "x" or "y"."""
    if axis == "x":
        plt_label = ax.set_xlabel
    if axis == "y":
        plt_label = ax.set_ylabel
    if param_selected == 0:
        # plt_label(r'$\delta^{(1)} - \delta^{(-1)}$ / $E_R$')
        plt_label(r'$\delta^{(1)} + \delta^{(-1)}$ / $E_R$')
    if param_selected == 1:
        plt_label(r"$\omega$ / $E_R \hbar^{-1}$")
    if param_selected == 2:
        plt_label(r"$\delta^{(0)}$ / $E_R$")


def set_selected_parameters_period(delta_pm_sum_period, omega_period, delta_0_period, param_selected):
    if param_selected == 0:
        param_diff_arr = delta_pm_sum_period
    if param_selected == 1:
        param_diff_arr = omega_period
    if param_selected == 2:
        param_diff_arr = delta_0_period

    return param_diff_arr


# SINGLE PLOTS
def plot_Full_modulation(delta_pm_sum_period, omega_period, delta_0_period, period_arr, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    T_pump = period_arr[-1]
    # ax.set_xlabel(r"t / $\tau$")
    ax.set_xlabel(r"t / $T$")
    # ax.set_ylabel(r"$x_0 - x_1$ / $E_R$")
    ax.set_ylabel(r"$x$ / $E_R$")
    # ax.plot(period_arr/T_pump, delta_pm_diff_period, color="black")
    ax.plot(period_arr/T_pump, delta_pm_sum_period, color="black")
    ax.plot(period_arr/T_pump, omega_period, color="red")
    ax.plot(period_arr/T_pump, delta_0_period, color="blue")
    # ax.legend([r'$\delta^{(1)} - \delta^{(-1)}$', r'$\omega$', r'$\delta^{(0)}$'])
    ax.legend([r'$\delta^{(1)} + \delta^{(-1)}$', r'$\omega$', r'$\delta^{(0)}$'])

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("Full_Modulation")
        plt.show()


def plot_Full_spectrum(E, S_eigen, S_time, period_arr, fig_ax=None, S_point=False):
    fig, ax = get_fig_ax(fig_ax)

    T_pump = period_arr[-1]
    # ax.set_xlabel(r"t / $\tau$")
    ax.set_xlabel(r"t / $T$")
    ax.set_ylabel(r"E / $E_R$")
    [ax.plot(period_arr/T_pump, E[:, i], color="black") for i in range(E.shape[1])]

    if S_point:
        ax.scatter(period_arr[S_time]/T_pump, E[S_time, S_eigen], s=30, color="magenta")

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("Full_Spectrum")
        plt.show()


def plot_Full_eigen(E, abs2_ket_eigen, N_band, S_eigen, colors, S_time, period_arr, fig_ax=None, file_name="Full_Eigen"):
    fig, ax = get_fig_ax(fig_ax)

    cell_arr = np.arange(abs2_ket_eigen.shape[-1]) / N_band

    ax.set_title(r"$t$=%.2f, $E$=%.2f" % (period_arr[S_time], E[S_time, S_eigen]))
    ax.set_xlabel(r"Cell index $r$")
    ax.set_ylabel(r"$|\psi|^2$")
    ax.axhline(0, color="black", ls="--")
    for i in range(N_band):
        ax.bar(cell_arr[i::N_band], abs2_ket_eigen[S_time, S_eigen, i::N_band], color=colors[i], width=1./N_band, align='edge')

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot(file_name)
        plt.show()


def plot_Full_parameter_path(delta_pm_sum_period, omega_period, delta_0_period, params_selected, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    x_param_sum_arr = set_selected_parameters_period(delta_pm_sum_period, omega_period, delta_0_period, params_selected[0])
    y_param_sum_arr = set_selected_parameters_period(delta_pm_sum_period, omega_period, delta_0_period, params_selected[1])
    set_selected_axis_label(ax, params_selected[0], 'x')
    set_selected_axis_label(ax, params_selected[1], 'y')
    ax.scatter(x_param_sum_arr, y_param_sum_arr, s=30, color="black")
    # ax.plot([Delta_points[:, 0]], [Delta_points[:, 1]], marker="*", markersize=10, markeredgecolor="red", markerfacecolor="red")

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("Full_Parameter_Path")
        plt.show()


def plot_ket_pcolormesh(abs2_ket, N_Ham, time_arr, T_pump, fig_ax=None, file_name="Ket_Pcolormesh"):
    fig, ax = get_fig_ax(fig_ax)

    coeff_plot_arr = np.arange(N_Ham, dtype=np.int64)
    # ax_pcolormesh = ax.pcolormesh(coeff_plot_arr, time_arr/T_pump, abs2_ket, cmap="viridis", shading="auto")
    ax_pcolormesh = ax.pcolormesh(coeff_plot_arr, time_arr/T_pump, abs2_ket, cmap="viridis", shading="auto", vmin=0.0, vmax=0.05)
    plt.colorbar(ax_pcolormesh, label=r"$|\langle W_R|\phi (t)\rangle|^2$", ax=ax)
    ax.set_xlabel(r"$|\langle W_R|\phi (t)\rangle|^2$")
    # ax.set_ylabel(r"t / $\tau$")
    ax.set_ylabel(r"t / $T$")

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot(file_name)
        plt.show()


def plot_ket_COM(ket_COM, time_arr, T_pump, fig_ax=None, file_name="Ket_COM"):
    fig, ax = get_fig_ax(fig_ax)

    ax.set_title(r'$x_{end} - x_{start}$ = %.2f' % (ket_COM[-1] - ket_COM[0]))
    # ax.set_xlabel(r"t / $\tau$")
    ax.set_xlabel(r"t / $T$")
    ax.set_ylabel(r"$\bar{x}$ / $a_0$")
    ax.plot(time_arr/T_pump, ket_COM, color="black")

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot(file_name)
        plt.show()


def plot_ket_COM_Full_CRM(ket_COM_Full, ket_COM_Full2, ket_COM_CRM, time_arr, T_pump, colors, fig_ax=None, file_name="Ket_COM_Full_CRM"):
    fig, ax = get_fig_ax(fig_ax)

    ax.set_title(r'$x_{end} - x_{start}$ = %.2f, %.2f' % (ket_COM_Full[-1] - ket_COM_Full[0], ket_COM_CRM[-1] - ket_COM_CRM[0]))
    # ax.set_xlabel(r"t / $\tau$")
    ax.set_xlabel(r"t / $T$")
    ax.set_ylabel(r"$\bar{x}$ / $a_0$")
    ax.plot(time_arr/T_pump, ket_COM_Full2, color=colors[0])
    ax.plot(time_arr/T_pump, ket_COM_CRM, color=colors[1])

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot(file_name)
        plt.show()


def plot_ket_COM_Full_Full2_CRM(ket_COM_Full, ket_COM_Full2, ket_COM_CRM, time_arr, T_pump, colors, fig_ax=None, file_name="Ket_COM_Full_CRM", showPlot=True):
    fig, ax = get_fig_ax(fig_ax)
    fig.set_size_inches(8, 5)

    ax.set_title(r'$x_{end} - x_{start}$ = %.2f, %.2f, %.2f' % (ket_COM_Full[-1] - ket_COM_Full[0], ket_COM_Full2[-1] - ket_COM_Full2[0], ket_COM_CRM[-1] - ket_COM_CRM[0]))
    # ax.set_xlabel(r"t / $\tau$")
    ax.set_xlabel(r"t / $T$")
    ax.set_ylabel(r"$\bar{x}$ / $a_0$")
    ax.plot(time_arr/T_pump, ket_COM_Full, color=colors[2])
    ax.plot(time_arr/T_pump, ket_COM_Full2, color=colors[0])
    ax.plot(time_arr/T_pump, ket_COM_CRM, color=colors[1])
    plt.legend([r'Full', r'Full2', r'CRM'], loc='upper left',
                prop=mpl.font_manager.FontProperties(size=10))

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot(file_name)
        if showPlot:
            plt.show()


def plot_band_pop(band_pop, time_arr, T_pump, colors, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    # ax.set_xlabel(r"t / $\tau$")
    ax.set_xlabel(r"t / $T$")
    ax.set_ylabel(r"Pop$_n$")
    [ax.plot(time_arr/T_pump, band_pop[i, :], color=colors[i]) for i in range(band_pop.shape[0])]

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("Band_Pop")
        plt.show()


def plot_ket2_snapshot(abs2_ket, time_arr, colors, S_time, N_lat, N_band, fig_ax=None, y_lim=None):
    fig, ax = get_fig_ax(fig_ax)

    # x_arr = np.arange(N_lat) - N_lat // 2
    # x_arr = np.repeat(x_arr, 2)
    cell_arr = np.arange(abs2_ket.shape[-1]) / N_band

    ax.set_title(r"$t$=%.2f" % time_arr[S_time])
    ax.set_xlabel(r"Cell index $r$")
    # ax.set_ylabel(r"$|\psi|^2$")
    ax.set_ylabel(r"$|\langle r, \alpha|\psi (t)\rangle|^2$")
    # ax.axhline(0, color="black", ls="--")
    # loc = matplotlib.ticker.LinearLocator(numticks=2)
    # loc.tick_values(vmin=x_arr[0], vmax=x_arr[-1])
    # ax.xaxis.set_major_locator(loc)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    for i in range(N_band):
        ax.bar(cell_arr[i::N_band], abs2_ket[S_time, i::N_band], color=colors[i], width=1./N_band, align='edge')

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("Ket2_Snapshot")
        plt.show()
# END OF SINGLE PLOTS


# COMPOSITE PLOTS
def plotN_spectrum_eigen(E, abs2_ket_eigen, period_arr, colors, N_band, S_period, S_eigen, file_name="Spectrum_Eigen"):
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    plot_Full_spectrum(E, S_eigen, S_period, period_arr, (fig, ax[0]))
    plot_Full_eigen(E, abs2_ket_eigen, N_band, S_eigen, colors, S_period, period_arr, (fig, ax[1]))

    set_plot_defaults(fig, ax)
    save_plot(file_name)
    plt.show()


def plotN_ket2_snapshots(abs2_ket, time_arr, colors, S_time, N_lat, N_band):
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    S_time0 = x2index((time_arr[0], time_arr[-1]), len(time_arr), 0.)
    y_lim = np.array([-0.00, 1.1])*abs2_ket[S_time0].max()

    plot_ket2_snapshot(abs2_ket, time_arr, colors, S_time0, N_lat, N_band, (fig, ax[0]), y_lim=y_lim)
    plot_ket2_snapshot(abs2_ket, time_arr, colors, S_time, N_lat, N_band, (fig, ax[1]), y_lim=y_lim)

    set_plot_defaults(fig, ax)
    save_plot("Ket2_Snapshots")
    plt.show()


def plotN_Full_all(E, abs2_ket_eigen, abs2_ket, ket_COM, band_pop, delta_pm_sum_period, omega_period, delta_0_period, params_selected, N_band, S_eigen, colors, N_Ham, S_time, time_arr, period_arr):
    # fig, ax = plt.subplots(1, 2)
    # fig.set_size_inches(12, 5)
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(18, 8)

    T_pump = period_arr[-1]
    plot_Full_modulation(delta_pm_sum_period, omega_period, delta_0_period, period_arr, (fig, ax[0, 0]))
    plot_Full_spectrum(E, S_eigen, S_time, period_arr, (fig, ax[0, 1]))
    plot_Full_eigen(E, abs2_ket_eigen, N_band, S_eigen, colors, S_time, period_arr, (fig, ax[0, 2]))
    plot_Full_parameter_path(delta_pm_sum_period, omega_period, delta_0_period, params_selected, (fig, ax[1, 0]))
    plot_ket_pcolormesh(abs2_ket, N_Ham, time_arr, T_pump, (fig, ax[1, 1]))
    plot_ket_COM(ket_COM, time_arr, T_pump, (fig, ax[1, 2]))

    set_plot_defaults(fig, ax)
    save_plot("Full_All")
    plt.show()


def plotN_compare_Full2CRM(Chern_number, E_Full, E_CRM, abs2_ket_eigen_Full, abs2_ket_eigen_CRM, abs2_ket_Full, abs2_ket_CRM, ket_COM_Full, ket_COM_CRM, band_pop_Full, band_pop_CRM, delta_pm_sum_period, omega_period, delta_0_period, t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, time_arr, T_pump, period_arr, addFirstOrder, params_selected, colors, N_band, S_eigen, S_time, showPlot=True):
    fig, ax = plt.subplots(2, 4)
    fig.set_size_inches(22, 8)
    if addFirstOrder:
        fig.suptitle(r"Top - Full, bottom - CRM (w/ first order Floquet correction). $C$=%i" % Chern_number)
    else:
        fig.suptitle(r"Top - Full, bottom - CRM (w/o first order Floquet correction). $C$=%i" % Chern_number)

    # plot_Full_parameter_path(delta_pm_sum_period, omega_period, delta_0_period, params_selected, (fig, ax[0, 0]))
    plot_ket_pcolormesh(abs2_ket_Full, abs2_ket_Full.shape[1], time_arr, T_pump, (fig, ax[0, 0]))
    plot_Full_spectrum(E_Full, S_eigen, S_time, period_arr, (fig, ax[0, 1]))
    # plot_Full_eigen(E_Full, abs2_ket_eigen_Full, N_band, S_eigen, colors, S_time, period_arr, (fig, ax[0, 2]))
    plot_band_pop(band_pop_Full, time_arr, T_pump, colors, (fig, ax[0, 2]))
    plot_ket_COM(ket_COM_Full, time_arr, T_pump, (fig, ax[0, 3]))

    CRM.plot_CRM_parameter_path(t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, params_selected, (fig, ax[1, 0]))
    CRM.plot_CRM_spectrum(E_CRM, period_arr, T_pump, S_time, S_eigen, (fig, ax[1, 1]))
    # CRM.plot_CRM_eigen(E_CRM, abs2_ket_eigen_CRM, period_arr, S_time, S_eigen, (fig, ax[1, 2]))
    plot_band_pop(band_pop_CRM, time_arr, T_pump, colors, (fig, ax[1, 2]))
    plot_ket_COM(ket_COM_CRM, time_arr, T_pump, (fig, ax[1, 3]))

    set_plot_defaults(fig, ax)
    save_plot("Full_CRM_Comparison")
    if showPlot:
        plt.show()


# def plotN_poster_Full_CRM(Chern_number, E_Full, E_CRM, abs2_ket_eigen_Full, abs2_ket_eigen_CRM, abs2_ket_Full, abs2_ket_CRM, ket_COM_Full, ket_COM_CRM, band_pop_Full, band_pop_CRM, delta_pm_sum_period, omega_period, delta_0_period, t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, time_arr, T_pump, period_arr, addFirstOrder, params_selected, colors, N_band, S_eigen, S_time):
#     fig, ax = plt.subplots(2, 3)
#     fig.set_size_inches(22, 8)
#     if addFirstOrder:
#         fig.suptitle(r"Top - Full, bottom - CRM (w/ first order Floquet correction). $C$=%i" % Chern_number)
#     else:
#         fig.suptitle(r"Top - Full, bottom - CRM (w/o first order Floquet correction). $C$=%i" % Chern_number)

#     # plot_Full_parameter_path(delta_pm_sum_period, omega_period, delta_0_period, params_selected, (fig, ax[0, 0]))
#     plot_ket_pcolormesh(abs2_ket_Full, abs2_ket_Full.shape[1], time_arr, T_pump, (fig, ax[0, 0]))
#     plot_Full_spectrum(E_Full, S_eigen, S_time, period_arr, (fig, ax[0, 1]))
#     # plot_Full_eigen(E_Full, abs2_ket_eigen_Full, N_band, S_eigen, colors, S_time, period_arr, (fig, ax[0, 2]))
#     plot_band_pop(band_pop_Full, time_arr, T_pump, colors, (fig, ax[0, 2]))
#     plot_ket_COM(ket_COM_Full, time_arr, T_pump, (fig, ax[0, 3]))

#     CRM.plot_CRM_parameter_path(t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, params_selected, (fig, ax[1, 0]))
#     CRM.plot_CRM_spectrum(E_CRM, period_arr, T_pump, S_time, S_eigen, (fig, ax[1, 1]))
#     # CRM.plot_CRM_eigen(E_CRM, abs2_ket_eigen_CRM, period_arr, S_time, S_eigen, (fig, ax[1, 2]))
#     plot_band_pop(band_pop_CRM, time_arr, T_pump, colors, (fig, ax[1, 2]))
#     plot_ket_COM(ket_COM_CRM, time_arr, T_pump, (fig, ax[1, 3]))

#     set_plot_defaults(fig, ax)
#     save_plot("Full_CRM_Comparison")
#     plt.show()
# END OF COMPOSITE PLOTS
# --------------- END OF PLOTTING ---------------
