from math import pi
import functools
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker

from QOPack.Math import *
from QOPack.Utility import *
from QOPack.RamanTimeDepDetuning.ZakPhase import *
# from QOPack.RamanTimeDepDetuning.AdiabaticPumping import calc_CRM_Chern_numbers, full_exact_CRM_Zak_arr_t, plot_x_Wannier
from QOPack.RamanTimeDepDetuning.AdiabaticPumping import *
from QOPack.RamanTimeDepDetuning.ReverseMapping import *
import QOPack.RamanTimeDepDetuning.TimeEvolutionFull as Full
import QOPack.RamanTimeDepDetuning.MomentumFull as Full_k

# plt.style.use("./matplotlibrc")
plt.style.use("default")

# UNUSED RICE-MELE FORMULATION:
# N_lat = 50  # Number of Rice-Mele cells.
# N_2lat -> [..., (C1, r), (C2, r), ...], where Ci is the i-th Rice-Mele chain and r is the lattice index.
# N_2lat = 2 * N_lat
# N_Ham = 2 * (2 * N_lat)  # 2 Rice-Mele chains and 2 sublattice indices


# --------------- MODULATION SCHEMES ---------------
@njit(cache=True)
def modulate_CRM_Gediminas(t, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, T_pump, modulate_parameters, alpha_t):
    if modulate_parameters[0]:
        t_pm_t = t_bar[0] + np.array([-1, 1]) * alpha_t * t_bar[1] * np.sin(2*pi*t/T_pump)
        # # t_pm_t[1] += 0.10

        # t_pm_t = np.array([t_bar[0] + 1.00 * np.cos(2*pi*t/T_pump), t_bar[1]])
    else:
        t_pm_t = t_pm
    if modulate_parameters[1]:
        epsilon_t = np.array([0, -(epsilon_bar[0] + alpha_t * epsilon_bar[1] * np.cos(2*pi*t/T_pump))])
        # # epsilon_t = np.array([epsilon_bar[0] + epsilon_bar[1] * np.cos(2*pi*t/T_pump), 0])
        # # epsilon_t = np.array([epsilon_bar[0]/2 + epsilon_bar[1]/2 * np.cos(2*pi*t/T_pump), -(epsilon_bar[0]/2 + epsilon_bar[1]/2 * np.cos(2*pi*t/T_pump))])

        # epsilon_t = np.array([epsilon_bar[0] + epsilon_bar[1] * np.sin(2*pi*t/T_pump), -(epsilon_bar[0] + epsilon_bar[1] * np.sin(2*pi*t/T_pump))])
    else:
        epsilon_t = epsilon
    if modulate_parameters[2]:
        # t_0alpha_t = t_0_bar[0] + np.array([-1, 1]) * t_0_bar[1] * np.cos(2*pi*t/T_pump)
        t_0alpha_t = np.array([t_0_bar[0] + alpha_t * t_0_bar[1] * np.cos(2*pi*t/T_pump), 0])
    else:
        t_0alpha_t = t_0alpha

    return t_pm_t, epsilon_t, t_0alpha_t


@njit(cache=True)
def modulate_CRM_Ian(t, t_0alpha, J_0, T_pump):
    """Adapted from Spielman2022.2202.05033.pdf"""
    epsilon_t = np.array([0., 0.])
    if np.remainder(t, T_pump) < 0.5*T_pump:
        t_pm_t = np.array([J_0, 0.])
        t_0alpha_t = t_0alpha
    else:
        t_pm_t = np.array([0., J_0])
        t_0alpha_t = t_0alpha

    return t_pm_t, epsilon_t, t_0alpha_t


@njit(cache=True)
def modulate_CRM_Book(t, t_0alpha, v_bar, w_0, T_pump):
    """Adapted from SFN.pdf, page 60."""
    epsilon_t = np.array([np.sin(2*pi*t/T_pump), -np.sin(2*pi*t/T_pump)])
    # epsilon_t = np.array([0, -2*np.sin(2*pi*t/T_pump)])
    # epsilon_t = np.array([2*np.sin(2*pi*t/T_pump), 0])
    t_pm_t = np.array([v_bar + np.cos(2*pi*t/T_pump), w_0])
    # t_pm_t = np.array([v_bar + 0.5*np.cos(2*pi*t/T_pump), w_0 - 0.5*np.cos(2*pi*t/T_pump)])
    t_0alpha_t = t_0alpha

    return t_pm_t, epsilon_t, t_0alpha_t


@njit(cache=True)
def set_modulation_CRM(t, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, alpha_t):
    if modulation_scheme == "Gediminas":
        return modulate_CRM_Gediminas(t, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, T_pump, modulate_parameters, alpha_t)  # [t_pm_t, epsilon_t, t_0alpha_t]
    elif modulation_scheme == "Ian":
        return modulate_CRM_Ian(t, t_0alpha, J_0, T_pump)
    elif modulation_scheme == "Book":
        return modulate_CRM_Book(t, t_0alpha, v_bar, w_0, T_pump)
    else:
        raise ValueError(r"Invalid modulation scheme.")
# --------------- END OF MODULATION SCHEMES ---------------


# --------------- SPECIFIC CALCULATIONS ---------------
def create_CRM_analytic_Wannier(epsilon, t_pm, t_0alpha, tn_alpha, gamma, N_state, N_k, withIdentity=True):
    ket_initial = exact_Wannier_pm(epsilon, t_pm, t_0alpha, tn_alpha, gamma, N_state, N_k, withIdentity)[0]

    return ket_initial


@njit(cache=True)
def calc_CRM_Hamiltonian(t, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_Ham):
    # if adiabaticLaunching:
    #     alpha_t = Full.calc_alpha_t(t, tau_adiabatic)
    # else:
    #     alpha_t = 1.
    alpha_t = 1.

    # PARAMETER MODULATION
    t_pm_t, epsilon_t, t_0alpha_t = set_modulation_CRM(t, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, alpha_t)
    # END OF PARAMETER MODULATION

    Ham = np.zeros((N_Ham, N_Ham), dtype=np.complex128)

    # *************** H ASSIGNMENT *******************
    # ---------- H_0 ASSIGNMENT ----------
    # epsilon ASSIGNMENT
    for i in range(N_lat):  # N_lat -> [..., r, r + 1, ...]
        for j in range(2):  # 2 -> [s, p]
            Ham[2*i+j, 2*i+j] += epsilon_t[j]
    # END OF epsilon ASSIGNMENT

    # tn_alpha ASSIGNMENT
    for i in range(N_lat - N_state):  # N_lat -> [..., r, r + 1, ...]
        for j in range(2):  # 2 -> [s, p]
            Ham[2*i+j, 2*(i+N_state)+j] += tn_alpha[j]
            Ham[2*(i+N_state)+j, 2*i+j] += tn_alpha[j]
    # END OF tn_alpha ASSIGNMENT
    # ---------- END OF H_0 ASSIGNMENT ----------

    # ---------- U ASSIGNMENT ----------
    # U_{-1} ASSIGNMENT
    T_m1_term = t_pm_t[0] * np.exp(1j * gamma[-1])
    T_m1_conj_term = np.conj(T_m1_term)
    for i in range(N_lat - 1):  # N_lat -> [..., r, r + 1, ...]
        Ham[2*i, 2*(i+1)+1] += T_m1_term
        Ham[2*(i+1)+1, 2*i] += T_m1_conj_term
    # END OF U_{-1} ASSIGNMENT

    # U_0 ASSIGNMENT
    for j in range(2):  # 2 -> [s, p]
        T_0alpha_term = t_0alpha_t[j] * np.exp(1j * gamma[0])
        T_0alpha_conj_term = np.conj(T_0alpha_term)
        for i in range(N_lat - 1):  # N_lat -> [..., r, r + 1, ...]
            Ham[2*i+j, 2*(i+1)+j] += T_0alpha_term
            Ham[2*(i+1)+j, 2*i+j] += T_0alpha_conj_term
    # END OF U_0 ASSIGNMENT

    # U_1 ASSIGNMENT
    T_p1_term = t_pm_t[1] * np.exp(1j * gamma[1])
    T_p1_conj_term = np.conj(T_p1_term)
    for i in range(N_lat - 1):  # N_lat -> [..., r, r + 1, ...]
        Ham[2*i+1, 2*(i+1)] += T_p1_term
        Ham[2*(i+1), 2*i+1] += T_p1_conj_term
    # END OF U_1 ASSIGNMENT
    # ---------- END OF U ASSIGNMENT ----------
    # *************** END OF H ASSIGNMENT **********

    # check_if_Hermitian(Ham, name="CRM_Ham")

    return Ham


@njit(cache=True)
def calc_CRM_k_Hamiltonian(t, k, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state):
    alpha_t = 1.

    # PARAMETER MODULATION
    t_pm_t, epsilon_t, t_0alpha_t = set_modulation_CRM(t, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, alpha_t)
    # END OF PARAMETER MODULATION

    # N_band = 2;
    Ham_k = np.zeros((2, 2), dtype=np.complex128)

    # *************** H ASSIGNMENT *******************
    # ---------- H_0 ASSIGNMENT -----------------
    # epsilon ASSIGNMENT
    for i in range(2):  # 2 -> [s, p]
        Ham_k[i, i] += epsilon_t[i]
    # END OF epsilon ASSIGNMENT

    # tn_alpha ASSIGNMENT
    for i in range(2):  # 2 -> [s, p]
        Ham_k[i, i] += 2*tn_alpha[i] * np.cos(N_state*k)
    # END OF tn_alpha ASSIGNMENT
    # ---------- END OF H_0 ASSIGNMENT ----------

    # ---------- U ASSIGNMENT -----------------
    # U_0 ASSIGNMENT
    for i in range(2):  # 2 -> [s, p]
        Ham_k[i, i] += 2*t_0alpha_t[i] * np.cos(k + gamma[0])
    # END OF U_0 ASSIGNMENT

    # OMEGA_1K ASSIGNMENT
    Omega_1k = t_pm[1]*np.exp(1j*(k + gamma[1])) + t_pm[0]*np.exp(-1j*(k + gamma[-1]))
    Ham_k[1, 0] += Omega_1k
    Ham_k[0, 1] += np.conj(Omega_1k)
    # END OF OMEGA_1K ASSIGNMENT
    # ---------- END OF U ASSIGNMENT ----------
    # *************** END OF H ASSIGNMENT **********

    return Ham_k


@njit(cache=True)
def calc_1st_order_Floquet_correction(t, W_W_overlap_arr, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_lat, N_Ham):
    # if adiabaticLaunching:
    #     alpha_t = Full.calc_alpha_t(t, tau_adiabatic)
    # else:
    #     alpha_t = 1.
    alpha_t = 1.

    # PARAMETER MODULATION
    delta_pm_t, omega_t, delta_0_t = Full.modulate_Full_parameters(t, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, alpha_t)
    # END OF PARAMETER MODULATION

    # SET DETUNING VALUES
    delta_p_t = np.empty(3, dtype=np.float64)
    delta_p_t[-1] = delta_pm_t[0]
    delta_p_t[0] = delta_0_t
    delta_p_t[1] = delta_pm_t[1]
    # END OF SET DETUNING VALUES

    Floquet_1 = np.zeros((N_Ham, N_Ham), dtype=np.complex128)

    # *************** FLOQUET_1 ASSIGNMENT *******************
    for i in range(N_lat):  # N_lat -> [..., r, r + 1, ...]
        # W_W_overlap_arr.shape = (N_band, N_band, sepN_len), for CRM - N_band = 2.
        # G_{s, p} = W_W_overlap_arr[1, 0, 1]

        # ---------- DIAGONAL ELEMENTS ----------
        # <r, s|H_{eff}|r, s>
        Floquet_1[2*i, 2*i] += W_W_overlap_arr[1, 0, 1]**2 / omega_t * (2*delta_p_t[0]**2 + 0.5*(delta_p_t[-1]**2 + delta_p_t[1]**2))
        # <r, p|H_{eff}|r, p>
        Floquet_1[2*i+1, 2*i+1] += -Floquet_1[2*i, 2*i]
        # ---------- END OF DIAGONAL ELEMENTS ----------

        # <r, s|H_{eff}|r, p>
        Floquet_1[2*i, 2*i+1] += delta_p_t[0] * W_W_overlap_arr[1, 0, 1] / omega_t * (W_W_overlap_arr[1, 1, 1] - W_W_overlap_arr[0, 0, 1]) * \
                                 (-delta_p_t[-1] * np.exp(1j*(gamma[-1]-gamma[0])) + delta_p_t[1] * np.exp(1j*(gamma[0]-gamma[1])))
        # <r, p|H_{eff}|r, s>
        Floquet_1[2*i+1, 2*i] += np.conj(Floquet_1[2*i, 2*i+1])

    for i in range(N_lat - 2):  # N_lat -> [..., r, r + 1, ...]
        # <r, s|H_{eff}|r+2, s>
        Floquet_1[2*i, 2*(i+2)] += -W_W_overlap_arr[1, 0, 1]**2 / omega_t * (delta_p_t[0]**2 * np.exp(2j*gamma[0]) + \
                                   0.5*(delta_p_t[-1] * delta_p_t[1] * np.exp(1j*(gamma[-1]+gamma[1]))))
        # <r+2, s|H_{eff}|r, s>
        Floquet_1[2*(i+2), 2*i] += np.conj(Floquet_1[2*i, 2*(i+2)])

        # <r, p|H_{eff}|r+2, p>
        Floquet_1[2*i+1, 2*(i+2)+1] += -Floquet_1[2*i, 2*(i+2)]
        # <r+2, p|H_{eff}|r, p>
        Floquet_1[2*(i+2)+1, 2*i+1] += np.conj(Floquet_1[2*i+1, 2*(i+2)+1])

        # <r, s|H_{eff}|r+2, p>
        Floquet_1[2*i, 2*(i+2)+1] += delta_p_t[-1] * delta_p_t[0] * W_W_overlap_arr[1, 0, 1] / omega_t * \
                                     (W_W_overlap_arr[1, 1, 1] - W_W_overlap_arr[0, 0, 1]) * np.exp(1j*(gamma[-1]+gamma[0]))
        # <r+2, p|H_{eff}|r, s>
        Floquet_1[2*(i+2)+1, 2*i] += np.conj(Floquet_1[2*i, 2*(i+2)+1])

        # <r, p|H_{eff}|r+2, s>
        Floquet_1[2*i+1, 2*(i+2)] += -delta_p_t[0] * delta_p_t[1] * W_W_overlap_arr[1, 0, 1] / omega_t * \
                                     (W_W_overlap_arr[1, 1, 1] - W_W_overlap_arr[0, 0, 1]) * np.exp(1j*(gamma[0]+gamma[1]))
        # <r+2, s|H_{eff}|r, p>
        Floquet_1[2*(i+2), 2*i+1] += np.conj(Floquet_1[2*i+1, 2*(i+2)])
    # *************** END OF FLOQUET_1 ASSIGNMENT *******************

    # check_if_Hermitian(Floquet_1, name="Floquet_1")

    return Floquet_1


def calc_CRM_ket_t(ket_initial, time_arr, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_Ham, N_time, addFirstOrder, W_W_overlap_arr, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar):
    calc_zero_order_Hamiltonian = functools.partial(calc_CRM_Hamiltonian, t_bar=t_bar, epsilon_bar=epsilon_bar, t_0_bar=t_0_bar, t_pm=t_pm, epsilon=epsilon, t_0alpha=t_0alpha, tn_alpha=tn_alpha, gamma=gamma, J_0=J_0, v_bar=v_bar, w_0=w_0, T_pump=T_pump, modulate_parameters=modulate_parameters, modulation_scheme=modulation_scheme, adiabaticLaunching=adiabaticLaunching, tau_adiabatic=tau_adiabatic, N_state=N_state, N_lat=N_lat, N_Ham=N_Ham)
    if addFirstOrder:
        calc_first_order_Floquet = functools.partial(calc_1st_order_Floquet_correction, W_W_overlap_arr=W_W_overlap_arr, omega=omega, delta_p=delta_p, gamma=gamma, delta_pm_bar=delta_pm_bar, omega_bar=omega_bar, delta_0_bar=delta_0_bar, J_0=J_0, v_bar=v_bar, w_0=w_0, T_pump=T_pump, modulate_parameters=modulate_parameters, modulation_scheme=modulation_scheme, adiabaticLaunching=adiabaticLaunching, tau_adiabatic=tau_adiabatic, N_lat=N_lat, N_Ham=N_Ham)

        calc_Hamiltonian = lambda t: calc_zero_order_Hamiltonian(t) + calc_first_order_Floquet(t)
    else:
        calc_Hamiltonian = calc_zero_order_Hamiltonian

    ket = Full.solve_Schrodinger(calc_Hamiltonian, ket_initial, time_span, N_time, N_Ham, time_arr)

    return ket


def diag_CRM_time_spectrum(W_W_overlap_arr, period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, N_state, N_lat, N_period, N_Ham, addFirstOrder, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar):
    # Looking at long-term behaviour, where the way the driving was started
    # (instantly or adiabatically) does not matter. Therefore,
    # adiabaticLaunching is set to False and tau_adiabatic is irrelevant.
    calc_Hamiltonian = functools.partial(calc_CRM_Hamiltonian, t_bar=t_bar, epsilon_bar=epsilon_bar, t_0_bar=t_0_bar, t_pm=t_pm, epsilon=epsilon, t_0alpha=t_0alpha, tn_alpha=tn_alpha, gamma=gamma, J_0=J_0, v_bar=v_bar, w_0=w_0, T_pump=T_pump, modulate_parameters=modulate_parameters, modulation_scheme=modulation_scheme, adiabaticLaunching=False, tau_adiabatic=0, N_state=N_state, N_lat=N_lat, N_Ham=N_Ham)
    if addFirstOrder:
        calc_first_order_Floquet = functools.partial(calc_1st_order_Floquet_correction, W_W_overlap_arr=W_W_overlap_arr, omega=omega, delta_p=delta_p, gamma=gamma, delta_pm_bar=delta_pm_bar, omega_bar=omega_bar, delta_0_bar=delta_0_bar, J_0=J_0, v_bar=v_bar, w_0=w_0, T_pump=T_pump, modulate_parameters=modulate_parameters, modulation_scheme=modulation_scheme, adiabaticLaunching=False, tau_adiabatic=0, N_lat=N_lat, N_Ham=N_Ham)

    E = np.empty((N_period, N_Ham), dtype=np.float64)
    ket_eigen = np.empty((N_period, N_Ham, N_Ham), dtype=np.complex128)
    for i in range(N_period):
        Ham = calc_Hamiltonian(period_arr[i])
        if addFirstOrder:
            Ham += calc_first_order_Floquet(period_arr[i])

        E[i, :], v = np.linalg.eigh(Ham)
        ket_eigen[i, :] = v.T

    return E, ket_eigen


def calc_CRM_time_evolution(ket_eigen, initial_condition, sigma_Gaussian, S_eigen, time_arr, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_Ham, N_time, addFirstOrder, W_W_overlap_arr, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar):
    # INITIAL CONDITION
    # ket_initial = Full.set_initial_condition(ket_eigen, initial_condition, sigma_Gaussian, N_lat, N_Ham, S_eigen, N_band=2)

    t_pm_t0, epsilon_t0, t_0alpha_t0 = set_modulation_CRM(0, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, alpha_t=1.)
    ket_initial = create_CRM_analytic_Wannier(epsilon_t0, t_pm_t0, t_0alpha_t0, tn_alpha, gamma, N_state, N_lat, withIdentity=True)
    # END OF INITIAL CONDITION

    # TIME EVOLUTION
    ket = calc_CRM_ket_t(ket_initial, time_arr, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_Ham, N_time, addFirstOrder, W_W_overlap_arr, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar)
    # END OF TIME EVOLUTION

    # CALCULATION OF WAVEFUNCTION PARAMETERS
    abs2_ket = abs2(ket)

    # Instead of period_arr and N_period, use time_arr and N_time.
    E_time, ket_eigen_time = diag_CRM_time_spectrum(W_W_overlap_arr, time_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, N_state, N_lat, N_time, N_Ham, addFirstOrder, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar)
    band_pop = Full.calc_band_pop(ket, ket_eigen_time, 2, N_lat, N_time)

    ket_COM = Full.calc_ket_COM(abs2_ket, N_state, 2, N_lat)
    # END OF CALCULATION OF WAVEFUNCTION PARAMETERS

    return abs2_ket, ket_COM, band_pop


def calc_Floquet_band_structure(t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_period):
    # DERIVED PARAMETERS
    k_arr = np.linspace(-pi, pi, N_lat)
    period_arr = np.linspace(0., T_pump, N_period)
    # END OF DERIVED PARAMETERS

    # RM MODEL
    calc_momentum_Hamiltonian = functools.partial(calc_CRM_k_Hamiltonian, t_bar=t_bar, epsilon_bar=epsilon_bar, t_0_bar=t_0_bar, t_pm=t_pm, epsilon=epsilon, t_0alpha=t_0alpha, tn_alpha=tn_alpha, gamma=gamma, J_0=J_0, v_bar=v_bar, w_0=w_0, T_pump=T_pump, modulate_parameters=modulate_parameters, modulation_scheme=modulation_scheme, adiabaticLaunching=adiabaticLaunching, tau_adiabatic=tau_adiabatic, N_state=N_state)

    # E_Floquet.shape = (N_lat, 2)
    # ket_Floquet.shape = (N_lat, 2, 2)
    E_Floquet, ket_Floquet = Full_k.diag_k_time_evolution_operator(calc_momentum_Hamiltonian, period_arr, k_arr, N_band=2)

    # u_k.shape = (N_period, N_lat, 2, 2)
    E_k, u_k = Full_k.diag_k_time_spectrum(calc_momentum_Hamiltonian, period_arr, k_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_period)

    # abs2_ket_Floquet = abs2(ket_Floquet)
    # TODO: magnetization probably incorrect.
    # magnetization = abs2_ket_Floquet[..., 0] - abs2_ket_Floquet[..., 1]
    # magnetization.shape = (N_lat, 2)

    # Floquet_abs2_overlaps = Full_k.calc_initial_time_Floquet_abs2_overlaps(u_k, ket_Floquet)
    Floquet_abs2_overlaps = Full_k.calc_period_averaged_Floquet_abs2_overlaps(u_k, ket_Floquet)
    magnetization = Full_k.calc_magnetization(Floquet_abs2_overlaps)
    # END OF RM MODEL

    return E_Floquet, magnetization
# --------------- END OF SPECIFIC CALCULATIONS ---------------


# @@@@@@@@@@@@@@@ FULL ROUTINES @@@@@@@@@@@@@@@
def full_CRM_time_diagonalization(W_W_overlap_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, params_selected, time_selected, S_eigen, N_state, N_lat, N_period, N_Ham, addFirstOrder, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar):
    # DERIVED PARAMETERS
    S_period = x2index((0., T_pump), N_period, time_selected)

    period_arr = np.linspace(0., T_pump, N_period)

    Delta_points = create_Delta_points(epsilon, t_0alpha, params_selected)
    t_pm_diff, epsilon_diff, t_0alpha_diff = calc_CRM_parameter_diffs(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, N_period)
    # END OF DERIVED PARAMETERS

    # TIME DIAGONALIZATION
    E, ket_eigen = diag_CRM_time_spectrum(W_W_overlap_arr, period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, N_state, N_lat, N_period, N_Ham, addFirstOrder, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar)
    IPR_eigen = Full.calc_IPR(ket_eigen)
    abs2_ket_eigen = abs2(ket_eigen)
    # END OF TIME DIAGONALIZATION

    # PLOTTING
    # plotN_CRM_edge_IPR(E, abs2_ket_eigen, IPR_eigen, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, T_pump, S_period, S_eigen, Delta_points, params_selected, N_lat)
    plotN_CRM_spectrum_eigen(E, abs2_ket_eigen, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, T_pump, Delta_points, S_period, S_eigen, params_selected)
    # END OF PLOTTING


def full_CRM_time_evolution(initial_condition, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, sigma_Gaussian, S_eigen, N_state, N_lat, N_period, N_time, time_selected, addFirstOrder, W_W_overlap_arr, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar):
    # DERIVED PARAMETERS
    N_Ham = 2 * N_lat  # 2 Bloch bands (s and p) and N_lat lattice indices.

    time_arr = np.linspace(*time_span, N_time)
    period_arr = np.linspace(0., T_pump, N_period)

    S_time = x2index(time_span, N_time, time_selected)
    # END OF DERIVED PARAMETERS

    E, ket_eigen = diag_CRM_time_spectrum(W_W_overlap_arr, period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, N_state, N_lat, N_period, N_Ham, addFirstOrder, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar)

    # TIME EVOLUTION
    abs2_ket, ket_COM, band_pop = calc_CRM_time_evolution(ket_eigen, initial_condition, sigma_Gaussian, S_eigen, time_arr, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_Ham, N_time, addFirstOrder, W_W_overlap_arr, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar)
    # END OF TIME EVOLUTION

    # PLOTTING
    Full.plotN_ket2_snapshots(abs2_ket, time_arr, ["red", "black"], S_time, N_lat, 2)
    Full.plot_ket_pcolormesh(abs2_ket, N_Ham, time_arr, T_pump)
    Full.plot_ket_COM(ket_COM, time_arr, T_pump)
    # plotN_CRM_ket(abs2_ket, ket_COM, time_arr, T_pump, N_Ham)
    # END OF PLOTTING


def full_CRM_all(W_W_overlap_arr, initial_condition, sigma_Gaussian, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, params_selected, colors, period_selected, time_selected, S_eigen, N_state, N_lat, N_period, N_time, N_Ham, addFirstOrder, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar):
    # DERIVED PARAMETERS
    S_period = x2index((0., T_pump), N_period, time_selected)
    S_time = x2index(time_span, N_time, time_selected)

    # N_lat_Floquet = 5*N_lat
    # k_arr = np.linspace(-pi, pi, N_lat_Floquet)

    time_arr = np.linspace(*time_span, N_time)
    period_arr = np.linspace(0., T_pump, N_period)

    t_pm_diff, epsilon_diff, t_0alpha_diff = calc_CRM_parameter_diffs(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, N_period)

    Delta_points = create_Delta_points(epsilon, t_0alpha, params_selected)

    # x_gap_param_arr, x_gap_plot_arr, y_gap_param_arr, y_gap_plot_arr = calc_energy_gaps_plot_arrs(params_selected, t_bar, epsilon_bar, t_0_bar, Delta_points, 198, 200)
    # END OF DERIVED PARAMETERS

    # CRM MODEL
    epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time = full_modulate(epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, (0., 1.), modulate_parameters, N_period)
    k_eigvec = full_exact_k_eigvec_kt(epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time, gamma, (-pi, pi), N_state, 300, N_period)  # (2, N_k, N_period, 2)
    Zak_arr_t = calc_general_Berry_phase(k_eigvec, prod_axis=1)  # (2, N_period)
    Zak_arr_t = smooth_Berry_phase(Zak_arr_t)
    print("Zak_t_diff:", Zak_arr_t[0, -1] - Zak_arr_t[0, 0])

    Chern_numbers = calc_CRM_Chern_numbers(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, (-pi, pi), (-0.1, 0.9), modulate_parameters, 500, N_time)
    E, ket_eigen = diag_CRM_time_spectrum(W_W_overlap_arr, period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, N_state, N_lat, N_period, N_Ham, addFirstOrder, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar)
    abs2_ket_eigen = abs2(ket_eigen)
    # IPR_eigen = Full.calc_IPR(ket_eigen)

    abs2_ket, ket_COM, band_pop = calc_CRM_time_evolution(ket_eigen, initial_condition, sigma_Gaussian, S_eigen, time_arr, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_Ham, N_time, addFirstOrder, W_W_overlap_arr, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar)

    # E_Floquet, magnetization = calc_Floquet_band_structure(t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state,
    #                                                        N_lat_Floquet, N_period)
    # N_lat_Floquet >> N_lat because Floquet quasienergies need more points.

    # energy_gaps = calc_energy_gaps(x_gap_param_arr, y_gap_param_arr, t_pm, epsilon, t_0alpha, tn_alpha, gamma, params_selected, N_state, 500, 198, 200,
    #                                "Direct", withIdentity=True)

    # x_Wannier = full_exact_CRM_Zak_arr_t(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, T_pump, modulate_parameters, time_selected, colors, 250, N_period, plotResults=False) / (2*pi)
    x_Wannier = full_exact_CRM_Zak_arr_t(Chern_numbers[0], N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, T_pump, modulate_parameters, params_selected, time_selected, colors, 250, N_period, plotResults=True) / (2*pi)
    # END OF CRM MODEL

    # DEBUG
    print("ket_COM(t=0) & ket_COM(t=time_selected):\n",
          ket_COM[x2index(time_span, N_time, 0.)], "\n",
          ket_COM[S_time])
    # END OF DEBUG

    # PLOTTING
    # plot_CRM_ket2_snapshot(abs2_ket, time_arr, S_time, N_lat)
    # Full.plotN_ket2_snapshots(abs2_ket, time_arr, colors, S_time, N_lat, 2)
    plotN_CRM_all(E, abs2_ket_eigen, abs2_ket, ket_COM, Chern_numbers[0], band_pop, x_Wannier, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, time_arr, T_pump, colors, Delta_points, S_period, S_eigen, params_selected, N_Ham,
                  withIPR=False, IPR_eigen=None,
                  withEnergyGaps=False, gap_string="Direct", energy_gaps=None, x_gap_plot_arr=None, y_gap_plot_arr=None)
    # plotN_CRM_poster(E, abs2_ket_eigen, abs2_ket, ket_COM, Chern_numbers[0], band_pop, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, time_arr, T_pump, colors, Delta_points, S_period, S_eigen, params_selected, N_Ham, withIPR=False, IPR_eigen=None, withEnergyGaps=False, gap_string="Direct", energy_gaps=None, x_gap_plot_arr=None, y_gap_plot_arr=None)
    # plotN_CRM_all2(E_Floquet, E, abs2_ket_eigen, abs2_ket, ket_COM, k_arr, Chern_numbers[0], band_pop, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, time_arr, T_pump, colors, Delta_points, S_period, S_eigen, params_selected, N_Ham, magnetization, withIPR=False, IPR_eigen=None, withEnergyGaps=False, gap_string="Direct", energy_gaps=None, x_gap_plot_arr=None, y_gap_plot_arr=None)
    # END OF PLOTTING


def Floquet_band_structure(t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_period):
    # DERIVED PARAMETERS
    k_arr = np.linspace(-pi, pi, N_lat)
    period_arr = np.linspace(0., T_pump, N_period)
    # END OF DERIVED PARAMETERS

    E_Floquet, magnetization = calc_Floquet_band_structure(t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, adiabaticLaunching, tau_adiabatic, N_state, N_lat, N_period)

    # PLOTTING
    Full_k.plot_k_Floquet_Quasienergy(E_Floquet, k_arr, magnetization, file_name="CRM_k_Floquet_Quasienergy")
    # Full_k.plot_k_Floquet_Quasienergy(E_Floquet, k_arr, file_name="CRM_k_Floquet_Quasienergy")
    # END OF PLOTTING
# @@@@@@@@@@@@@@@ END OF FULL ROUTINES @@@@@@@@@@@@@@@


# ----------------- PLOTTING -----------------
def calc_CRM_parameter_diffs(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, N_period):
    t_pm_period = np.empty((2, N_period), dtype=np.float64)
    epsilon_period = np.empty_like(t_pm_period, dtype=np.float64)
    t_0alpha_period = np.empty_like(t_pm_period, dtype=np.float64)
    for i, time in enumerate(period_arr):
        # t_pm_period[:, i], epsilon_period[:, i], t_0alpha_period[:, i] = modulate_CRM_Gediminas(time, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, T_pump, modulate_parameters, alpha_t=1)
        t_pm_period[:, i], epsilon_period[:, i], t_0alpha_period[:, i] = set_modulation_CRM(time, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, J_0, v_bar, w_0, T_pump, modulate_parameters, modulation_scheme, alpha_t=1)

    t_pm_diff = t_pm_period[1, :] - t_pm_period[0, :]
    epsilon_diff = epsilon_period[0, :] - epsilon_period[1, :]
    t_0alpha_diff = t_0alpha_period[0, :] - t_0alpha_period[1, :]

    return t_pm_diff, epsilon_diff, t_0alpha_diff


def calc_energy_gaps_plot_arrs(params_selected, t_bar, epsilon_bar, t_0_bar, Delta_points, N_param1, N_param2):
    params_amplitude = np.empty(2, dtype=np.float64)
    params_avg = np.empty(2, dtype=np.float64)
    for i in range(2):
        if params_selected[i] == 0:
            # TODO: Not the best amplitude selection, but satisfactory.
            params_amplitude[i] = 2 * np.amax([np.abs(t_bar[1]), *Delta_points[:, i]])
            params_avg[i] = t_bar[0]
        if params_selected[i] == 1:
            params_amplitude[i] = 2 * np.amax([np.abs(epsilon_bar[1]), *Delta_points[:, i]])
            params_avg[i] = epsilon_bar[0]
        if params_selected[i] == 2:
            params_amplitude[i] = 2 * np.amax([np.abs(t_0_bar[1]), *Delta_points[:, i]])
            params_avg[i] = t_0_bar[0]

    x_gap_param_diff_span = 1.1 * params_amplitude[0] * np.array([-1, 1])
    y_gap_param_diff_span = 1.1 * params_amplitude[1] * np.array([-1, 1])

    x_gap_param_arr, x_gap_plot_arr = create_param_arrs(x_gap_param_diff_span, params_avg[0], N_param1)
    y_gap_param_arr, y_gap_plot_arr = create_param_arrs(y_gap_param_diff_span, params_avg[1], N_param2)

    return x_gap_param_arr, x_gap_plot_arr, y_gap_param_arr, y_gap_plot_arr


def set_selected_axis_label(ax, param_selected, axis):
    """axis argument should be either "x" or "y"."""
    if axis == "x":
        plt_label = ax.set_xlabel
    if axis == "y":
        plt_label = ax.set_ylabel
    if param_selected == 0:
        plt_label(r"$t_1 - t_{-1}$ / $E_R$")
    if param_selected == 1:
        plt_label(r"$\bar{ϵ}^{(s)} - \bar{ϵ}^{(p)}$ / $E_R$")
    if param_selected == 2:
        plt_label(r"$t_{0s} - t_{0p}$ / $E_R$")


def set_selected_param_diff_arr(t_pm_diff, epsilon_diff, t_0alpha_diff, param_selected):
    if param_selected == 0:
        param_diff_arr = t_pm_diff
    if param_selected == 1:
        param_diff_arr = epsilon_diff
    if param_selected == 2:
        param_diff_arr = t_0alpha_diff

    return param_diff_arr


# SINGLE PLOTS
def plot_CRM_modulation(t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, T_pump, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    # ax.set_xlabel(r"t / $\tau$")
    ax.set_xlabel(r"t / $T$")
    ax.set_ylabel(r"$x_0 - x_1$ / $E_R$")
    ax.plot(period_arr/T_pump, t_pm_diff, color="black")
    ax.plot(period_arr/T_pump, epsilon_diff, color="red")
    ax.plot(period_arr/T_pump, t_0alpha_diff, color="blue")
    ax.legend([r'$t_1 - t_{-1}$', r'$\bar{ϵ}^{(s)} - \bar{ϵ}^{(p)}$', r'$t_{0s} - t_{0p}$'])

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("CRM_Modulation")
        plt.show()


def plot_CRM_spectrum(E, period_arr, T_pump, S_period, S_eigen, fig_ax=None, S_point=False, withIPR=False, IPR_eigen=None):
    """If withIPR is set to True, IPR_eigen must be provided."""
    fig, ax = get_fig_ax(fig_ax)

    # ax.set_xlabel(r"t / $\tau$")
    ax.set_xlabel(r"t / $T$")
    ax.set_ylabel(r"E / $E_R$")

    if withIPR:
        # DEBUG
        # print(IPR_eigen[30, 100])
        # print(IPR_eigen.min())
        # print(IPR_eigen.max())
        # END OF DEBUG

        # WITH IPR COLORS
        [ax.scatter(period_arr/T_pump, E[:, i], c=IPR_eigen[:, i], cmap="jet", s=10) for i in range(1, E.shape[1])]
        scatter = ax.scatter(period_arr/T_pump, E[:, 0], c=IPR_eigen[:, 0], cmap="jet", s=10, norm=mpl.colors.Normalize(vmin=IPR_eigen.min(), vmax=IPR_eigen.max()))
        plt.colorbar(scatter, label="$IPR$", ax=ax)
        # END OF WITH IPR COLORS
    else:
        # ALL BLACK (WITHOUT IPR COLORS)
        [ax.plot(period_arr/T_pump, E[:, i], color="black") for i in range(E.shape[1])]
        if S_point:
            ax.scatter(period_arr[S_period]/T_pump, E[S_period, S_eigen], s=30, color="magenta")
        # END OF ALL BLACK (WITHOUT IPR COLORS)

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("CRM_Spectrum")
        plt.show()


def plot_CRM_eigen(E, abs2_ket_eigen, period_arr, S_period, S_eigen, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    cell_arr = np.arange(abs2_ket_eigen.shape[-1]) / 2

    ax.set_title(r"$t$=%.2f, $E$=%.2f" % (period_arr[S_period], E[S_period, S_eigen]))
    ax.set_xlabel(r"Cell index $r$")
    ax.set_ylabel(r"$|\psi|^2$")
    # ax.axhline(0, color="black", ls="--")
    # ax.plot(ket_eigen[S_period, S_eigen], color="red")
    ax.bar(cell_arr[::2], abs2_ket_eigen[S_period, S_eigen, ::2], color="red", width=0.5, align='edge')
    ax.bar(cell_arr[1::2], abs2_ket_eigen[S_period, S_eigen, 1::2], color="black", width=0.5, align='edge')

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("CRM_Eigen")
        plt.show()


def plot_CRM_parameter_path(t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, params_selected, fig_ax=None, withEnergyGaps=False, gap_string="Direct", energy_gaps=None, x_gap_plot_arr=None, y_gap_plot_arr=None):
    fig, ax = get_fig_ax(fig_ax)

    x_param_diff_arr = set_selected_param_diff_arr(t_pm_diff, epsilon_diff, t_0alpha_diff, params_selected[0])
    y_param_diff_arr = set_selected_param_diff_arr(t_pm_diff, epsilon_diff, t_0alpha_diff, params_selected[1])

    if withEnergyGaps:
        # scatter is only used to configure colorbar (to place it near the correct plot)
        scatter = ax.scatter(x_gap_plot_arr[0], y_gap_plot_arr[0], color="white")

        if energy_gaps.min() > 0.:  # If energy gaps are negative, logarithmic scale cannot be used.
            ax.pcolormesh(x_gap_plot_arr, y_gap_plot_arr, np.transpose(energy_gaps), cmap=plt.get_cmap("viridis"), shading="nearest",
                          norm=mpl.colors.LogNorm(vmin=energy_gaps.min(), vmax=energy_gaps.max()))
        else:
            ax.pcolormesh(x_gap_plot_arr, y_gap_plot_arr, np.transpose(energy_gaps), cmap=plt.get_cmap("viridis"), shading="nearest")

        if gap_string.lower() == "direct":
            plt.colorbar(scatter, label=r"Direct band gap $\mathrm{min}(E_{i+1} - E_i)$", aspect=30, ax=ax)
        if gap_string.lower() == "indirect":
            plt.colorbar(scatter, label=r"Indirect band gap $\mathrm{min}(E_{i+1}) - \mathrm{max}(E_i)$", aspect=30, ax=ax)

    set_selected_axis_label(ax, params_selected[0], "x")
    set_selected_axis_label(ax, params_selected[1], "y")
    ax.scatter(x_param_diff_arr, y_param_diff_arr, s=30, color="black")
    ax.plot([Delta_points[:, 0]], [Delta_points[:, 1]], marker="*", markersize=10, markeredgecolor="red", markerfacecolor="red")

    if fig_ax is None:
        if withEnergyGaps:
            set_plot_defaults(fig, ax, addGrid=False)
        else:
            set_plot_defaults(fig, ax)
        save_plot("CRM_Parameter_Path")
        plt.show()
    else:
        if withEnergyGaps:
            ax.grid()
# END OF SINGLE PLOTS


# COMPOSITE PLOTS
def plotN_CRM_ket(abs2_ket, ket_COM, time_arr, T_pump, N_Ham):
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    Full.plot_ket_pcolormesh(abs2_ket, N_Ham, time_arr, T_pump, (fig, ax[0]))
    Full.plot_ket_COM(ket_COM, time_arr, T_pump, (fig, ax[1]))

    set_plot_defaults(fig, ax)
    save_plot("CRM_ket")
    plt.show()


def plotN_CRM_spectrum_eigen(E, abs2_ket_eigen, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, T_pump, Delta_points, S_period, S_eigen, params_selected):
    # fig, ax = plt.subplots(1, 2)
    # fig.set_size_inches(12, 5)
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)

    plot_CRM_spectrum(E, period_arr, T_pump, S_period, S_eigen, (fig, ax[0, 0]), S_point=True)
    plot_CRM_eigen(E, abs2_ket_eigen, period_arr, S_period, S_eigen, (fig, ax[0, 1]))
    # plot_CRM_parameter_path(gap_arr, t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, params_selected, (fig, ax[1, 0]))
    plot_CRM_modulation(t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, T_pump, (fig, ax[1, 1]))

    set_plot_defaults(fig, ax)
    save_plot("CRM_Spectrum_Eigen")
    plt.show()


def plotN_CRM_edge_IPR(E, abs2_ket_eigen, IPR_eigen, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, T_pump, S_period, S_eigen, Delta_points, params_selected, N_lat):
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(18, 8)

    # DEBUG
    # S_period1 = x2index((period_arr[0], period_arr[-1]), len(period_arr), 0.49*period_arr[-1])
    # plot_CRM_eigen(E, abs2_ket_eigen, period_arr, S_period1, N_lat + 5, (fig, ax[0, 1]))
    # plot_CRM_eigen(E, abs2_ket_eigen, period_arr, S_period1, N_lat + 10, (fig, ax[0, 2]))

    plot_CRM_eigen(E, abs2_ket_eigen, period_arr, 0, -2, (fig, ax[0, 1]))
    plot_CRM_eigen(E, abs2_ket_eigen, period_arr, -1, -2, (fig, ax[0, 2]))
    # END OF DEBUG

    plot_CRM_spectrum(E, period_arr, T_pump, S_period, S_eigen, (fig, ax[0, 0]), withIPR=True, IPR_eigen=IPR_eigen)
    # plot_CRM_eigen(E, abs2_ket_eigen, period_arr, S_period, 0, (fig, ax[0, 1]))
    # plot_CRM_eigen(E, abs2_ket_eigen, period_arr, S_period, -1, (fig, ax[0, 2]))
    plot_CRM_parameter_path(t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, params_selected, (fig, ax[1, 0]))
    plot_CRM_eigen(E, abs2_ket_eigen, period_arr, S_period, N_lat - 1, (fig, ax[1, 1]))
    plot_CRM_eigen(E, abs2_ket_eigen, period_arr, S_period, N_lat, (fig, ax[1, 2]))

    set_plot_defaults(fig, ax)
    save_plot("CRM_Edge_IPR")
    plt.show()


def plotN_CRM_all(E, abs2_ket_eigen, abs2_ket, ket_COM, Chern_number, band_pop, x_Wannier, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, time_arr, T_pump, colors, Delta_points, S_period, S_eigen, params_selected, N_Ham, withIPR=False, IPR_eigen=None, withEnergyGaps=False, gap_string="Direct", energy_gaps=None, x_gap_plot_arr=None, y_gap_plot_arr=None):
    # fig, ax = plt.subplots(1, 2)
    # fig.set_size_inches(12, 5)
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(18, 8)
    fig.suptitle(r'$C$=%i' % Chern_number)

    # plot_CRM_modulation(t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, T_pump, (fig, ax[0, 0]))
    plot_x_Wannier(x_Wannier, period_arr, colors, (fig, ax[0, 0]))
    plot_CRM_spectrum(E, period_arr, T_pump, S_period, S_eigen, (fig, ax[0, 1]), withIPR=withIPR, IPR_eigen=IPR_eigen)
    # plot_CRM_eigen(E, abs2_ket_eigen, period_arr, S_period, S_eigen, (fig, ax[0, 2]))
    Full.plot_band_pop(band_pop, time_arr, T_pump, colors, (fig, ax[0, 2]))
    plot_CRM_parameter_path(t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, params_selected, (fig, ax[1, 0]), withEnergyGaps=withEnergyGaps, gap_string=gap_string, energy_gaps=energy_gaps, x_gap_plot_arr=x_gap_plot_arr, y_gap_plot_arr=y_gap_plot_arr)
    Full.plot_ket_pcolormesh(abs2_ket, N_Ham, time_arr, T_pump, (fig, ax[1, 1]))
    Full.plot_ket_COM(ket_COM, time_arr, T_pump, (fig, ax[1, 2]))

    set_plot_defaults(fig, ax)
    save_plot("CRM_All")
    plt.show()


def plotN_CRM_all2(E_Floquet, E, abs2_ket_eigen, abs2_ket, ket_COM, k_arr, Chern_number, band_pop, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, time_arr, T_pump, colors, Delta_points, S_period, S_eigen, params_selected, N_Ham, magnetization=None, withIPR=False, IPR_eigen=None, withEnergyGaps=False, gap_string="Direct", energy_gaps=None, x_gap_plot_arr=None, y_gap_plot_arr=None):
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(18, 8)
    fig.suptitle(r'$C$=%i' % Chern_number)

    plot_CRM_modulation(t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, T_pump, (fig, ax[0, 0]))
    Full.plot_ket_pcolormesh(abs2_ket, N_Ham, time_arr, T_pump, (fig, ax[0, 1]))
    Full_k.plot_k_Floquet_Quasienergy(E_Floquet, k_arr, magnetization, (fig, ax[0, 2]))
    plot_CRM_parameter_path(t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, params_selected, (fig, ax[1, 0]), withEnergyGaps=withEnergyGaps, gap_string=gap_string, energy_gaps=energy_gaps, x_gap_plot_arr=x_gap_plot_arr, y_gap_plot_arr=y_gap_plot_arr)
    Full.plot_band_pop(band_pop, time_arr, T_pump, colors, (fig, ax[1, 1]))
    Full.plot_ket_COM(ket_COM, time_arr, T_pump, (fig, ax[1, 2]))

    set_plot_defaults(fig, ax)
    save_plot("CRM_All2")
    plt.show()


def plotN_CRM_poster(E, abs2_ket_eigen, abs2_ket, ket_COM, Chern_number, band_pop, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, time_arr, T_pump, colors, Delta_points, S_period, S_eigen, params_selected, N_Ham, withIPR=False, IPR_eigen=None, withEnergyGaps=False, gap_string="Direct", energy_gaps=None, x_gap_plot_arr=None, y_gap_plot_arr=None):
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)
    fig.suptitle(r'$C$=%i' % Chern_number)

    plot_CRM_modulation(t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, T_pump, (fig, ax[0, 0]))
    plot_CRM_spectrum(E, period_arr, T_pump, S_period, S_eigen, (fig, ax[0, 1]), withIPR=withIPR, IPR_eigen=IPR_eigen)
    plot_CRM_parameter_path(t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, params_selected, (fig, ax[1, 0]), withEnergyGaps=withEnergyGaps, gap_string=gap_string, energy_gaps=energy_gaps, x_gap_plot_arr=x_gap_plot_arr, y_gap_plot_arr=y_gap_plot_arr)
    Full.plot_ket_COM(ket_COM, time_arr, T_pump, (fig, ax[1, 1]))

    set_plot_defaults(fig, ax)
    save_plot("CRM_Poster")
    plt.show()
# END OF COMPOSITE PLOTS
# ----------------- END OF PLOTTING -----------------
