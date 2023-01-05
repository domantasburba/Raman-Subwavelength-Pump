from datetime import datetime
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import pathlib
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

from QOPack.Utility import *
from QOPack.Math import *
from QOPack.RamanTimeDepDetuning.ZakPhase import calc_general_Berry_phase, smooth_Berry_phase, create_Delta_points
import QOPack.RamanTimeDepDetuning.TimeEvolutionCRM as CRM


# --------------- GENERAL CALCULATIONS ---------------
def calc_general_Chern_number(k_eigvec, k_axis=1, t_axis=2, component_axis=-1):
    """Based on Fukui et al. (2005), Chern Numbers in Discretized Brillouin Zone: Efficient Method of Computing (Spin) Hall Conductances, 10.1143/JPSJ.74.1674"""

    # Assuming periodic k and t span.
    # PART 1: Eq. (7), calculate link variable from the wave functions.
    U_k = np.conj(k_eigvec) * np.roll(k_eigvec, -1, axis=k_axis)
    U_k = np.sum(U_k, axis=component_axis)
    # NOTE: If division by zero occurs, one should change the discretization.
    # (N_band, N_k, N_time)
    U_k = U_k / np.abs(U_k)

    U_time = np.conj(k_eigvec) * np.roll(k_eigvec, -1, axis=t_axis)
    U_time = np.sum(U_time, axis=component_axis)
    # NOTE: If division by zero occurs, one should change the discretization.
    # (N_band, N_k, N_time)
    U_time = U_time / np.abs(U_time)
    # END OF PART 1

    # PART 2: Eq. (8), calculate lattice field strength.
    # (N_band, N_k, N_time)
    F_field = np.log(U_k * np.roll(U_time, -1, axis=k_axis) / (np.roll(U_k, -1, axis=t_axis) * U_time))
    # END OF PART 2

    # PART 3: Eq. (9), calculate Chern number.
    # (N_band,)
    # Chern_number = np.sum(F_field, axis=(1, 2)) / (2.j * pi)
    Chern_number = np.imag(np.sum(F_field, axis=(1, 2)) / (2. * pi))
    # print("Chern number:", Chern_number)
    # END OF PART 3

    # check_if_real(Chern_number, "Chern number")
    check_if_int(Chern_number, "Chern number")
    Chern_number = np.int64(np.round(Chern_number))

    return Chern_number  # (N_band,)
# --------------- END OF GENERAL CALCULATIONS ---------------


def calc_CRM_Chern_numbers(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, k_span, time_span, modulate_parameters, N_k, N_time):
    epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time = full_modulate(epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, time_span, modulate_parameters, N_time)
    k_eigvec = full_exact_k_eigvec_kt(epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time, gamma, k_span, N_state, N_k, N_time)
    Chern_number = calc_general_Chern_number(k_eigvec)  # (2,)

    return Chern_number  # (2,)


def create_arg_arrs(time_arr):
    arg_time = 2. * pi * time_arr
    sin_arg_time = np.sin(arg_time)
    cos_arg_time = np.cos(arg_time)
    return arg_time, sin_arg_time, cos_arg_time


def exact_Omega_kt(k_arr, t_pm, t_0alpha, gamma, N_k, N_time):
    # (2, N_k, N_time); 2 -> [0, 1]; N_k -> [-pi/a_0, ..., pi/a_0]; N_time -> [0, ..., time_end]
    Omega = np.zeros((2, N_k, N_time), dtype=np.complex128)
    Omega[0] = 2. * (t_0alpha[0] - t_0alpha[1])[np.newaxis, :] * np.cos(k_arr + gamma[0])[:, np.newaxis]
    Omega[1] = t_pm[0, np.newaxis, :] * np.exp(-1.j * (k_arr + gamma[-1]))[:, np.newaxis] + \
               t_pm[1, np.newaxis, :] * np.exp(1.j * (k_arr + gamma[1]))[:, np.newaxis]

    return Omega


def exact_Delta_kt(Omega, k_arr, tn_alpha, epsilon, N_state):
    # (N_k, N_time); N_k -> [-pi/a_0, ..., pi/a_0]; N_time -> [0, ..., time_end]
    return (epsilon[0] - epsilon[1])[np.newaxis, :] + 2. * (tn_alpha[0] - tn_alpha[1])[np.newaxis, :] * np.cos(N_state * k_arr)[:, np.newaxis] + Omega[0]


def exact_Lambda_kt(k_arr, epsilon, t_0alpha, tn_alpha, gamma, N_state):
    # (2, N_k, N_time); 2 -> [s, p]; N_k -> [-pi/a_0, ..., pi/a_0]; N_time -> [0, ..., time_end]
    epsilon_term = epsilon[:, np.newaxis, :]
    natural_term = 2. * tn_alpha[:, np.newaxis, :] * np.cos(N_state * k_arr)[np.newaxis, :, np.newaxis]
    interchain_term = 2. * t_0alpha[:, np.newaxis, :] * np.cos(k_arr + gamma[0])[np.newaxis, :, np.newaxis]
    # UNUSED ALTERNATIVE (all other instances of k_arr and k_arr bounds would need to be changed):
    # natural_term = 2. * tn_alpha[:, np.newaxis, :] * np.cos(k_arr)[np.newaxis, :, np.newaxis]
    # interchain_term = 2. * t_0alpha[:, np.newaxis, :] * np.cos(k_arr / N_state + gamma[0])[np.newaxis, :, np.newaxis]

    return epsilon_term + natural_term + interchain_term


def exact_E_kt(Delta, Lambda, Omega, N_k, N_time):
    # (2, N_k, N_time); 2 -> [-, +]; N_k -> [-pi/a_0, ..., pi/a_0]; N_time -> [0, ..., time_end]
    E_k = np.zeros((2, N_k, N_time), dtype=np.float64)
    Lambda_term = 0.5 * (Lambda[0] + Lambda[1])
    sq_term = 0.5 * np.sqrt(Delta**2 + 4.0 * np.abs(Omega[1])**2)
    check_if_real(sq_term, "E_k")
    # E_k[0] = Lambda_term - np.real(sq_term)
    # E_k[1] = Lambda_term + np.real(sq_term)
    E_k[0] = -np.real(sq_term)
    E_k[1] = np.real(sq_term)

    return E_k


def exact_N_eigvec_kt(E_k, Lambda, Delta, Omega):
    # (2, N_k, N_time); 2 -> [-, +]; N_k -> [-pi/a_0, ..., pi/a_0]; N_time -> [0, ..., time_end]
    Omega_term = np.abs(Omega[1])**2
    # energy_term = (E_k - (Lambda[0])[np.newaxis, ...])**2
    energy_term = (E_k - 0.5 * Delta[np.newaxis, ...])**2

    return 1.0 / np.sqrt(Omega_term[np.newaxis, ...] + energy_term)


def exact_k_eigvec_kt(E_k, Lambda, Delta, N_eigvec, Omega, N_k, N_time):
    # (2, N_k, N_time, 2); 2 -> [-, +]; N_k -> [-pi/a_0, ..., pi/a_0]; N_time -> [0, ..., time_end]; 2 -> [x_component, y_component]
    k_eigvec = np.zeros((2, N_k, N_time, 2), dtype=np.complex128)
    k_eigvec[0, ..., 0] = N_eigvec[0] * np.conj(Omega[1])
    # k_eigvec[0, ..., 1] = N_eigvec[0] * (E_k[0] - Lambda[0])
    k_eigvec[0, ..., 1] = N_eigvec[0] * (E_k[0] - 0.5 * Delta)
    k_eigvec[1, ..., 0] = N_eigvec[1] * np.conj(Omega[1])
    # k_eigvec[1, ..., 1] = N_eigvec[1] * (E_k[1] - Lambda[0])
    k_eigvec[1, ..., 1] = N_eigvec[1] * (E_k[1] - 0.5 * Delta)

    return k_eigvec


def full_exact_E_kt(epsilon, t_pm, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k, N_time):
    # (2, N_k, N_time); 2 -> [-, +]; N_k -> [-pi/a_0, ..., pi/a_0]; N_time -> [0, ..., time_end]
    k_arr = np.linspace(k_span[0], k_span[1], N_k)
    Omega = exact_Omega_kt(k_arr, t_pm, t_0alpha, gamma, N_k, N_time)
    Delta = exact_Delta_kt(Omega, k_arr, tn_alpha, epsilon, N_state)
    Lambda = exact_Lambda_kt(k_arr, epsilon, t_0alpha, tn_alpha, gamma, N_state)
    E_k = exact_E_kt(Delta, Lambda, Omega, N_k, N_time)

    return E_k


def full_exact_k_eigvec_kt(epsilon, t_pm, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k, N_time):
    # (2, N_k, N_time, 2); 2 -> [-, +]; N_k -> [-pi/a_0, ..., pi/a_0]; 2 -> [x_component, y_component]
    k_arr = np.linspace(k_span[0], k_span[1], N_k)
    Omega = exact_Omega_kt(k_arr, t_pm, t_0alpha, gamma, N_k, N_time)
    Delta = exact_Delta_kt(Omega, k_arr, tn_alpha, epsilon, N_state)
    Lambda = exact_Lambda_kt(k_arr, epsilon, t_0alpha, tn_alpha, gamma, N_state)
    E_k = exact_E_kt(Delta, Lambda, Omega, N_k, N_time)
    N_eigvec = exact_N_eigvec_kt(E_k, Lambda, Delta, Omega)
    k_eigvec = exact_k_eigvec_kt(E_k, Lambda, Delta, N_eigvec, Omega, N_k, N_time)
    # check_ket_orthonorm(k_eigvec, "k_eigvec")

    return k_eigvec  # (2, N_k, N_time, 2)


def full_modulate(epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, time_span, modulate_parameters, N_time):
    time_arr = np.linspace(time_span[0], time_span[1], N_time)

    t_pm_time = np.empty((2, N_time), dtype=np.float64)
    epsilon_time = np.empty_like(t_pm_time)
    t_0alpha_time = np.empty_like(t_pm_time)
    for i, t in enumerate(time_arr):
        t_pm_time[:, i], epsilon_time[:, i], t_0alpha_time[:, i] = CRM.modulate_CRM_Gediminas(t, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, 1.0, modulate_parameters, 1.0)
    tn_alpha_time = np.full((2, N_time), tn_alpha[:, np.newaxis])

    return epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time


# ----------------- PLOTTING -----------------
def set_selected_bar(bar_arr, select_bar, t_bar, epsilon_bar, t_0_bar, i):
    if select_bar[0] == 0:
        t_bar[select_bar[1]] = bar_arr[i]
    if select_bar[0] == 1:
        epsilon_bar[select_bar[1]] = bar_arr[i]
    if select_bar[0] == 2:
        t_0_bar[select_bar[1]] = bar_arr[i]


def set_selected_axis_label(select_bar, axis):
    # axis argument should be either "x" or "y".
    if axis == "x":
        plt_label = plt.xlabel
    if axis == "y":
        plt_label = plt.ylabel
    if np.array_equal(select_bar, np.array([0, 0])):
        plt_label(r"$\bar{t}$ / $E_R$")
    if np.array_equal(select_bar, np.array([0, 1])):
        plt_label(r"$\bar{\bar{t}}$ / $E_R$")
    if np.array_equal(select_bar, np.array([1, 0])):
        plt_label(r"$E_0$ / $E_R$")
    if np.array_equal(select_bar, np.array([1, 1])):
        plt_label(r"$ϵ$ / $E_R$")
    if np.array_equal(select_bar, np.array([2, 0])):
        plt_label(r"$\bar{t}_0$ / $E_R$")
    if np.array_equal(select_bar, np.array([2, 1])):
        plt_label(r"$\tau$ / $E_R$")


def plot_Berry_arr(Berry_arr, bar0_arr, bar1_arr, select_bar0, select_bar1, select_k):
    # plt.style.use("classic")
    fig = plt.figure(r"Berry phase", figsize=(6, 7))
    ax = plt.axes()

    set_selected_axis_label(select_bar0, "x")
    set_selected_axis_label(select_bar1, "y")

    # plt.tick_params(axis="both", direction="in")
    # ax.xaxis.set_ticks_position("both")
    # ax.yaxis.set_ticks_position("both")
    plt.pcolormesh(bar0_arr, bar1_arr, np.transpose(Berry_arr), cmap=plt.get_cmap("jet"), shading="nearest")
    cbar = plt.colorbar(label=r"Berry phase $\gamma_B(k=%.2f)$" % select_k, location="top", aspect=30)
    cbar.ax.xaxis.label.set_size(12)
    cbar.ax.tick_params(length=2, direction="in")
    cbar.ax.xaxis.set_ticks_position("both")

    set_plot_defaults(fig, ax, addGrid=False)
    save_plot("Berry_Phase")
    plt.show()


def plot_Chern_arr(Chern_arr, bar0_arr, bar1_arr, select_bar0, select_bar1):
    # plt.style.use("classic")
    fig = plt.figure(r"Chern number", figsize=(6, 7))
    ax = plt.axes()

    set_selected_axis_label(select_bar0, "x")
    set_selected_axis_label(select_bar1, "y")

    # plt.tick_params(axis="both", direction="in")
    # ax.xaxis.set_ticks_position("both")
    # ax.yaxis.set_ticks_position("both")
    plt.pcolormesh(bar0_arr, bar1_arr, np.transpose(np.real(Chern_arr)), cmap=plt.get_cmap("jet"), shading="nearest")
    cbar = plt.colorbar(label=r"Chern number $C$", location="top", aspect=30)
    cbar.ax.xaxis.label.set_size(12)
    cbar.ax.tick_params(which="both", length=2, direction="in")
    cbar.ax.xaxis.set_ticks_position("both")

    set_plot_defaults(fig, ax, addGrid=False)
    save_plot("Chern_Array")
    plt.show()


def plot_x_Wannier(x_Wannier, period_arr, colors, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    T_pump = period_arr[-1]
    ax.set_xlabel(r"t / $T$")
    ax.set_ylabel(r"$x_W$ / $a_0$")
    ax.set_ylim(-0.1, 1.1)
    [ax.plot(period_arr/T_pump, x_Wannier[i, :], color=colors[i]) for i in range(x_Wannier.shape[0])]

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("x_Wannier")
        plt.show()


def plot_CRM_Energy_kt(E_k, k_arr, period_arr, colors, S_period, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    ax.set_title(r"$t$=%.2f" % period_arr[S_period])
    ax.set_xlabel(r"$k$ / rad")
    ax.set_ylabel(r"$E$ / $E_R$")
    [ax.plot(k_arr, E_k[i, :, S_period], color=colors[i]) for i in range(E_k.shape[0])]

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("CRM_Energy_kt")
        plt.show()


def plot_inset_x_Wannier(Chern_number, x_Wannier, period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, modulate_parameters, params_selected):
    ##### MAIN PLOT: WANNIER CENTER #####
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    ax = plt.axes()

    T_pump = period_arr[-1]
    N_period = len(period_arr)

    ax.set_title(r"$C$=%i" % Chern_number)
    ax.set_xlabel(r"t / $T$")
    ax.set_ylabel(r"$x_W$ / $a_0$")
    ax.set_ylim(-0.1, 1.1)
    ax.plot(period_arr/T_pump, x_Wannier[0, :], color="black")
    set_plot_defaults(fig, ax)
    ##### END OF MAIN PLOT: WANNIER CENTER #####

    ##### INSET PLOT: CRM PARAMETER PATH #####
    # TODO Why does it take in the static parameters?
    Delta_points = create_Delta_points(epsilon, t_0alpha, params_selected)

    t_pm_diff, epsilon_diff, t_0alpha_diff = CRM.calc_CRM_parameter_diffs(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, 0, 0, 0, T_pump, modulate_parameters, "Gediminas", N_period)

    x_param_diff_arr = CRM.set_selected_param_diff_arr(t_pm_diff, epsilon_diff, t_0alpha_diff, params_selected[0])
    y_param_diff_arr = CRM.set_selected_param_diff_arr(t_pm_diff, epsilon_diff, t_0alpha_diff, params_selected[1])

    ax_inset = plt.axes([0, 0, 1, 1])
    # ip = InsetPosition(ax, [0.55, 0.1, 0.4, 0.4])
    ip = InsetPosition(ax, [0.65, 0.1, 0.3, 0.3])
    # ip = InsetPosition(ax, [0.1, 0.65, 0.3, 0.3])
    ax_inset.set_axes_locator(ip)

    if params_selected[0] == 0:
        ax_inset.set_xlabel(r"$t_1 - t_{-1}$ / $E_R$")
    if params_selected[0] == 1:
        ax_inset.set_xlabel(r"$\bar{ϵ}^{(s)} - \bar{ϵ}^{(p)}$ / $E_R$")
    if params_selected[0] == 2:
        ax_inset.set_xlabel(r"$t_{0s} - t_{0p}$ / $E_R$")
    if params_selected[1] == 0:
        ax_inset.set_ylabel(r"$t_1 - t_{-1}$ / $E_R$")
    if params_selected[1] == 1:
        ax_inset.set_ylabel(r"$\bar{ϵ}^{(s)} - \bar{ϵ}^{(p)}$ / $E_R$")
    if params_selected[1] == 2:
        ax_inset.set_ylabel(r"$t_{0s} - t_{0p}$ / $E_R$")

    ax_inset.scatter(x_param_diff_arr, y_param_diff_arr, s=30, color="black")
    ax_inset.plot([Delta_points[:, 0]], [Delta_points[:, 1]], marker="*", markersize=10, markeredgecolor="red", markerfacecolor="red")
    set_plot_defaults(fig, ax_inset)
    ##### END OF INSET PLOT: CRM PARAMETER PATH #####

    save_plot("Inset_x_Wannier")
    plt.show()


def plot_multiple_x_Wannier(x_Wannier_arr, period_arr, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    T_pump = period_arr[-1]
    ax.set_xlabel(r"t / $T$")
    # ax.set_ylabel(r"$x_W$ / $a_0$")
    ax.set_ylabel(r"$\gamma_{Zak}$ / $2\pi$")
    # ax.set_ylim(-0.1, 1.1)
    # ax.set_ylim(-0.1, 2.1)
    # ax.set_ylim(-0.4, 2.1)
    ax.plot(period_arr/T_pump, x_Wannier_arr[0, :], color="red")
    ax.plot(period_arr/T_pump, x_Wannier_arr[1, :], linestyle="dashed", color="red")
    ax.plot(period_arr/T_pump, x_Wannier_arr[2, :], linestyle="dotted", color="red")

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("Multiple_x_Wannier")
        plt.show()
# ----------------- END OF PLOTTING -----------------


def full_exact_CRM_Zak_arr_t(Chern_number, N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, T_pump, modulate_parameters, select_params, time_selected, colors, N_k, N_period, plotResults=True):
    k_arr = np.linspace(-pi, pi, N_k)
    period_arr = np.linspace(0., T_pump, N_period)
    S_period = x2index((0., T_pump), N_period, time_selected)
    epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time = full_modulate(epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, (0., 1.), modulate_parameters, N_period)

    k_eigvec = full_exact_k_eigvec_kt(epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time, gamma, (-pi, pi), N_state, N_k, N_period)  # (2, N_k, N_period, 2)
    Zak_arr_t = calc_general_Berry_phase(k_eigvec, prod_axis=1)  # (2, N_period)
    Zak_arr_t = smooth_Berry_phase(Zak_arr_t)
    x_Wannier = Zak_arr_t / (2*pi)  # (2, N_period)
    if plotResults:
        plot_x_Wannier(x_Wannier, period_arr, colors)
        # plot_inset_x_Wannier(Chern_number, x_Wannier, period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, modulate_parameters, select_params)

    return Zak_arr_t


def full_multiple_exact_CRM_Zak_arr_t(Chern_number, N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, T_pump, modulate_parameters, select_params, time_selected, colors, N_k, N_period):
    k_arr = np.linspace(-pi, pi, N_k)
    period_arr = np.linspace(0., T_pump, N_period)
    S_period = x2index((0., T_pump), N_period, time_selected)

    if modulate_parameters[1] == True:
        two_t0_diff = 2 * (t_0alpha[0] - t_0alpha[1])
        epsilon_bar_arr = np.array([0.00, 1.00, 2.00]) * two_t0_diff
        epsilon_barbar_arr = np.array([1.25, 0.50, 0.50]) * two_t0_diff

        N_Chern = len(epsilon_bar_arr)  # = 3
    else:
        half_epsilon_diff = (epsilon[0] - epsilon[1]) / 2
        t_0_bar_arr = np.array([0.00, 1.00, -1.00]) * half_epsilon_diff
        t_0_barbar_arr = np.array([1.25, 0.5, 0.5]) * half_epsilon_diff

        N_Chern = len(t_0_bar_arr)  # = 3

    x_Wannier_arr = np.empty((N_Chern, N_period), dtype=np.float64)
    for idx_Chern in range(N_Chern):
        if modulate_parameters[1] == True:
            epsilon_bar[0] = epsilon_bar_arr[idx_Chern]
            epsilon_bar[1] = epsilon_barbar_arr[idx_Chern]
            epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time = full_modulate(epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, (0., 1.), modulate_parameters, N_period)
        else:
            t_0_bar[0] = t_0_bar_arr[idx_Chern]
            t_0_bar[1] = t_0_barbar_arr[idx_Chern]
            epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time = full_modulate(epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, (0., 1.), modulate_parameters, N_period)

        k_eigvec = full_exact_k_eigvec_kt(epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time, gamma, (-pi, pi), N_state, N_k, N_period)  # (2, N_k, N_period, 2)
        Zak_arr_t = calc_general_Berry_phase(k_eigvec, prod_axis=1)  # (2, N_period)
        Zak_arr_t = smooth_Berry_phase(Zak_arr_t)
        x_Wannier_arr[idx_Chern, :] = Zak_arr_t[0, :] / (2*pi)

        print(x_Wannier_arr[idx_Chern, -1] - x_Wannier_arr[idx_Chern, 0])

        if modulate_parameters[2] == True:
            # Kind of hacky, solution could be more general.
            if idx_Chern == 0:
                x_Wannier_arr[idx_Chern, :] -= 0.5
            if idx_Chern == 2:
                x_Wannier_arr[idx_Chern, :] -= 1.0

        if modulate_parameters[1] == True:
            # Kind of hacky, solution could be more general.
            if idx_Chern == 0:
                x_Wannier_arr[idx_Chern, N_period//2:] += 1.0
                # print(np.gradient(x_Wannier_arr[idx_Chern, N_period//2-3:N_period//2+3]))
            if idx_Chern == 2:
                x_Wannier_arr[idx_Chern, :] -= 0.5

    if modulate_parameters[1] == True:
        save_array_NPY(x_Wannier_arr, "x_Wannier_arr!epsilon")
    else:
        save_array_NPY(x_Wannier_arr, "x_Wannier_arr!t0alpha")

    ##### PLOTTING #####
    plot_multiple_x_Wannier(x_Wannier_arr, period_arr)
    ##### END OF PLOTTING #####


def full_exact_CRM_Energy_kt(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, T_pump, modulate_parameters, time_selected, colors, N_k, N_period, plotResults=True):
    k_arr = np.linspace(-pi, pi, N_k)
    period_arr = np.linspace(0., T_pump, N_period)
    S_period = x2index((0., T_pump), N_period, time_selected)
    epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time = full_modulate(epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, (0., 1.), modulate_parameters, N_period)

    E_k = full_exact_E_kt(epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time, gamma, (-pi, pi), N_state, N_k, N_period)
    if plotResults:
        plot_CRM_Energy_kt(E_k, k_arr, period_arr, colors, S_period)

    return E_k


def full_Berry_arr(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, k_span, time_span, modulate_parameters, bar0_span, bar1_span, select_bar0, select_bar1, N_k, N_time, N_bar0, N_bar1, select_k):
    bar0_arr = np.linspace(bar0_span[0], bar0_span[1], N_bar0)
    bar1_arr = np.linspace(bar1_span[0], bar1_span[1], N_bar1)
    Berry_arr = np.zeros((N_bar0, N_bar1), dtype=np.float64)
    for i in range(N_bar0):
        set_selected_bar(bar0_arr, select_bar0, t_bar, epsilon_bar, t_0_bar, i)
        for j in range(N_bar1):
            set_selected_bar(bar1_arr, select_bar1, t_bar, epsilon_bar, t_0_bar, j)
            epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time = full_modulate(epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, time_span, modulate_parameters, N_time)
            k_eigvec = full_exact_k_eigvec_kt(epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time, gamma, k_span, N_state, N_k, N_time)
            Berry_arr[i, j] = calc_general_Berry_phase(k_eigvec, 1)[x2index(k_span, N_k, select_k)]
    plot_Berry_arr(Berry_arr, bar0_arr, bar1_arr, select_bar0, select_bar1, select_k)


def full_Chern_arr(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, k_span, time_span, modulate_parameters, bar0_span, bar1_span, select_bar0, select_bar1, N_k, N_time, N_bar0, N_bar1):
    bar0_arr = np.linspace(bar0_span[0], bar0_span[1], N_bar0)
    bar1_arr = np.linspace(bar1_span[0], bar1_span[1], N_bar1)
    Chern_arr = np.zeros((N_bar0, N_bar1), dtype=np.complex128)
    for i in range(N_bar0):
        set_selected_bar(bar0_arr, select_bar0, t_bar, epsilon_bar, t_0_bar, i)
        for j in range(N_bar1):
            set_selected_bar(bar1_arr, select_bar1, t_bar, epsilon_bar, t_0_bar, j)
            epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time = full_modulate(epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, time_span, modulate_parameters, N_time)
            k_eigvec = full_exact_k_eigvec_kt(epsilon_time, t_pm_time, t_0alpha_time, tn_alpha_time, gamma, k_span, N_state, N_k, N_time)
            Chern_arr[i, j] = calc_general_Chern_number(k_eigvec)[0]
    # check_if_real(Chern_arr, "Chern array")
    # check_if_int(Chern_arr, "Chern array")
    # plot_Chern_arr(np.int64(np.round(np.real(Chern_arr))), bar0_arr, bar1_arr, select_bar0, select_bar1)
    plot_Chern_arr(np.real(Chern_arr), bar0_arr, bar1_arr, select_bar0, select_bar1)
