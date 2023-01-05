import functools
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from QOPack.Math import *
from QOPack.Utility import *
from QOPack.RamanTimeDepDetuning.ZakPhase import create_Delta_points
from QOPack.RamanTimeDepDetuning.AdiabaticPumping import calc_CRM_Chern_numbers

# UNUSED RICE-MELE FORMULATION:
# N_lat = 50  # Number of Rice-Mele cells.
# N_2lat -> [..., (C1, r), (C2, r), ...], where Ci is the i-th Rice-Mele chain and r is the lattice index.
# N_2lat = 2 * N_lat
# N_Ham = 2 * (2 * N_lat)  # 2 Rice-Mele chains and 2 sublattice indices


@njit(cache=True)
def create_centered_single(N_lat, N_Ham):
    psi_initial = np.zeros(N_Ham, dtype=np.complex128)
    psi_initial[2*(N_lat//2)] = 1.
    # psi_initial[2*(N_lat//2)] = 0.5
    # psi_initial[2*(N_lat//2+1)+1] = 0.5

    return psi_initial


# @njit(cache=True)
def create_centered_eigen(psi_eigen, N_lat, N_Ham):
    Re_psi0_eigen = np.real(psi_eigen[0, :, 2*(N_lat//2)])
    Im_psi0_eigen = np.imag(psi_eigen[0, :, 2*(N_lat//2)])

    phi = np.arctan(Im_psi0_eigen / Re_psi0_eigen)
    for i in range(N_Ham):
        if Re_psi0_eigen[i] < 0:
            phi[i] += np.pi
    
    # plt.plot(phi)
    # plt.show()
    
    psi_initial = np.zeros(N_Ham, dtype=np.complex128)
    for i in range(N_Ham // 2 - 1):
        psi_initial += np.exp(1j * phi[i]) * psi_eigen[0, i, :]

    # Normalization
    psi_initial /= np.sqrt(np.sum(abs2(psi_initial)))

    # plt.plot(psi_initial.real, color="black")
    # plt.plot(psi_initial.real + psi_initial.imag, color="red")
    # plt.show()

    return psi_initial


@njit(cache=True)
def create_centered_Gaussian(sigma, N_lat):
    x_arr = np.arange(N_lat) - N_lat // 2
    x_arr = np.repeat(x_arr, 2)

    psi_initial = np.exp(-0.5*(x_arr/sigma)**2)

    # Normalization
    psi_initial /= np.sqrt(np.sum(abs2(psi_initial)))

    return psi_initial.astype(np.complex128)


@njit(cache=True)
def create_selected_eigen(psi_eigen, S_eigen):
    return psi_eigen[0, S_eigen, :]


@njit(cache=True)
def modulate_parameters(t, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, modulate_parameters):
    if modulate_parameters[0]:
        T_pm = t_bar[0] + np.array([-1, 1]) * t_bar[1] * np.sin(2 * np.pi * t)
    else:
        T_pm = t_pm
    if modulate_parameters[1]:
        Epsilon = np.array([epsilon_bar[0] + epsilon_bar[1] * np.cos(2 * np.pi * t), 0])
        # Epsilon = np.array([epsilon_bar[0]/2 + epsilon_bar[1]/2 * np.cos(2*np.pi*t), -(epsilon_bar[0]/2 + epsilon_bar[1]/2 * np.cos(2*np.pi*t))])
    else:
        Epsilon = epsilon
    if modulate_parameters[2]:
        # t_0alpha_t = t_0_bar[0] + np.array([-1, 1]) * t_0_bar[1] * np.cos(2 * np.pi * t)
        t_0alpha_t = np.array([t_0_bar[0] + t_0_bar[1] * np.cos(2 * np.pi * t), 0])
    else:
        t_0alpha_t = t_0alpha
    
    return T_pm, Epsilon, t_0alpha_t


@njit(cache=True)
def calc_CRM_Hamiltonian(t, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, N_state, N_lat, N_Ham):
    # PARAMETER MODULATION
    T_pm, Epsilon, t_0alpha_t = modulate_parameters(t, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, modulate_parameters)
    # END OF PARAMETER MODULATION
    
    Ham = np.zeros((N_Ham, N_Ham), dtype=np.complex128)

    # *************** H ASSIGNMENT *******************
    # ---------- H_0 ASSIGNMENT -----------------
    # epsilon ASSIGNMENT
    for i in range(N_lat):  # N_lat -> [..., r, r + 1, ...]
        for j in range(2):  # 2 -> [s, p]
            Ham[2*i+j, 2*i+j] += Epsilon[j]
    # END OF epsilon ASSIGNMENT

    # tn_alpha ASSIGNMENT
    for i in range(N_lat - N_state):  # N_lat -> [..., r, r + 1, ...]
        for j in range(2):  # 2 -> [s, p]
            Ham[2*i+j, 2*(i+N_state)+j] += tn_alpha[j]
            Ham[2*(i+N_state)+j, 2*i+j] += tn_alpha[j]
    # END OF tn_alpha ASSIGNMENT
    # ---------- END OF H_0 ASSIGNMENT ----------

    # ---------- U ASSIGNMENT -----------------
    # U_{-1} ASSIGNMENT
    T_m1_term = T_pm[0] * np.exp(1j * gamma[-1])
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
    T_p1_term = T_pm[1] * np.exp(1j * gamma[1])
    T_p1_conj_term = np.conj(T_p1_term)
    for i in range(N_lat - 1):  # N_lat -> [..., r, r + 1, ...]
        Ham[2*i+1, 2*(i+1)] += T_p1_term
        Ham[2*(i+1), 2*i+1] += T_p1_conj_term
    # END OF U_1 ASSIGNMENT
    # ---------- END OF U ASSIGNMENT ----------
    # *************** END OF H ASSIGNMENT **********

    return Ham


def calc_CRM_time_spectrum(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, N_state, N_lat, N_period, N_Ham):
    calc_Hamiltonian = functools.partial(calc_CRM_Hamiltonian, t_bar=t_bar, epsilon_bar=epsilon_bar, t_0_bar=t_0_bar, t_pm=t_pm, epsilon=epsilon, t_0alpha=t_0alpha, tn_alpha=tn_alpha, gamma=gamma, modulate_parameters=modulate_parameters, N_state=N_state, N_lat=N_lat, N_Ham=N_Ham)

    E = np.empty((N_period, N_Ham), dtype=np.float64)
    psi_eigen = np.empty((N_period, N_Ham, N_Ham), dtype=np.complex128)
    for i in range(N_period):
        Ham = calc_Hamiltonian(period_arr[i])
        E[i, :], v = np.linalg.eigh(Ham)
        psi_eigen[i, :] = v.T
    
    return E, psi_eigen


def time_evolution_CRM(psi_initial, time_arr, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, N_state, N_lat, N_Ham, N_time):
    calc_Hamiltonian = functools.partial(calc_CRM_Hamiltonian, t_bar=t_bar, epsilon_bar=epsilon_bar, t_0_bar=t_0_bar, t_pm=t_pm, epsilon=epsilon, t_0alpha=t_0alpha, tn_alpha=tn_alpha, gamma=gamma, modulate_parameters=modulate_parameters, N_state=N_state, N_lat=N_lat, N_Ham=N_Ham)

    def Schrodinger(t, psi):
        Ham = calc_Hamiltonian(t)
        return -1.j * Ham @ psi

    sol = solve_ivp(Schrodinger, time_span, psi_initial, t_eval=time_arr)

    # SAVING SOLUTION INTO 2D ARRAY.
    psi = np.zeros((N_time, N_Ham), dtype=np.complex128)
    for j in range(N_Ham):
        psi[:, j] = sol.y[j]
    # END OF SAVING SOLUTION INTO 2D ARRAY.
    
    return psi


@njit(cache=True)
def calc_psi_COM(abs2_psi, N_state, N_lat):
    x_arr = np.arange(N_lat) - N_lat // 2
    x_arr = np.repeat(x_arr, 2)

    # x_mul_psi2.shape = (N_time, N_Ham)
    x_mul_psi2 = np.expand_dims(x_arr, axis=0) * abs2_psi

    psi_COM = np.sum(x_mul_psi2, axis=-1) / N_state

    return psi_COM


def calc_CRM_parameter_diffs(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, modulate_parameters, N_period):
    t_pm_period = np.empty((2, N_period), dtype=np.float64)
    epsilon_period = np.empty_like(t_pm_period, dtype=np.float64)
    t_0alpha_period = np.empty_like(t_pm_period, dtype=np.float64)
    for i, time in enumerate(period_arr):
        t_pm_period[:, i], epsilon_period[:, i], t_0alpha_period[:, i] = modulate_parameters(time, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, modulate_parameters)
    
    t_pm_diff = t_pm_period[0, :] - t_pm_period[1, :]
    epsilon_diff = epsilon_period[0, :] - epsilon_period[1, :]
    t_0alpha_diff = t_0alpha_period[0, :] - t_0alpha_period[1, :]

    return t_pm_diff, epsilon_diff, t_0alpha_diff


# ---------- PLOTTING -----------------
def get_fig_ax(fig_ax=None):
    if fig_ax is None:
        fig = plt.figure()
        ax = plt.axes()
    else:
        fig, ax = fig_ax
    
    return fig, ax


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
def plot_psi_pcolormesh(abs2_psi, time_arr, N_Ham, fig_ax=None):
    """fig_ax - (fig, ax), tuple of size 2.
    If fig_ax is None, create a standalone plot. Otherwise, plot in the given figure and axis.
    """
    fig, ax = get_fig_ax(fig_ax)

    coeff_plot_arr = np.arange(N_Ham, dtype=np.int64)
    ax_pcolormesh = ax.pcolormesh(coeff_plot_arr, time_arr, abs2_psi, cmap="viridis", shading="auto")
    plt.colorbar(ax_pcolormesh, label=r"$|\langle W_R|\phi (t)\rangle|^2$", ax=ax)
    ax.set_xlabel(r"$|\langle W_R|\phi (t)\rangle|^2$")
    ax.set_ylabel(r"t / $\tau$")
    
    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("Time_Evolution_RM")
        plt.show()


def plot_psi_COM(psi_COM, time_arr, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    ax.set_xlabel(r"t / $\tau$")
    ax.set_ylabel(r"$\bar{x}$ / $a_0$")
    ax.plot(time_arr, psi_COM, color="black")

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("psi_COM")
        plt.show()


def plot_CRM_spectrum(E, period_arr, S_time, S_eigen, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    ax.set_xlabel(r"t / $\tau$")
    ax.set_ylabel(r"E / $E_R$")
    [ax.plot(period_arr, E[:, i], color="black") for i in range(E.shape[1])]
    ax.scatter(period_arr[S_time], E[S_time, S_eigen], s=30, color="magenta")

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("CRM_Spectrum")
        plt.show()


def plot_CRM_eigen(E, abs2_psi_eigen, period_arr, S_time, S_eigen, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    cell_arr = np.arange(abs2_psi_eigen.shape[-1]) / 2

    ax.set_title(r"$t$=%.2f, $E$=%.2f" % (period_arr[S_time], E[S_time, S_eigen]))
    ax.set_xlabel(r"Cell index $r$")
    ax.set_ylabel(r"$|\psi|^2$")
    ax.axhline(0, color="black", ls="--")
    # ax.plot(psi_eigen[S_time, S_eigen], color="red")
    ax.bar(cell_arr[::2], abs2_psi_eigen[S_time, S_eigen, ::2], color="red", width=0.5)
    ax.bar(cell_arr[1::2], abs2_psi_eigen[S_time, S_eigen, 1::2], color="black", width=0.5)

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("CRM_Eigen")
        plt.show()


def plot_CRM_parameter_path(t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, params_selected, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    x_param_diff_arr = set_selected_param_diff_arr(t_pm_diff, epsilon_diff, t_0alpha_diff, params_selected[0])
    y_param_diff_arr = set_selected_param_diff_arr(t_pm_diff, epsilon_diff, t_0alpha_diff, params_selected[1])
    set_selected_axis_label(ax, params_selected[0], 'x')
    set_selected_axis_label(ax, params_selected[1], 'y')
    ax.scatter(x_param_diff_arr, y_param_diff_arr, s=30, color="black")
    ax.plot([Delta_points[:, 0]], [Delta_points[:, 1]], marker="*", markersize=10, markeredgecolor="red", markerfacecolor="red")

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("CRM_Parameter_Path")
        plt.show()


def plot_CRM_modulation(t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    ax.set_xlabel(r"t / $\tau$")
    ax.set_ylabel(r"$x_0 - x_1$ / $E_R$")
    ax.plot(period_arr, t_pm_diff, color="black")
    ax.plot(period_arr, epsilon_diff, color="red")
    ax.plot(period_arr, t_0alpha_diff, color="blue")
    ax.legend([r'$t_1 - t_{-1}$', r'$\bar{ϵ}^{(s)} - \bar{ϵ}^{(p)}$', r'$t_{0s} - t_{0p}$'])

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("CRM_Modulation")
        plt.show()
# END OF SINGLE PLOTS


# COMPOSITE PLOTS
def plotN_CRM_psi(abs2_psi, psi_COM, time_arr, N_Ham):
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    plot_psi_pcolormesh(abs2_psi, time_arr, N_Ham, (fig, ax[0]))
    plot_psi_COM(psi_COM, time_arr, (fig, ax[1]))

    set_plot_defaults(fig, ax)
    save_plot("CRM_psi")
    plt.show()


def plotN_CRM_spectrum_eigen(E, abs2_psi_eigen, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, Delta_points, S_time, S_eigen, params_selected):
    # fig, ax = plt.subplots(1, 2)
    # fig.set_size_inches(12, 5)
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)

    plot_CRM_spectrum(E, period_arr, S_time, S_eigen, (fig, ax[0, 0]))
    plot_CRM_eigen(E, abs2_psi_eigen, period_arr, S_time, S_eigen, (fig, ax[0, 1]))
    plot_CRM_parameter_path(t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, params_selected, (fig, ax[1, 0]))
    plot_CRM_modulation(t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, (fig, ax[1, 1]))

    set_plot_defaults(fig, ax)
    save_plot("CRM_Spectrum_Eigen")
    plt.show()


def plotN_CRM_all(E, abs2_psi_eigen, abs2_psi, psi_COM, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, time_arr, Delta_points, S_time, S_eigen, params_selected, N_Ham):
    # fig, ax = plt.subplots(1, 2)
    # fig.set_size_inches(12, 5)
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(18, 8)

    plot_CRM_modulation(t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, (fig, ax[0, 0]))
    plot_CRM_spectrum(E, period_arr, S_time, S_eigen, (fig, ax[0, 1]))
    plot_CRM_eigen(E, abs2_psi_eigen, period_arr, S_time, S_eigen, (fig, ax[0, 2]))
    plot_CRM_parameter_path(t_pm_diff, epsilon_diff, t_0alpha_diff, Delta_points, params_selected, (fig, ax[1, 0]))
    plot_psi_pcolormesh(abs2_psi, time_arr, N_Ham, (fig, ax[1, 1]))
    plot_psi_COM(psi_COM, time_arr, (fig, ax[1, 2]))

    set_plot_defaults(fig, ax)
    save_plot("CRM_All")
    plt.show()
# END OF COMPOSITE PLOTS
# ---------- END OF PLOTTING -----------------


def full_CRM_time_spectrum(t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, params_selected, time_selected, S_eigen, N_state, N_lat, N_period, N_Ham):
    # DERIVED PARAMETERS
    S_time = x2index((0., 1.), N_period, time_selected)

    period_arr = np.linspace(0., T_pump, N_period)
    # END OF DERIVED PARAMETERS

    # CALCULATION
    E, psi_eigen = calc_CRM_time_spectrum(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, N_state, N_lat, N_period, N_Ham)
    
    abs2_psi_eigen = abs2(psi_eigen)
    # END OF CALCULATION

    # CALCULATION OF ADDITIONAL PARAMETERS
    Delta_points = create_Delta_points(epsilon, t_0alpha, params_selected)

    t_pm_diff, epsilon_diff, t_0alpha_diff = calc_CRM_parameter_diffs(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, modulate_parameters, N_period)
    # END OF CALCULATION OF ADDITIONAL PARAMETERS

    # PLOTTING
    plotN_CRM_spectrum_eigen(E, abs2_psi_eigen, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, Delta_points, S_time, S_eigen, params_selected)
    # END OF PLOTTING


def full_CRM_time_evolution(psi_eigen, initial_condition, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, sigma_Gaussian, S_eigen, N_state, N_lat, N_period, N_time, addFirstOrder, W_W_overlap_arr, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar):
    # DERIVED PARAMETERS
    N_Ham = 2 * N_lat  # 2 Bloch bands (s and p) and N_lat lattice indices.

    period_arr = np.linspace(0., T_pump, N_period)

    time_arr = np.linspace(*time_span, N_time)
    # END OF DERIVED PARAMETERS

    # INITIAL CONDITION
    psi_initial = create_centered_single(N_lat, N_Ham)
    # psi_initial = create_centered_Gaussian(sigma_Gaussian, N_lat)

    E, psi_eigen = calc_CRM_time_spectrum(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, N_state, N_lat, N_period, N_Ham)
    # psi_initial = create_centered_eigen(psi_eigen, N_lat, N_Ham)
    # psi_initial = create_selected_eigen(psi_eigen, S_eigen)
    # END OF INITIAL CONDITION

    # TIME EVOLUTION
    psi = time_evolution_CRM(psi_initial, time_arr, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, N_state, N_lat, N_Ham, N_time)
    # END OF TIME EVOLUTION

    # CALCULATION OF WAVEFUNCTION PARAMETERS
    abs2_psi = abs2(psi)
    psi_COM = calc_psi_COM(abs2_psi, N_state, N_lat)
    # END OF CALCULATION OF WAVEFUNCTION PARAMETERS

    # PLOTTING
    # plot_psi_pcolormesh(abs2_psi, time_arr, N_Ham)
    # plot_psi_COM(psi_COM, time_arr)
    plotN_CRM_psi(abs2_psi, psi_COM, time_arr, N_Ham)
    # END OF PLOTTING


def full_CRM_all(initial_condition, sigma_Gaussian, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, params_selected, time_selected, S_eigen, N_state, N_lat, N_period, N_time, N_Ham, addFirstOrder, W_W_overlap_arr, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar):
    # DERIVED PARAMETERS
    S_time = x2index((0., 1.), N_period, time_selected)

    period_arr = np.linspace(0., T_pump, N_period)

    time_arr = np.linspace(*time_span, N_time)
    # END OF DERIVED PARAMETERS

    # *************** TIME SPECTRUM ***************
    # CALCULATION
    E, psi_eigen = calc_CRM_time_spectrum(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, N_state, N_lat, N_period, N_Ham)
    
    abs2_psi_eigen = abs2(psi_eigen)
    # END OF CALCULATION

    # CALCULATION OF ADDITIONAL PARAMETERS
    Delta_points = create_Delta_points(epsilon, t_0alpha, params_selected)

    t_pm_diff, epsilon_diff, t_0alpha_diff = calc_CRM_parameter_diffs(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, modulate_parameters, N_period)
    # END OF CALCULATION OF ADDITIONAL PARAMETERS
    # *************** END OF TIME SPECTRUM ***************
    

    # *************** TIME EVOLUTION ***************
    # INITIAL CONDITION
    psi_initial = create_centered_single(N_lat, N_Ham)
    # psi_initial = create_centered_Gaussian(sigma_Gaussian, N_lat)

    E, psi_eigen = calc_CRM_time_spectrum(period_arr, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, N_state, N_lat, N_period, N_Ham)
    # psi_initial = create_centered_eigen(psi_eigen, N_lat, N_Ham)
    # psi_initial = create_selected_eigen(psi_eigen, S_eigen)
    # END OF INITIAL CONDITION

    # TIME EVOLUTION
    psi = time_evolution_CRM(psi_initial, time_arr, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, N_state, N_lat, N_Ham, N_time)
    # END OF TIME EVOLUTION

    # CALCULATION OF WAVEFUNCTION PARAMETERS
    abs2_psi = abs2(psi)
    psi_COM = calc_psi_COM(abs2_psi, N_state, N_lat)
    # END OF CALCULATION OF WAVEFUNCTION PARAMETERS
    # *************** END OF TIME EVOLUTION ***************

    # PLOTTING
    plotN_CRM_all(E, abs2_psi_eigen, abs2_psi, psi_COM, t_pm_diff, epsilon_diff, t_0alpha_diff, period_arr, time_arr, Delta_points, S_time, S_eigen, params_selected, N_Ham)
    # END OF PLOTTING


def main():
    # PARAMETERS
    N_state = 3

    # STATIC SYSTEM PARAMETERS
    epsilon = np.array([0.00, 0.00])
    t_pm = np.array([1.00, 1.00])
    t_0alpha = np.array([0.10, 0.00])
    tn_alpha = np.array([0.15, -0.05])

    gamma = np.zeros(3, dtype=np.float64)
    gamma[-1] = 0.  # np.pi / 2.
    gamma[0] = -np.pi / 2.  # 0.
    gamma[1] = gamma[-1]
    # END OF STATIC SYSTEM PARAMETERS

    # SYSTEM MODULATION PARAMETERS
    t_bar = np.array([1.00, 0.85])
    epsilon_bar = np.array([0.10, -0.40])
    t_0_bar = np.array([0.15, 0.15])
    # epsilon_bar = np.array([0.10, 0.45])  # 0.05, 0.25, 0.45
    # t_0_bar = np.array([0.15, 0.15])

    # [t_pm, epsilon, t_0alpha]
    # t_pm should always be modulated, alongside epsilon or t_0alpha or both.
    modulate_parameters = np.array([True, True, False])
    # modulate_parameters = np.array([True, False, True])
    # END OF SYSTEM MODULATION PARAMETERS

    # 2 -> [x, y]; [t_pm, epsilon, t_0alpha]
    params_selected = np.array([0, 1])  # Should agree with modulate_parameters.

    N_lat = 100  # Number of lattice sites.

    N_period = 100

    N_time = 20
    time_span = (0., 20.)

    N_Ham = 2 * N_lat

    # TIME SPECTRUM PARAMETERS
    time_selected = 0.6  # Must be between 0 and 1 (in the first period).
    # S_eigen = N_Ham // 2 - 1
    S_eigen = 0
    # END OF TIME SPECTRUM PARAMETERS

    # INITIAL CONDITION PARAMETERS
    sigma_Gaussian = 5
    # END OF INITIAL CONDITION PARAMETERS
    # END OF PARAMETERS

    Chern_number = calc_CRM_Chern_numbers(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, (-np.pi, np.pi), (-0.1, 0.9), modulate_parameters, 5000, N_time)
    print("Chern number:", Chern_number)
    # full_CRM_time_evolution(psi_eigen, initial_condition, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, sigma_Gaussian, S_eigen, N_state, N_lat, N_period, N_time, addFirstOrder, W_W_overlap_arr, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar)
    # full_CRM_time_spectrum(t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, params_selected, time_selected, S_eigen, N_state, N_lat, N_period, N_Ham)
    full_CRM_all(initial_condition, sigma_Gaussian, time_span, t_bar, epsilon_bar, t_0_bar, t_pm, epsilon, t_0alpha, tn_alpha, gamma, modulate_parameters, params_selected, time_selected, S_eigen, N_state, N_lat, N_period, N_time, N_Ham, addFirstOrder, W_W_overlap_arr, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar)

    save_parameter_TXT(r"C=%i" % Chern_number)


if __name__ == "__main__":
    main()
