from tqdm import tqdm  # Progress bar library
from datetime import datetime
from math import pi
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pathlib
# plt.style.use("../matplotlibrc")


from QOPack.Band import *
from QOPack.Utility import *
from QOPack.Math import *
from QOPack.Wannier import full_calc_Wannier
from QOPack.RamanTimeDepDetuning.ReverseMapping import CRM2Full


# --------------- GENERAL CALCULATIONS ---------------
def calc_general_Berry_phase(ket_eigen, prod_axis, component_axis=-1):
    """For Zak_phase, prod_axis should be k_axis.\n
    prod_axis - axis over which the product shall be taken.\n
    component_axis - axis over which inner product is performed.\n
    The Berry phase array shape will be two dimensions less than the ket_eigen
    shape, namely, the Berry phase array will no longer have the prod_axis and
    component_axis.\n
    If not specified, assumes component axis is the last one.\n
    Based on presentation "Wannier functions, Modern theory of polarization".\n
    Found in https://theorie.physik.uni-konstanz.de/burkard/sites/default/files/ts16/Wannier-talk.pdf"""

    # Using Berry phase formula in slide 42/51.
    U_mu = np.conj(ket_eigen) * np.roll(ket_eigen, -1, axis=prod_axis)
    U_mu = np.sum(U_mu, axis=component_axis)
    # NOTE: If division by zero occurs, one should change the discretization.
    # U_mu = U_mu / np.abs(U_mu)  # Does not seem to affect the result (tested by looking at various plots).

    prod_U = np.prod(U_mu, axis=prod_axis)
    # print(prod_U)
    # check_if_imag(prod_U, r'prod_U')
    Berry_phase = -np.imag(np.log(prod_U))

    return Berry_phase
# --------------- END OF GENERAL CALCULATIONS ---------------


# @njit(cache=True)
# TODO: Still not perfect, but good enough.
def smooth_Berry_phase(Berry_phase):
    """Smooths out (removes discontinuities) Berry phase array. Useful if Berry phase will be plotted.

    Assumes that Berry phase array is 2D and that its shape is (N_band, N_lat||N_k).
    Shift from [-pi, pi) to [0, 2pi)."""
    atol = 1  # Should not be too small to make sure discontinuities are corrected, but should definitely be smaller than pi.

    for i in range(Berry_phase.shape[0]):
        if np.any(np.abs(Berry_phase[i] - np.roll(Berry_phase[i], -1))[2:-2] > atol):
            for j in range(Berry_phase.shape[1]):
                if Berry_phase[i, j] < 0.:
                    Berry_phase[i, j] += 2*np.pi
        else:
            Berry_phase[i] += pi

        # Edge cases
        if np.isclose(Berry_phase[i, 0], 0, atol=atol) and np.isclose(Berry_phase[i, 1], 2*pi, atol=atol):
            Berry_phase[i, 0] += 2*pi
        if np.isclose(Berry_phase[i, 0], 2*pi, atol=atol) and np.isclose(Berry_phase[i, 1], 0, atol=atol):
            Berry_phase[i, 0] -= 2*pi
        if np.isclose(Berry_phase[i, -1], 0, atol=atol) and np.isclose(Berry_phase[i, -2], 2*pi, atol=atol):
            Berry_phase[i, -1] += 2*pi
        if np.isclose(Berry_phase[i, -1], 2*pi, atol=atol) and np.isclose(Berry_phase[i, -2], 0, atol=atol):
            Berry_phase[i, -1] -= 2*pi

    return Berry_phase


def create_param_arrs(param_diff_span, param_avg, N_param):
    param_diff_arr = np.linspace(param_diff_span[0], param_diff_span[1], N_param)
    param_arr = param_avg + 0.5 * np.array([-1., 1.])[:, np.newaxis] * param_diff_arr[np.newaxis, :]
    return param_arr, param_diff_arr


def exact_Omega_k(k_arr, t_pm, t_0alpha, gamma, N_k):
    # (2, N_k); 2 -> [0, 1]; N_k -> [-pi/a_0, ..., pi/a_0]
    Omega = np.zeros((2, N_k), dtype=np.complex128)
    Omega[0] = 2. * (t_0alpha[0] - t_0alpha[1]) * np.cos(k_arr + gamma[0])
    Omega[1] = t_pm[0] * np.exp(-1.j * (k_arr + gamma[-1])) + \
               t_pm[1] * np.exp(1.j * (k_arr + gamma[1]))

    return Omega


def exact_Delta_k(Omega, k_arr, tn_alpha, epsilon, N_state):
    # N_k; N_k -> [-pi/a_0, ..., pi/a_0]
    return epsilon[0] - epsilon[1] + 2. * (tn_alpha[0] - tn_alpha[1]) * np.cos(N_state * k_arr) + Omega[0]


def exact_Lambda_k(k_arr, epsilon, t_0alpha, tn_alpha, gamma, N_state):
    # (2, N_k); 2 -> [s, p]; N_k -> [-pi/a_0, ..., pi/a_0]
    epsilon_term = epsilon[:, np.newaxis]
    natural_term = 2. * tn_alpha[:, np.newaxis] * np.cos(N_state * k_arr)[np.newaxis, :]
    interchain_term = 2. * t_0alpha[:, np.newaxis] * np.cos(k_arr + gamma[0])[np.newaxis, :]
    # UNUSED ALTERNATIVE (all other instances of k_arr and k_arr bounds would need to be changed):
    # natural_term = 2. * tn_alpha[:, np.newaxis] * np.cos(k_arr)[np.newaxis, :]
    # interchain_term = 2. * t_0alpha[:, np.newaxis] * np.cos(k_arr / N_state + gamma[0])[np.newaxis, :]

    return epsilon_term + natural_term + interchain_term


def exact_E_k(Delta, Lambda, Omega, N_k, withIdentity=True):
    # (2, N_k); 2 -> [-, +]; N_k -> [-pi/a_0, ..., pi/a_0]
    E_k = np.zeros((2, N_k), dtype=np.float64)
    Lambda_term = 0.5 * (Lambda[0] + Lambda[1])
    sq_term = 0.5 * np.sqrt(Delta**2 + 4.0 * np.abs(Omega[1])**2)
    check_if_real(sq_term, "E_k")
    if withIdentity:
        E_k[0] = Lambda_term - np.real(sq_term)
        E_k[1] = Lambda_term + np.real(sq_term)
    else:
        E_k[0] = -np.real(sq_term)
        E_k[1] = np.real(sq_term)

    return E_k


def exact_N_eigvec_k(E_k, Lambda, Delta, Omega, withIdentity=True):
    # (2, N_k); 2 -> [-, +]; N_k -> [-pi/a_0, ..., pi/a_0]
    Omega_term = np.abs(Omega[1])**2
    if withIdentity:
        energy_term = (E_k - (Lambda[0])[np.newaxis, :])**2
    else:
        energy_term = (E_k - 0.5 * Delta[np.newaxis, :])**2

    return 1.0 / np.sqrt(Omega_term[np.newaxis, :] + energy_term)
    # return 1.0 / np.hypot(np.abs(Omega[1]), (E_k - (Lambda[0])[np.newaxis, :]))


def exact_k_eigvec_k(E_k, Lambda, Delta, N_eigvec, Omega, N_k, withIdentity=True):
    # (2, N_k, 2); 2 -> [-, +]; N_k -> [-pi/a_0, ..., pi/a_0]; 2 -> [x_component, y_component]
    k_eigvec = np.zeros((2, N_k, 2), dtype=np.complex128)
    k_eigvec[0, :, 0] = N_eigvec[0] * np.conj(Omega[1])
    k_eigvec[1, :, 0] = N_eigvec[1] * np.conj(Omega[1])
    if withIdentity:
        k_eigvec[0, :, 1] = N_eigvec[0] * (E_k[0] - Lambda[0])
        k_eigvec[1, :, 1] = N_eigvec[1] * (E_k[1] - Lambda[0])
    else:
        k_eigvec[0, :, 1] = N_eigvec[0] * (E_k[0] - 0.5 * Delta)
        k_eigvec[1, :, 1] = N_eigvec[1] * (E_k[1] - 0.5 * Delta)
    # k_eigvec[:, 0] = np.einsum("ij,j->ij", N_eigvec, np.conj(Omega[1]))

    check_ket_orthonorm(k_eigvec, "k_eigvec")

    return k_eigvec


def full_exact_E_k(epsilon, t_pm, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k, withIdentity=True):
    # (2, N_k); 2 -> [-, +]; N_k -> [-pi/a_0, ..., pi/a_0]
    k_arr = np.linspace(k_span[0], k_span[1], N_k)
    Omega = exact_Omega_k(k_arr, t_pm, t_0alpha, gamma, N_k)
    Delta = exact_Delta_k(Omega, k_arr, tn_alpha, epsilon, N_state)
    Lambda = exact_Lambda_k(k_arr, epsilon, t_0alpha, tn_alpha, gamma, N_state)
    E_k = exact_E_k(Delta, Lambda, Omega, N_k, withIdentity=withIdentity)

    return E_k


def full_exact_k_eigvec_k(epsilon, t_pm, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k, withIdentity=True):
    # (2, N_k, 2); 2 -> [-, +]; N_k -> [-pi/a_0, ..., pi/a_0]; 2 -> [x_component, y_component]
    k_arr = np.linspace(k_span[0], k_span[1], N_k)
    Omega = exact_Omega_k(k_arr, t_pm, t_0alpha, gamma, N_k)
    Delta = exact_Delta_k(Omega, k_arr, tn_alpha, epsilon, N_state)
    Lambda = exact_Lambda_k(k_arr, epsilon, t_0alpha, tn_alpha, gamma, N_state)
    E_k = exact_E_k(Delta, Lambda, Omega, N_k, withIdentity=withIdentity)
    N_eigvec = exact_N_eigvec_k(E_k, Lambda, Delta, Omega, withIdentity=withIdentity)
    k_eigvec = exact_k_eigvec_k(E_k, Lambda, Delta, N_eigvec, Omega, N_k, withIdentity=withIdentity)

    return k_eigvec


@njit(cache=True)
def calc_Wannier_pm(k_eigvec, k_arr, N_Ham):
    Wannier_pm = np.zeros((2, N_Ham), dtype=np.complex128)

    N_k = len(k_arr)
    dk = (k_arr[-1] - k_arr[0]) / (N_k - 1)

    a = 1.

    # TODO: Can probably be done with IFFT.
    for s in range(2):  # 2 -> [-, +]
        for k in range(N_k):
            for r in range(N_k):  # N_lat = N_k
                for alpha in range(2):
                    Wannier_pm[s, 2*r+alpha] += np.exp(1j*k_arr[k]*(r-N_k//2)*a) * k_eigvec[s, k, alpha]
    Wannier_pm *= dk

    norm = np.sum(abs2(Wannier_pm), axis=-1)  # (2,)
    Wannier_pm = Wannier_pm / np.expand_dims(np.sqrt(norm), axis=1)

    return Wannier_pm


def exact_Wannier_pm(epsilon, t_pm, t_0alpha, tn_alpha, gamma, N_state, N_k, withIdentity=True):
    # (2, N_Ham), where N_Ham = 2*N_lat;
    N_Ham = 2*N_k  # N_lat = N_k

    k_span = (-pi, pi)
    # k_arr = np.linspace(k_span[0], k_span[1], N_k)
    k_arr = np.linspace(k_span[0], k_span[1], N_k, endpoint=False)

    k_eigvec = full_exact_k_eigvec_k(epsilon, t_pm, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k, withIdentity)

    Wannier_pm = calc_Wannier_pm(k_eigvec, k_arr, N_Ham)

    return Wannier_pm


def exact_A_k(t_pm, Delta, Omega, N_state):
    # N_k; N_k -> [-pi/a_0, ..., pi/a_0]
    a = 1. / N_state
    abs2_Omega_1 = abs2(Omega[1])
    cos_theta = Delta / np.sqrt(Delta**2 + 4 * abs2_Omega_1)
    t_term = (t_pm[1]**2 - t_pm[0]**2) / abs2_Omega_1
    # print(abs2_Omega_1[10000//2-5:10000//2+5])
    # print(t_term[10000//2-5:10000//2+5])

    A_k = 0.5*a * (1 - cos_theta) * t_term
    check_if_real(A_k, "A_k")
    A_k = np.real(A_k)

    return A_k


def full_exact_A_k(epsilon, t_pm, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k):
    """Calculates the momentum component of the Berry connection A_k."""
    # N_k; N_k -> [-pi/a_0, ..., pi/a_0]
    k_arr = np.linspace(k_span[0], k_span[1], N_k)
    Omega = exact_Omega_k(k_arr, t_pm, t_0alpha, gamma, N_k)
    Delta = exact_Delta_k(Omega, k_arr, tn_alpha, epsilon, N_state)
    A_k = exact_A_k(t_pm, Delta, Omega, N_state)

    return A_k


# ----------------- PLOTTING -----------------
def create_Delta_points(epsilon, t_0alpha, select_params):
    if select_params[0] == 1:
        t_0 = t_0alpha[0] - t_0alpha[1]
        return np.array([[-2. * t_0, 0.], [2. * t_0, 0.]])
    if select_params[1] == 1:
        t_0 = t_0alpha[0] - t_0alpha[1]
        return np.array([[0., -2. * t_0], [0., 2. * t_0]])
    if select_params[0] == 2:
        epsilon_diff = epsilon[0] - epsilon[1]
        return np.array([[-0.5 * epsilon_diff, 0.], [0.5 * epsilon_diff, 0.]])
    if select_params[1] == 2:
        epsilon_diff = epsilon[0] - epsilon[1]
        return np.array([[0., -0.5 * epsilon_diff], [0., 0.5 * epsilon_diff]])


def set_selected_param(param_arr, select_param, t_pm, epsilon, t_0alpha, i):
    if select_param == 0:
        t_pm[:] = param_arr[:, i]
    if select_param == 1:
        epsilon[:] = param_arr[:, i]
    if select_param == 2:
        t_0alpha[:] = param_arr[:, i]


def set_selected_axis_label(select_param, axis):
    """axis argument should be either "x" or "y"."""
    if axis == "x":
        plt_label = plt.xlabel
    if axis == "y":
        plt_label = plt.ylabel
    if select_param == 0:
        plt_label(r"$t_1 - t_{-1}$ / $E_R$")
    if select_param == 1:
        plt_label(r"$\bar{ϵ}^{(s)} - \bar{ϵ}^{(p)}$ / $E_R$")
    if select_param == 2:
        plt_label(r"$t_{0s} - t_{0p}$ / $E_R$")


def plot_energy_gaps(gap_arr, plot0_arr, plot1_arr, select_params, Delta_points, gap_string, savePlot=True, save_path=None, showPlot=True):
    """gap_string should either be "direct" or "indirect"."""
    fig = plt.figure(r"Gap arr", figsize=(6, 7))
    ax = plt.axes()

    set_selected_axis_label(select_params[0], "x")
    set_selected_axis_label(select_params[1], "y")

    # ax.minorticks_on()
    # plt.tick_params(axis="both", which="both", direction="in")
    # ax.xaxis.set_ticks_position("both")
    # ax.yaxis.set_ticks_position("both")

    # plt.pcolormesh(plot0_arr, plot1_arr, np.transpose(gap_arr), cmap=plt.get_cmap("viridis"), shading="nearest")
    if gap_arr.min() > 0.:  # If energy gaps are negative, logarithmic scale cannot be used.
        plt.pcolormesh(plot0_arr, plot1_arr, np.transpose(gap_arr), cmap=plt.get_cmap("viridis"), shading="nearest",
                    norm=mpl.colors.LogNorm(vmin=gap_arr.min(), vmax=gap_arr.max()))
    else:
        plt.pcolormesh(plot0_arr, plot1_arr, np.transpose(gap_arr), cmap=plt.get_cmap("viridis"), shading="nearest")

    if gap_string.lower() == "direct":
        cbar = plt.colorbar(label=r"Direct band gap $\mathrm{min}(E_{i+1} - E_i)$", location="top", aspect=30)
    if gap_string.lower() == "indirect":
        cbar = plt.colorbar(label=r"Indirect band gap $\mathrm{min}(E_{i+1}) - \mathrm{max}(E_i)$", location="top", aspect=30)

    cbar.ax.xaxis.label.set_size(12)
    cbar.ax.tick_params(which="both", length=2, direction="in")
    cbar.ax.xaxis.set_ticks_position("both")

    # plt.plot([Delta_points[:, 0]], [Delta_points[:, 1]], marker="o", markersize=5, markeredgecolor="black", markerfacecolor="black")
    # plt.tight_layout()
    set_plot_defaults(fig, ax, addGrid=False)

    if savePlot:
        save_plot("%s_Gap" % gap_string.capitalize(), save_path=save_path)

    if showPlot:
        plt.show()
    else:
        plt.clf()


def plot_Zak_arr(Zak_phase, plot0_arr, plot1_arr, select_params, Delta_points, savePlot=True, save_path=None, showPlot=True):
    """If save_path=None, the graph will be saved in the default directory, namely, CURRENT_DIRECTORY_OF_MAIN_SCRIPT/Graphs/CURRENT_DATE.
    If save_path is specified, the file will be saved in the save_path (save_path must be an absolute path).
    """

    # plt.rcParams["mathtext.fontset"] = "cm"
    # plt.rcParams["text.usetex"] = True
    # X, Y = np.meshgrid(epsilon_diff_arr, t_diff_arr)
    fig = plt.figure(r"Zak phase", figsize=(6, 7))
    ax = plt.axes()

    set_selected_axis_label(select_params[0], "x")
    set_selected_axis_label(select_params[1], "y")

    plt.pcolormesh(plot0_arr, plot1_arr, np.transpose(Zak_phase), cmap=plt.get_cmap("bwr"), shading="nearest")
    # plt.pcolormesh(epsilon_diff_arr, t_diff_arr, np.transpose(np.real(Zak_phase))[:-1, :-1], cmap=plt.get_cmap("bwr"), shading="flat")
    # with mpl.rc_context({'axes.labelsize': 12}):  # Temporary settings
    cbar = plt.colorbar(label=r"Zak phase $\gamma_{\mathrm{Zak}}$", location="top", aspect=30)
    cbar.ax.xaxis.label.set_size(12)
    cbar.ax.tick_params(which="both", length=2, direction="in")
    cbar.ax.xaxis.set_ticks_position("both")

    # if not (select_params[0] == 1 and select_params[1] == 2):
    plt.plot([Delta_points[:, 0]], [Delta_points[:, 1]], marker="o", markersize=5, markeredgecolor="black", markerfacecolor="black")

    set_plot_defaults(fig, ax, addGrid=False)

    if savePlot:
        # save_plot("Zak_Phase", file_type="pdf", save_path=save_path)
        save_plot("Zak_Phase", save_path=save_path)
    if showPlot:
        plt.show()
    else:
        plt.clf()


def plot_Zak_arr_v2(Zak_phase, plot0_arr, plot1_arr, select_params, Delta_points, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax)

    if select_params[0] == 0:
        ax.set_xlabel(r"$t_1 - t_{-1}$ / $E_R$")
    if select_params[0] == 1:
        ax.set_xlabel(r"$\bar{ϵ}^{(s)} - \bar{ϵ}^{(p)}$ / $E_R$")
    if select_params[0] == 2:
        ax.set_xlabel(r"$t_{0s} - t_{0p}$ / $E_R$")
    if select_params[1] == 0:
        ax.set_ylabel(r"$t_1 - t_{-1}$ / $E_R$")
    if select_params[1] == 1:
        ax.set_ylabel(r"$\bar{ϵ}^{(s)} - \bar{ϵ}^{(p)}$ / $E_R$")
    if select_params[1] == 2:
        ax.set_ylabel(r"$t_{0s} - t_{0p}$ / $E_R$")

    ax.pcolormesh(plot0_arr, plot1_arr, np.transpose(Zak_phase), cmap=plt.get_cmap("bwr"), shading="nearest")
    cbar = ax.colorbar(label=r"Zak phase $\gamma_{\mathrm{Zak}}$", location="top", aspect=30)
    cbar.ax.xaxis.label.set_size(12)
    cbar.ax.tick_params(which="both", length=2, direction="in")
    cbar.ax.xaxis.set_ticks_position("both")

    # if not (select_params[0] == 1 and select_params[1] == 2):
    ax.plot([Delta_points[:, 0]], [Delta_points[:, 1]], marker="o", markersize=5, markeredgecolor="black", markerfacecolor="black")

    if fig_ax is None:
        set_plot_defaults(fig, ax, addGrid=False)
        save_plot("Zak_Phase")
        plt.show()


def plot_A_k(k_arr, A_k):
    fig = plt.figure(r"Berry connection", figsize=(6, 5))
    ax = plt.axes()

    plt.title("Berry connection $A_k$")
    plt.xlabel("k / $k_0$")
    plt.ylabel("$A_k$ / $a_0$")
    plt.plot(k_arr, A_k, color="black")

    set_plot_defaults(fig, ax)
    save_plot("A_k")
    plt.show()
# ----------------- END OF PLOTTING -----------------


def calc_energy_gaps(x_param_arr, y_param_arr, t_pm, epsilon, t_0alpha, tn_alpha, gamma, select_params, N_state, N_k, N_x, N_y, gap_string, withIdentity=True):
    energy_gaps = np.zeros((N_x, N_y), dtype=np.float64)
    for i in range(N_x):
        set_selected_param(x_param_arr, select_params[0], t_pm, epsilon, t_0alpha, i)
        for j in range(N_y):
            set_selected_param(y_param_arr, select_params[1], t_pm, epsilon, t_0alpha, j)

            E_k = full_exact_E_k(epsilon, t_pm, t_0alpha, tn_alpha, gamma, (-pi, pi), N_state, N_k, withIdentity=withIdentity)

            if gap_string.lower() == "direct":
                energy_gaps[i, j] = calc_direct_gaps(E_k)[0]
            if gap_string.lower() == "indirect":
                energy_gaps[i, j] = calc_indirect_gaps(E_k)[0]

    return energy_gaps


def full_energy_gaps(x_param_diff_span, y_param_diff_span, x_param_avg, y_param_avg, t_pm, epsilon, t_0alpha, tn_alpha, gamma, select_params, N_state, N_k, N_x, N_y, gap_string, withIdentity=True, savePlot=True, save_path=None, showPlot=True):
    x_param_arr, x_param_diff_arr = create_param_arrs(x_param_diff_span, x_param_avg, N_x)
    y_param_arr, y_param_diff_arr = create_param_arrs(y_param_diff_span, y_param_avg, N_y)

    energy_gaps = calc_energy_gaps(x_param_arr, y_param_arr, t_pm, epsilon, t_0alpha, tn_alpha, gamma, select_params, N_state, N_k, N_x, N_y, gap_string,
                                   withIdentity=withIdentity)

    # Assumes:
    # 1) Not all 3 parameters are modulated;
    # 2) t_1 - t_{-1} axis is the y axis;
    if select_params[0] == 1 and gap_string == "Direct":
        save_array_NPY(energy_gaps, "Energy_gaps!epsilon!Direct!t_avg=%s" % y_param_avg)
    elif select_params[0] == 1 and gap_string == "Indirect":
        save_array_NPY(energy_gaps, "Energy_gaps!epsilon!Indirect!t_avg=%s" % y_param_avg)
    elif select_params[0] == 2 and gap_string == "Direct":
        save_array_NPY(energy_gaps, "Energy_gaps!t0alpha!Direct!t_avg=%s" % y_param_avg)
    elif select_params[0] == 2 and gap_string == "Indirect":
        save_array_NPY(energy_gaps, "Energy_gaps!t0alpha!Indirect!t_avg=%s" % y_param_avg)

    Delta_points = create_Delta_points(epsilon, t_0alpha, select_params)
    # Delta_points = []
    plot_energy_gaps(energy_gaps, x_param_diff_arr, y_param_diff_arr, select_params, Delta_points, gap_string,
                     savePlot=savePlot, save_path=save_path, showPlot=showPlot)


def full_Zak_phase(epsilon, t_pm, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k):
    k_eigvec = full_exact_k_eigvec_k(epsilon, t_pm, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k)
    Zak_phase = calc_general_Berry_phase(k_eigvec[0], prod_axis=0)

    return Zak_phase


def calc_Zak_arr(x_param_arr, y_param_arr, t_pm, epsilon, t_0alpha, tn_alpha, gamma, k_span, select_params, N_state, N_k, N_x, N_y, withIdentity=True):
    Zak_arr = np.zeros((N_x, N_y), dtype=np.float64)
    for i in range(N_x):
        set_selected_param(x_param_arr, select_params[0], t_pm, epsilon, t_0alpha, i)
        for j in range(N_y):
            set_selected_param(y_param_arr, select_params[1], t_pm, epsilon, t_0alpha, j)

            k_eigvec = full_exact_k_eigvec_k(epsilon, t_pm, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k, withIdentity=withIdentity)
            Zak_arr[i, j] = calc_general_Berry_phase(k_eigvec[0], prod_axis=0)

    return Zak_arr


# TODO: Does not work for (t_1+t_{-1})/2 = 0, figure out why.
def full_Zak_arr(x_param_diff_span, y_param_diff_span, x_param_avg, y_param_avg, t_pm, epsilon, t_0alpha, tn_alpha, gamma, k_span, select_params, N_state, N_k, N_x, N_y, withIdentity=True, savePlot=True, save_path=None, showPlot=True):
    x_param_arr, x_param_diff_arr = create_param_arrs(x_param_diff_span, x_param_avg, N_x)
    y_param_arr, y_param_diff_arr = create_param_arrs(y_param_diff_span, y_param_avg, N_y)

    Zak_arr = calc_Zak_arr(x_param_arr, y_param_arr, t_pm, epsilon, t_0alpha, tn_alpha, gamma, k_span, select_params, N_state, N_k, N_x, N_y, withIdentity)

    # Assumes:
    # 1) Not all 3 parameters are modulated;
    # 2) t_1 - t_{-1} axis is the y axis;
    if select_params[0] == 1:
        save_array_NPY(Zak_arr, "Zak_arr!epsilon!t_avg=%s" % y_param_avg)
    elif select_params[0] == 2:
        save_array_NPY(Zak_arr, "Zak_arr!t0alpha!t_avg=%s" % y_param_avg)

    Delta_points = create_Delta_points(epsilon, t_0alpha, select_params)
    plot_Zak_arr(Zak_arr, x_param_diff_arr, y_param_diff_arr, select_params, Delta_points, savePlot=savePlot, save_path=save_path, showPlot=showPlot)


def full_A_k(epsilon, t_pm, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k):
    """Calculates the momentum component of the Berry connection A_k."""
    k_arr = np.linspace(k_span[0], k_span[1], N_k)
    A_k = full_exact_A_k(epsilon, t_pm, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k)
    plot_A_k(k_arr, A_k)


def bulk_gap_arr(x_param_avg_bulk_span, y_param_avg_bulk_span, N_bulk1, N_bulk2, x_param_diff_span, y_param_diff_span, t_pm, epsilon, t_0alpha, tn_alpha, gamma, N_state, N_k, N_x, N_y, withIdentity):
    x_param_avg_arr = np.linspace(*x_param_avg_bulk_span, N_bulk1)
    y_param_avg_arr = np.linspace(*y_param_avg_bulk_span, N_bulk2)

    general_path = "%s/Bulk/%s/Energy_Gaps/" % (get_main_dir(), datetime.now().strftime("%Y-%m-%d"))
    save_parameter_TXT(save_path=general_path)
    with tqdm(total=N_bulk1*N_bulk2*2*3) as pbar:
        for x_param_avg in x_param_avg_arr:
            for y_param_avg in y_param_avg_arr:
                specific_path = "x_avg=%.2f!y_avg=%.2f" % (x_param_avg, y_param_avg)
                full_path = general_path + specific_path

                pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
                for gap_string in ['Direct', 'Indirect']:
                    for select_params in [np.array([0, 1]), np.array([0, 2]), np.array([1, 2])]:
                        full_energy_gaps(x_param_diff_span, y_param_diff_span, x_param_avg, y_param_avg, t_pm, epsilon, t_0alpha, tn_alpha, gamma, select_params, N_state, N_k, N_x, N_y, gap_string, withIdentity=withIdentity, savePlot=True, save_path=full_path, showPlot=False)

                        pbar.update(1)


def bulk_Zak_arr(x_param_avg_bulk_span, y_param_avg_bulk_span, N_bulk1, N_bulk2, x_param_diff_span, y_param_diff_span, t_pm, epsilon, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k, N_x, N_y, withIdentity):
    x_param_avg_arr = np.linspace(*x_param_avg_bulk_span, N_bulk1)
    y_param_avg_arr = np.linspace(*y_param_avg_bulk_span, N_bulk2)

    general_path = "%s/Bulk/%s/Zak_Phase/" % (get_main_dir(), datetime.now().strftime("%Y-%m-%d"))
    save_parameter_TXT(save_path=general_path)
    with tqdm(total=N_bulk1*N_bulk2*3) as pbar:
        for x_param_avg in x_param_avg_arr:
            for y_param_avg in y_param_avg_arr:
                specific_path = "x_avg=%.2f!y_avg=%.2f" % (x_param_avg, y_param_avg)
                full_path = general_path + specific_path

                pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
                for select_params in [np.array([0, 1]), np.array([0, 2]), np.array([1, 2])]:
                    full_Zak_arr(x_param_diff_span, y_param_diff_span, x_param_avg, y_param_avg, t_pm, epsilon, t_0alpha, tn_alpha, gamma, k_span, select_params, N_state, N_k, N_x, N_y, withIdentity=withIdentity, savePlot=True, save_path=full_path, showPlot=False)

                    pbar.update(1)
