from math import pi
import math
# import multiprocessing as mp
from multiprocessing.dummy import Pool
import time
import numpy as np
import numba
from numba import njit, prange
from scipy.special import hermite
import scipy.sparse
import scipy.sparse.linalg
from scipy.interpolate import splrep, splev
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.colors

from QOPack.Utility import fun_time, profile, save_plot, set_plot_defaults
from QOPack.Math import abs2, spl_Dn_y, translate_y
from QOPack.Wannier import change_Wannier_discretization, full_diag_cos_lattice, full_calc_Wannier, diag_cos_lattice
from QOPack.WannierV2 import *
from QOPack.Band import calc_indirect_gaps, calc_bandwidths


def spline_2D_arr(arr_2D, x_old_arr, x_new_arr):
    dim = np.shape(arr_2D)
    spl_arr_2D = np.zeros((np.shape(x_new_arr)[0], dim[1]), dtype=np.float64)

    for i in range(dim[1]):
        tck_arr = splrep(x_old_arr, arr_2D[:, i])
        spl_arr_2D[:, i] = splev(x_new_arr, tck_arr, der=0)

    return spl_arr_2D


def spline_4D_arr(arr_4D, x_old_arr, x_new_arr):
    dim = np.shape(arr_4D)
    spl_arr_4D = np.zeros((np.shape(x_new_arr)[0], dim[1], dim[2], dim[3]), dtype=np.float64)

    for i in range(dim[1]):
        for j in range(dim[2]):
            for k in range(dim[3]):
                tck_arr = splrep(x_old_arr, arr_4D[:, i, j, k])
                spl_arr_4D[:, i, j, k] = splev(x_new_arr, tck_arr, der=0)

    return spl_arr_4D


def create_chi_HO(x_arr, Omega, band_num, K, k_0):
    chi_HO = np.zeros((band_num, K), dtype=np.float64)
    k_x_arr = k_0 * x_arr
    for a in prange(band_num):
        H_alpha = hermite(a)
        chi_HO[a] = (-1.0)**a * np.exp(-Omega**0.5 * k_x_arr**2 / 2.0) * H_alpha(Omega**0.25 * k_x_arr)
        chi2_int = np.trapz(abs2(chi_HO[a]), x_arr)
        # chi2_int = np.trapz((np.abs(chi_HO[a]))**2, x_arr)
        # chi2_int = np.trapz(np.real(chi_HO[a])**2 + np.imag(chi_HO[a])**2, x_arr)
        chi_HO[a] /= np.sqrt(chi2_int)

    return chi_HO


@njit(cache=True, parallel=True, nogil=True)
def calc_overlap(psi, x_arr, x_span, single_shift, band_num, sepN_len):
    psi_overlap_arr = np.zeros((band_num, band_num, sepN_len), dtype=np.float64)
    for i in prange(band_num):
        for j in prange(band_num):
            for k in prange(sepN_len):
                psi_psi = psi[i] * translate_y(psi[j], x_span, k * single_shift)
                psi_overlap_arr[i, j, k] = np.trapz(psi_psi, x_arr)

    return psi_overlap_arr


@njit(cache=True, parallel=True, nogil=True)
def calc_interaction(psi, x_arr, x_span, single_shift, band_num, sepN_len):
    psi2 = abs2(psi)
    # psi2 = (np.abs(psi))**2
    # psi2 = np.real(psi)**2 + np.imag(psi)**2

    psi_interaction_arr = np.zeros((band_num, band_num, sepN_len), dtype=np.float64)
    for i in prange(band_num):
        for j in prange(band_num):
            for k in prange(sepN_len):
                psi2_psi2 = psi2[i] * translate_y(psi2[j], x_span, k * single_shift)
                psi_interaction_arr[i, j, k] = np.trapz(psi2_psi2, x_arr)

    return psi_interaction_arr


@njit(cache=True, parallel=True, nogil=True)
def calc_kinetic(psi, D2_psi, x_arr, x_span, k_0, band_num, sep_len):
    psi_T_psi_arr = np.zeros((band_num, band_num, sep_len), dtype=np.float64)

    for i in prange(band_num):
        for j in prange(band_num):
            for k in prange(sep_len):
                psi_T_psi = psi[i] * translate_y(D2_psi[j], x_span, k)
                psi_T_psi_arr[i, j, k] = np.trapz(psi_T_psi, x_arr)

    psi_T_psi_arr /= -k_0**2

    return psi_T_psi_arr


@njit(cache=True, parallel=True, nogil=True)
def calc_potential(psi, V_epsilon, x_arr, x_span, band_num, sep_len):
    psi_V_psi_arr = np.zeros((band_num, band_num, sep_len), dtype=np.float64)

    for i in prange(band_num):
        for j in prange(band_num):
            for k in prange(sep_len):
                psi_V_psi = psi[i] * V_epsilon * translate_y(psi[j], x_span, k)
                # print(np.shape(psi_V_psi))
                psi_V_psi_arr[i, j, k] = np.trapz(psi_V_psi, x_arr)

    return psi_V_psi_arr


@njit(cache=True, parallel=True, nogil=True)
def calc_psi_x_psi(psi, x_arr, x_span, band_num, sep_pm_len):
    psi_x_psi_arr = np.zeros((band_num, band_num, sep_pm_len), dtype=np.float64)

    for i in prange(band_num):
        for j in prange(band_num):
            for k in prange(sep_pm_len):
                psi_x_psi = psi[i] * x_arr * translate_y(psi[j], x_span, k - sep_pm_len // 2)
                psi_x_psi_arr[i, j, k] = np.trapz(psi_x_psi, x_arr)

    return psi_x_psi_arr


def exact_HO_overlap(Omega, N_state, band_num, sepN_len):
    HO_exact_overlap_arr = np.zeros((band_num, band_num, sepN_len), dtype=np.float64)
    # WARNING: prange here gives wrong results, figure out why, FIX LATER!!!
    for k in range(sepN_len):
        HO_exp = np.exp(-(k * np.pi / N_state)**2 * np.sqrt(Omega))
        # logging.info("%i: %.7f" % (k, HO_exp))
        HO_exact_overlap_arr[0, 0, k] = HO_exp
        HO_exact_overlap_arr[0, 1, k] = k * np.pi * np.sqrt(2.0) / N_state * Omega**0.25 * HO_exp
        HO_exact_overlap_arr[1, 0, k] = -HO_exact_overlap_arr[0, 1, k] 
        HO_exact_overlap_arr[1, 1, k] = (1.0 - 2.0 * (k * np.pi / N_state)**2 * np.sqrt(Omega)) * HO_exp
    # logging.info("************")

    return HO_exact_overlap_arr


def exact_HO_tunneling(Omega, band_num, sep_len):
    HO_exact_tunneling_arr = np.zeros((band_num, band_num, sep_len), dtype=np.float64)

    sq_Omega = np.sqrt(Omega)
    exp_4 = np.exp(-0.25 / sq_Omega)
    exp_pi = np.exp(-np.pi**2 * sq_Omega)
    HO_exact_tunneling_arr[0, 0, 0] = 0.5 * sq_Omega - 2.0 * Omega * exp_4
    HO_exact_tunneling_arr[0, 0, 1] = exp_pi * (0.5 * sq_Omega - np.pi**2 * Omega + 2.0 * Omega * exp_4)
    # Still need sp components
    HO_exact_tunneling_arr[1, 1, 0] = (1.5 + exp_4) * sq_Omega - 2.0 * Omega * exp_4
    HO_exact_tunneling_arr[1, 1, 1] = ((1.5 - exp_4) * sq_Omega - (6.0 * np.pi**2 - 2.0 * exp_4) * Omega + (2.0 * np.pi**4 - 4.0 * np.pi**2 * exp_4) * Omega * sq_Omega) * exp_pi

    return HO_exact_tunneling_arr


# @njit(cache=True, parallel=True, nogil=True)
def routine_calc_overlaps(W_R, x_arr, x_span, Omega, N_state, band_num, k_0, single_shift, sepN_len, K):
    W_W_overlap_arr = calc_overlap(W_R, x_arr, x_span, single_shift, band_num, sepN_len)

    HO_exact_overlap_arr = exact_HO_overlap(Omega, N_state, band_num, sepN_len)

    chi_HO = create_chi_HO(x_arr, Omega, band_num, K, k_0)
    HO_calc_overlap_arr = calc_overlap(chi_HO, x_arr, x_span, single_shift, band_num, sepN_len)

    W_W_alpha_s_arr = np.zeros(sepN_len, dtype=np.float64)
    for k in prange(sepN_len):
        W_W_alpha_s_arr[k] = W_W_overlap_arr[1, 1, k] / W_W_overlap_arr[0, 0, k]

    HO_exact_alpha_s_arr = np.zeros(sepN_len, dtype=np.float64)
    for k in prange(sepN_len):
        HO_exact_alpha_s_arr[k] = HO_exact_overlap_arr[1, 1, k] / HO_exact_overlap_arr[0, 0, k]

    HO_calc_alpha_s_arr = np.zeros(sepN_len, dtype=np.float64)
    for k in prange(sepN_len):
        HO_calc_alpha_s_arr[k] = HO_calc_overlap_arr[1, 1, k] / HO_calc_overlap_arr[0, 0, k]

    return W_W_overlap_arr, HO_exact_overlap_arr, HO_calc_overlap_arr, W_W_alpha_s_arr, HO_exact_alpha_s_arr, HO_calc_alpha_s_arr


# @njit(cache=True, parallel=True, nogil=True)
def set_tunnelings(W_R, D2_W_R, V_epsilon, x_arr, x_span, Omega, band_num, k_0, sep_len, K):
    ## Calculating Hamiltonian matrix elements from the dispersion relation E(k) is more accurate than calculating from Wannier functions.
    # W_T_W_arr = calc_kinetic(W_R, D2_W_R, x_arr, x_span, k_0, band_num, sep_len)
    # W_V_W_arr = calc_potential(W_R, V_epsilon, x_arr, x_span, band_num, sep_len)
    # W_H_W_arr = W_T_W_arr + W_V_W_arr
    W_H_W_arr = calc_spectrum_tunnelings(Omega, sep_len, band_num, N_k=251, N_Fourier=100)

    HO_exact_tunneling_arr = exact_HO_tunneling(Omega, band_num, sep_len)

    chi_HO = create_chi_HO(x_arr, Omega, band_num, K, k_0)
    # chi_HO, D2_chi_HO = change_Wannier_discretization(chi_HO, x_arr, x_arr, band_num, K)
    D2_chi_HO = np.zeros((band_num, K), dtype=np.float64)
    for a in range(band_num):
        D2_chi_HO[a] = spl_Dn_y(chi_HO[a], x_arr, x_arr, der=2)
    chi_T_chi_arr = calc_kinetic(chi_HO, D2_chi_HO, x_arr, x_span, k_0, band_num, sep_len)
    chi_V_chi_arr = calc_potential(chi_HO, V_epsilon, x_arr, x_span, band_num, sep_len)
    chi_H_chi_arr = chi_T_chi_arr + chi_V_chi_arr

    return W_H_W_arr, HO_exact_tunneling_arr, chi_H_chi_arr


def plot_Wannier_HO(W_R, chi_HO, x_arr, K, a_0, sen):
    fig = plt.figure("Wannier")
    ax = plt.axes()
    plt.title("Wannier")
    plt.xlabel("x / $a_0$")
    plt.ylabel("W / $a_0^{-0.5}$")
    plt.plot(x_arr / a_0, np.zeros(K), color="red", linestyle="--")
    plt.plot(x_arr / a_0, W_R[sen].real, color="black")
    plt.fill_between(x_arr / a_0, W_R[sen].real, W_R[sen].real + W_R[sen].imag, color="red")
    plt.plot(x_arr / a_0, chi_HO[sen], color="purple")
    set_plot_defaults(fig, ax)
    save_plot("Wannier_HO")
    plt.show()


def plot_W_W_overlap(W_W_overlap_arr, HO_exact_overlap_arr, HO_calc_overlap_arr, W_W_alpha_s_arr, HO_exact_alpha_s_arr, HO_calc_alpha_s_arr, Omega_arr, Omega_ext, colors):
    spl_W_W_overlap_arr = spline_4D_arr(W_W_overlap_arr, Omega_arr, Omega_ext)
    spl_HO_exact_overlap_arr = spline_4D_arr(HO_exact_overlap_arr, Omega_arr, Omega_ext)
    spl_HO_calc_overlap_arr = spline_4D_arr(HO_calc_overlap_arr, Omega_arr, Omega_ext)
    spl_W_W_alpha_s_arr = spline_2D_arr(W_W_alpha_s_arr, Omega_arr, Omega_ext)
    spl_HO_exact_alpha_s_arr = spline_2D_arr(HO_exact_alpha_s_arr, Omega_arr, Omega_ext)
    spl_HO_calc_alpha_s_arr = spline_2D_arr(HO_calc_alpha_s_arr, Omega_arr, Omega_ext)

    name_string = r"$ss$ NN overlap"
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel("$\Omega$ / $E_R$")
    plt.ylabel("$G_{0, 0}(a)$")
    plt.plot(Omega_ext, spl_W_W_overlap_arr[:, 0, 0, 1], color=colors[0])
    plt.plot(Omega_ext, spl_HO_exact_overlap_arr[:, 0, 0, 1], linestyle="dashed", color=colors[0])
    plt.plot(Omega_ext, spl_HO_calc_overlap_arr[:, 0, 0, 1], linestyle="dotted", color=colors[0])
    set_plot_defaults(fig, ax)
    save_plot(r"ss_NN_overlap")
    plt.show()

    name_string = r"$sp$ NN overlap"
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel("$\Omega$ / $E_R$")
    plt.ylabel("$G_{0, 1}(a)$")
    plt.plot(Omega_ext, spl_W_W_overlap_arr[:, 0, 1, 1], color=colors[1])
    plt.plot(Omega_ext, spl_HO_exact_overlap_arr[:, 0, 1, 1], linestyle="dashed", color=colors[1])
    plt.plot(Omega_ext, spl_HO_calc_overlap_arr[:, 0, 1, 1], linestyle="dotted", color=colors[1])
    set_plot_defaults(fig, ax)
    save_plot(name_string)
    plt.show()

    name_string = r"$pp$ NN overlap"
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel("$\Omega$ / $E_R$")
    plt.ylabel("$G_{1, 1}(a)$")
    plt.plot(Omega_ext, spl_W_W_overlap_arr[:, 1, 1, 1], color=colors[2])
    plt.plot(Omega_ext, spl_HO_exact_overlap_arr[:, 1, 1, 1], linestyle="dashed", color=colors[2])
    plt.plot(Omega_ext, spl_HO_calc_overlap_arr[:, 1, 1, 1], linestyle="dotted", color=colors[2])
    set_plot_defaults(fig, ax)
    save_plot(r"sp_NN_overlap")
    plt.show()

    name_string = r"$pp$ and $ss$ NN overlap ratio"
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel(r"$\Omega$ / $E_R$")
    plt.ylabel(r"$\alpha_s$")
    plt.plot(Omega_ext, spl_W_W_alpha_s_arr[:, 1], color=colors[3])
    plt.plot(Omega_ext, spl_HO_exact_alpha_s_arr[:, 1], linestyle="dashed", color=colors[3])
    plt.plot(Omega_ext, spl_HO_calc_alpha_s_arr[:, 1], linestyle="dotted", color=colors[3])
    set_plot_defaults(fig, ax)
    save_plot(r"pp_and_ss_NN_overlap_ratio")
    plt.show()


def plot_W_tunneling(W_H_W_arr, HO_exact_tunneling_arr, chi_H_chi_arr, Omega_arr, Omega_ext, colors):
    spl_W_H_W_arr = spline_4D_arr(W_H_W_arr, Omega_arr, Omega_ext)
    spl_HO_exact_tunneling_arr = spline_4D_arr(HO_exact_tunneling_arr, Omega_arr, Omega_ext)
    spl_chi_H_chi_arr = spline_4D_arr(chi_H_chi_arr, Omega_arr, Omega_ext)

    name_string = r"$ss$ On-site energy"
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel("$\Omega$ / $E_R$")
    plt.ylabel("$\epsilon^{(0)}$ / $E_R$")
    plt.plot(Omega_ext, spl_W_H_W_arr[:, 0, 0, 0], color=colors[0])
    plt.plot(Omega_ext, spl_chi_H_chi_arr[:, 0, 0, 0], linestyle="dashed", color=colors[0])
    plt.plot(Omega_ext, spl_HO_exact_tunneling_arr[:, 0, 0, 0], linestyle="dotted", color=colors[0])
    set_plot_defaults(fig, ax)
    save_plot(r"ss_On-site_energy")
    plt.show()

    name_string = r"$ss$ Natural tunneling"
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel("$\Omega$ / $E_R$")
    plt.ylabel("$t^{(0)}$ / $E_R$")
    plt.plot(Omega_ext, spl_W_H_W_arr[:, 0, 0, 1], color=colors[1])
    plt.plot(Omega_ext, spl_chi_H_chi_arr[:, 0, 0, 1], linestyle="dashed", color=colors[1])
    plt.plot(Omega_ext, spl_HO_exact_tunneling_arr[:, 0, 0, 1], linestyle="dotted", color=colors[1])
    set_plot_defaults(fig, ax)
    save_plot(r"ss_Natural_tunneling")
    plt.show()

    ## Non-diagonal matrix elements should be zero.
    ## TODO: Figure out why.
    # name_string = r"$sp$ Natural tunneling"
    # fig = plt.figure(name_string)
    # ax = plt.axes()
    # plt.title(name_string)
    # plt.xlabel("$\Omega$ / $E_R$")
    # plt.ylabel("$t^{(sp)}$ / $E_R$")
    # plt.plot(Omega_ext, spl_W_H_W_arr[:, 0, 1, 1], color=colors[1])
    # plt.plot(Omega_ext, spl_chi_H_chi_arr[:, 0, 1, 1], linestyle="dashed", color=colors[1])
    # plt.plot(Omega_ext, spl_HO_exact_tunneling_arr[:, 0, 1, 1], linestyle="dotted", color=colors[1])
    # set_plot_defaults(fig, ax)
    # save_plot(r"sp_Natural_tunneling")
    # plt.show()

    name_string = r"$pp$ On-site energy"
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel("$\Omega$ / $E_R$")
    plt.ylabel("$\epsilon^{(1)}$ / $E_R$")
    plt.plot(Omega_ext, spl_W_H_W_arr[:, 1, 1, 0], color=colors[2])
    plt.plot(Omega_ext, spl_chi_H_chi_arr[:, 1, 1, 0], linestyle="dashed", color=colors[2])
    plt.plot(Omega_ext, spl_HO_exact_tunneling_arr[:, 1, 1, 0], linestyle="dotted", color=colors[2])
    set_plot_defaults(fig, ax)
    save_plot(r"pp_On-site_energy")
    plt.show()

    name_string = r"$pp$ Natural tunneling"
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel("$\Omega$ / $E_R$")
    plt.ylabel("$t^{(1)}$ / $E_R$")
    plt.plot(Omega_ext, spl_W_H_W_arr[:, 1, 1, 1], color=colors[3])
    plt.plot(Omega_ext, spl_chi_H_chi_arr[:, 1, 1, 1], linestyle="dashed", color=colors[3])
    plt.plot(Omega_ext, spl_HO_exact_tunneling_arr[:, 1, 1, 1], linestyle="dotted", color=colors[3])
    set_plot_defaults(fig, ax)
    save_plot(r"pp_Natural_tunneling")
    plt.show()

    # Energy detuning also depends on detuning modulation frequency \omega.
    name_string = r"Energy detuning without omega"
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel("$\Omega$ / $E_R$")
    plt.ylabel(r"$ϵ^{(s)} - ϵ^{(p)}$ / $E_R$")
    plt.plot(Omega_ext, spl_W_H_W_arr[:, 0, 0, 0] - spl_W_H_W_arr[:, 1, 1, 0], color=colors[4])
    plt.plot(Omega_ext, spl_chi_H_chi_arr[:, 0, 0, 0] - spl_chi_H_chi_arr[:, 1, 1, 0], linestyle="dashed", color=colors[4])
    plt.plot(Omega_ext, spl_HO_exact_tunneling_arr[:, 0, 0, 0] - spl_HO_exact_tunneling_arr[:, 1, 1, 0], linestyle="dotted", color=colors[4])
    set_plot_defaults(fig, ax)
    save_plot(r"Energy_detuning_without_omega")
    plt.show()


def plot_band_flatness(Omega_arr, Omega_ext, band_flatness, colors):
    band_num = band_flatness.shape[1]

    spl_band_flatness = spline_2D_arr(band_flatness, Omega_arr, Omega_ext)
    name_string = r"Band flatness"
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel("$\Omega$ / $E_R$")
    plt.ylabel(r"Band flatness")
    [plt.plot(Omega_ext, spl_band_flatness[:, i], color=colors[i]) for i in range(band_num)]

    band_names = ['s', 'p', 'd', 'f', 'g', 'h']
    if band_num < 6:
        plt.legend(band_names[:band_num], loc='upper right',
                   prop=mpl.font_manager.FontProperties(size=10))

    set_plot_defaults(fig, ax)
    save_plot(r"Band_flatness")
    plt.show()


def plot_W_natural_tunnelings(W_H_W_arr, Omega_arr, Omega_ext, colors):
    # shape = (R, band_num, band_num, sep_len)
    spl_W_H_W_arr = spline_4D_arr(W_H_W_arr, Omega_arr, Omega_ext)

    name_string = r"$ss$ Natural tunneling"
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel("$\Omega$ / $E_R$")
    plt.ylabel("$t^{(s)}$ / $E_R$")
    plt.plot(Omega_ext, spl_W_H_W_arr[:, 0, 0, 1], color=colors[0])
    plt.plot(Omega_ext, spl_W_H_W_arr[:, 0, 0, 2], linestyle="dashed", color=colors[0])
    plt.plot(Omega_ext, spl_W_H_W_arr[:, 0, 0, 3], linestyle="dotted", color=colors[0])
    plt.legend([r'$a_0$', r'$2a_0$', r'$3a_0$'], loc='lower right',
                prop=mpl.font_manager.FontProperties(size=10))
    set_plot_defaults(fig, ax)
    save_plot(r"ss_Natural_tunneling")
    plt.show()

    name_string = r"$pp$ Natural tunneling"
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel("$\Omega$ / $E_R$")
    plt.ylabel("$t^{(p)}$ / $E_R$")
    plt.plot(Omega_ext, spl_W_H_W_arr[:, 1, 1, 1], color=colors[1])
    plt.plot(Omega_ext, spl_W_H_W_arr[:, 1, 1, 2], linestyle="dashed", color=colors[1])
    plt.plot(Omega_ext, spl_W_H_W_arr[:, 1, 1, 3], linestyle="dotted", color=colors[1])
    plt.legend([r'$a_0$', r'$2a_0$', r'$3a_0$'], loc='upper right',
                prop=mpl.font_manager.FontProperties(size=10))
    set_plot_defaults(fig, ax)
    save_plot(r"pp_Natural_tunneling")
    plt.show()


def plot_W_interactions(W_W_interaction_arr, Omega_arr, Omega_ext, N_state):
    spl_W_W_interaction_arr = spline_4D_arr(W_W_interaction_arr, Omega_arr, Omega_ext)

    name_string = r"On-site interaction, $N=%i$" % N_state
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel(r"$\Omega$/$E_\mathrm{R}$")
    plt.ylabel(r"$F_0$")
    plt.plot(Omega_ext, spl_W_W_interaction_arr[:, 0, 0, 0], color="green")
    set_plot_defaults(fig, ax)
    save_plot(r"On-site_Interaction")
    plt.show()

    # name_string = r"NN interaction, $N=%i$" % N_state
    name_string = r"Neighbour interaction, $N=%i$" % N_state
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel(r"$\Omega$/$E_\mathrm{R}$")
    # plt.ylabel(r"$F_1$")
    plt.ylabel("$F$")
    plt.plot(Omega_ext, spl_W_W_interaction_arr[:, 0, 0, 1], color="blue")
    plt.plot(Omega_ext, spl_W_W_interaction_arr[:, 0, 0, 2], color="blue", linestyle="dashed")
    plt.plot(Omega_ext, spl_W_W_interaction_arr[:, 0, 0, 3], color="blue", linestyle="dotted")
    plt.plot(Omega_ext, spl_W_W_interaction_arr[:, 0, 0, 4], color="blue", linestyle="dashdot")
    # plt.legend([r'NN', r'NNN'], loc='upper right')
    plt.legend([r'NN', r'NNN', r'4N', r'5N'], loc='upper right')
    set_plot_defaults(fig, ax)
    save_plot(r"Neighbour_Interaction")
    plt.show()

    name_string = r"Interaction ratio, $N=%i$" % N_state
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel(r"$\Omega$/$E_\mathrm{R}$")
    plt.ylabel(r"$F_0/F_1$")
    plt.plot(Omega_ext, spl_W_W_interaction_arr[:, 0, 0, 0] / spl_W_W_interaction_arr[:, 0, 0, 1], color="red")
    set_plot_defaults(fig, ax)
    save_plot(r"Interaction_Ratio")
    plt.show()


def plot_W_overlap_v2(W_W_overlap_arr, Omega_arr, Omega_ext, N_state):
    spl_W_W_overlap_arr = spline_4D_arr(W_W_overlap_arr, Omega_arr, Omega_ext)

    # name_string = r"Neighbour overlap, $N=%i$" % N_state
    # fig = plt.figure(name_string)
    # ax = plt.axes()
    # plt.title(name_string)
    # plt.xlabel(r"$\Omega$/$E_\mathrm{R}$")
    # plt.ylabel("$G$")
    # plt.plot(Omega_ext, spl_W_W_overlap_arr[:, 0, 0, 1], color="blue")
    # plt.plot(Omega_ext, spl_W_W_overlap_arr[:, 0, 0, 2], color="blue", linestyle="dashed")
    # plt.plot(Omega_ext, spl_W_W_overlap_arr[:, 0, 0, 3], color="blue", linestyle="dotted")
    # plt.plot(Omega_ext, spl_W_W_overlap_arr[:, 0, 0, 4], color="blue", linestyle="dashdot")
    # plt.legend([r'NN', r'NNN', r'4N', r'5N'], loc='upper right')
    # set_plot_defaults(fig, ax)
    # save_plot(r"Neighbour_Overlap")
    # plt.show()

    name_string = r"$sp$ NN overlap, $N=%i$" % N_state
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel(r"$\Omega$/$E_\mathrm{R}$")
    plt.ylabel("$G$")
    plt.plot(Omega_ext, spl_W_W_overlap_arr[:, 0, 1, 1], color="blue")
    # plt.plot(Omega_ext, spl_W_W_overlap_arr[:, 0, 0, 2], color="blue", linestyle="dashed")
    # plt.plot(Omega_ext, spl_W_W_overlap_arr[:, 0, 0, 3], color="blue", linestyle="dotted")
    # plt.plot(Omega_ext, spl_W_W_overlap_arr[:, 0, 0, 4], color="blue", linestyle="dashdot")
    # plt.legend([r'NN', r'NNN', r'4N', r'5N'], loc='upper right')
    set_plot_defaults(fig, ax)
    save_plot(r"sp_NN_overlap")
    plt.show()

    # name_string = r"NN and NNN overlap ratio, $N=%i$" % N_state
    # fig = plt.figure(name_string)
    # ax = plt.axes()
    # plt.title(name_string)
    # plt.xlabel(r"$\Omega$/$E_\mathrm{R}$")
    # plt.ylabel(r"$G_1/G_2$")
    # plt.plot(Omega_ext, spl_W_W_overlap_arr[:, 0, 0, 1] / spl_W_W_overlap_arr[:, 0, 0, 2], color="red")
    # set_plot_defaults(fig, ax)
    # save_plot(r"NN_and_NNN_Overlap_Ratio")
    # plt.show()


def plot_W_tunneling_v2(W_H_W_arr, Omega_arr, Omega_ext, N_state):
    spl_W_H_W_arr = spline_4D_arr(W_H_W_arr, Omega_arr, Omega_ext)

    name_string = r"Natural tunneling, $N=%i$" % N_state
    fig = plt.figure(name_string)
    ax = plt.axes()
    plt.title(name_string)
    plt.xlabel(r"$\Omega$/$E_\mathrm{R}$")
    plt.ylabel(r"$J_\mathrm{natural}$")
    plt.plot(Omega_ext, spl_W_H_W_arr[:, 0, 0, 1], color="purple")
    plt.plot(Omega_ext, spl_W_H_W_arr[:, 0, 0, 2], color="purple", linestyle="--")
    plt.legend([r'NN', r'NNN'], loc='upper right')
    set_plot_defaults(fig, ax)
    save_plot(r"Natural_Tunneling")
    plt.show()


def overlap_tunneling(N_state, Omega_span, a_0, K, R, band_num, max_sep, x_max):
    # start_time = time.perf_counter()

    # DERIVED PARAMETERS
    k_0 = 2.0 * np.pi / a_0

    x_min = -x_max

    sep_int = int(max_sep)
    sep_len = int(max_sep) + 1
    sepN_int = int(N_state * max_sep)
    sepN_len = int(N_state * max_sep) + 1

    K = 2 * (K // 2) + 1

    x_span = (x_min, x_max)
    x_arr = np.linspace(x_min, x_max, K)
    x_step = (x_max - x_min) / (K - 1)
    K_0 = K // 2  # Assumes x_min = -x_max.

    single_shift = 1. / N_state

    Km_a = math.floor(a_0 / x_step)
    Km_a2 = Km_a // 2
    Km_2a = 2 * Km_a
    Km_aN = Km_a // N_state
    Km_2aN = 2 * Km_a // N_state

    Omega_arr = np.linspace(Omega_span[0], Omega_span[1], R)

    V_epsilon = -2.0 * Omega_arr[:, np.newaxis] * np.cos(k_0 * x_arr[np.newaxis, :])
    # END OF DERIVED PARAMETERS

    W_W_overlap_arr = np.zeros((R, band_num, band_num, sepN_len), dtype=np.float64)
    HO_exact_overlap_arr = np.zeros((R, band_num, band_num, sepN_len), dtype=np.float64)
    HO_calc_overlap_arr = np.zeros((R, band_num, band_num, sepN_len), dtype=np.float64)
    W_W_alpha_s_arr = np.zeros((R, sepN_len), dtype=np.float64)
    HO_exact_alpha_s_arr = np.zeros((R, sepN_len), dtype=np.float64)
    HO_calc_alpha_s_arr = np.zeros((R, sepN_len), dtype=np.float64)

    W_H_W_arr = np.zeros((R, band_num, band_num, sep_len), dtype=np.float64)
    chi_H_chi_arr = np.zeros((R, band_num, band_num, sep_len), dtype=np.float64)
    HO_exact_tunneling_arr = np.zeros((R, band_num, band_num, sep_len), dtype=np.float64)

    for r in range(R):
        Omega = Omega_arr[r]

        # PART 1: GET band_num PRIMARY LATTICE WANNIER FUNCTIONS W_R
        # Make sure same Wannier function implementation is used everywhere to rule out inconsistencies.

        # W_R, D2_W_R = full_calc_Wannier(Omega, x_max, band_num, K)
        W_R, D2_W_R = full_calc_Wannier_Kohn(Omega, (-x_max, x_max), band_num, K, N_Fourier=50, useCached=False)
        # END OF PART 1

        # PART 2
        W_W_overlap_arr[r], HO_exact_overlap_arr[r], HO_calc_overlap_arr[r], W_W_alpha_s_arr[r], HO_exact_alpha_s_arr[r], HO_calc_alpha_s_arr[r] = routine_calc_overlaps(W_R, x_arr, x_span, Omega, N_state, band_num, k_0, single_shift, sepN_len, K)

        chi_HO = create_chi_HO(x_arr, Omega, band_num, K, k_0)
        # plot_Wannier_HO(W_R, chi_HO, x_arr, K, a_0, 1)
        # END OF PART 2

        # PART 3
        W_H_W_arr[r], HO_exact_tunneling_arr[r], chi_H_chi_arr[r] = set_tunnelings(W_R, D2_W_R, V_epsilon[r], x_arr, x_span, Omega, band_num, k_0, sep_len, K)
        # END OF PART 3

    save_array_NPY(W_W_alpha_s_arr, "W_W_alpha_s_arr!Omega_start=%s!Omega_end=%s" % (Omega_span[0], Omega_span[1]))
    save_array_NPY(HO_exact_alpha_s_arr, "HO_exact_alpha_s_arr!Omega_start=%s!Omega_end=%s" % (Omega_span[0], Omega_span[1]))

    return W_W_overlap_arr, HO_exact_overlap_arr, HO_calc_overlap_arr, W_W_alpha_s_arr, HO_exact_alpha_s_arr, HO_calc_alpha_s_arr, W_H_W_arr, chi_H_chi_arr, HO_exact_tunneling_arr


def overlap_interaction_tunneling(N_state, Omega_span, K, R, band_num, max_sep, x_max):
    # DERIVED PARAMETERS
    k_0 = 2.0*pi

    x_min = -x_max

    sep_int = int(max_sep)
    sep_len = int(max_sep) + 1
    sepN_int = int(N_state * max_sep)
    sepN_len = int(N_state * max_sep) + 1

    K = 2 * (K // 2) + 1

    x_span = (x_min, x_max)
    x_arr = np.linspace(x_min, x_max, K)
    x_step = (x_max - x_min) / (K - 1)
    K_0 = K // 2  # Assumes x_min = -x_max.

    single_shift = 1. / N_state

    Km_a = math.floor(1.0 / x_step)
    Km_a2 = Km_a // 2
    Km_2a = 2 * Km_a
    Km_aN = Km_a // N_state
    Km_2aN = 2 * Km_a // N_state

    Omega_arr = np.linspace(Omega_span[0], Omega_span[1], R)

    V_epsilon = -2.0 * Omega_arr[:, np.newaxis] * np.cos(k_0 * x_arr[np.newaxis, :])
    # END OF DERIVED PARAMETERS

    W_W_overlap_arr = np.zeros((R, band_num, band_num, sepN_len), dtype=np.float64)
    W_W_interaction_arr = np.zeros((R, band_num, band_num, sepN_len), dtype=np.float64)
    W_H_W_arr = np.zeros((R, band_num, band_num, sep_len), dtype=np.float64)

    for r in range(R):
        Omega = Omega_arr[r]

        # PART 1: GET band_num PRIMARY LATTICE WANNIER FUNCTIONS W_R
        W_R, D2_W_R = full_calc_Wannier_Kohn(Omega, (-x_max, x_max), band_num, K, N_Fourier=50, useCached=False)
        # W_R, D2_W_R = full_calc_Wannier_Kohn(Omega, (-x_max, x_max), band_num, K, N_Fourier=20, useCached=False)
        # END OF PART 1

        # PART 2
        W_W_overlap_arr[r] = calc_overlap(W_R, x_arr, x_span, single_shift, band_num, sepN_len)
        W_W_interaction_arr[r] = calc_interaction(W_R, x_arr, x_span, single_shift, band_num, sepN_len)
        W_H_W_arr[r] = calc_spectrum_tunnelings(Omega, sep_len, band_num, N_k=251, N_Fourier=100)
        # END OF PART 2

    # SAVING TO CSV
    save_array_CSV(W_W_interaction_arr[:, 0, 0, 0], "On-site_Interaction!N_state=%i" % N_state)
    save_array_CSV(W_W_interaction_arr[:, 0, 0, 0] / W_W_interaction_arr[:, 0, 0, 1], "Interaction_Ratio!N_state=%i" % N_state)
    save_array_CSV(W_W_interaction_arr[:, 0, 0, 1], "NN_Interaction!N_state=%i" % N_state)
    save_array_CSV(W_W_interaction_arr[:, 0, 0, 2], "NNN_Interaction!N_state=%i" % N_state)
    save_array_CSV(W_W_interaction_arr[:, 0, 0, 3], "4N_Interaction!N_state=%i" % N_state)
    save_array_CSV(W_W_interaction_arr[:, 0, 0, 4], "5N_Interaction!N_state=%i" % N_state)

    save_array_CSV(W_W_overlap_arr[:, 0, 0, 1], "NN_Overlap!N_state=%i" % N_state)
    save_array_CSV(W_W_overlap_arr[:, 0, 0, 1] / W_W_overlap_arr[:, 0, 0, 2], "Overlap_Ratio!N_state=%i" % N_state)
    save_array_CSV(W_W_overlap_arr[:, 0, 0, 2], "NNN_Overlap!N_state=%i" % N_state)
    save_array_CSV(W_W_overlap_arr[:, 0, 0, 3], "4N_Overlap!N_state=%i" % N_state)
    save_array_CSV(W_W_overlap_arr[:, 0, 0, 4], "5N_Overlap!N_state=%i" % N_state)

    save_array_CSV(W_H_W_arr[:, 0, 0, 1], "NN_Natural_Tunneling!N_state=%i" % N_state)
    save_array_CSV(W_H_W_arr[:, 0, 0, 2], "NNN_Natural_Tunneling!N_state=%i" % N_state)
    # END OF SAVING TO CSV

    return W_W_overlap_arr, W_W_interaction_arr, W_H_W_arr


def overlap_tunneling_mp(N_state, Omega_span, a_0, K, R, band_num, max_sep, x_max):
    """Multiprocessing version."""
    # start_time = time.perf_counter()

    # DERIVED PARAMETERS
    k_0 = 2.0 * np.pi / a_0

    x_min = -x_max

    sep_int = int(max_sep)
    sep_len = int(max_sep) + 1
    sepN_int = int(N_state * max_sep)
    sepN_len = int(N_state * max_sep) + 1

    K = 2 * (K // 2) + 1

    x_span = (x_min, x_max)
    x_arr = np.linspace(x_min, x_max, K)
    x_step = (x_max - x_min) / (K - 1)
    K_0 = K // 2  # Assumes x_min = -x_max.

    single_shift = 1. / N_state

    Km_a = math.floor(a_0 / x_step)
    Km_a2 = Km_a // 2
    Km_2a = 2 * Km_a
    Km_aN = Km_a // N_state
    Km_2aN = 2 * Km_a // N_state

    Omega_arr = np.linspace(Omega_span[0], Omega_span[1], R)

    V_epsilon = -2.0 * Omega_arr[:, np.newaxis] * np.cos(k_0 * x_arr[np.newaxis, :])
    # END OF DERIVED PARAMETERS

    W_W_overlap_arr = np.zeros((R, band_num, band_num, sepN_len), dtype=np.float64)
    HO_exact_overlap_arr = np.zeros((R, band_num, band_num, sepN_len), dtype=np.float64)
    HO_calc_overlap_arr = np.zeros((R, band_num, band_num, sepN_len), dtype=np.float64)
    W_W_alpha_s_arr = np.zeros((R, sepN_len), dtype=np.float64)
    HO_exact_alpha_s_arr = np.zeros((R, sepN_len), dtype=np.float64)
    HO_calc_alpha_s_arr = np.zeros((R, sepN_len), dtype=np.float64)

    W_H_W_arr = np.zeros((R, band_num, band_num, sep_len), dtype=np.float64)
    chi_H_chi_arr = np.zeros((R, band_num, band_num, sep_len), dtype=np.float64)
    HO_exact_tunneling_arr = np.zeros((R, band_num, band_num, sep_len), dtype=np.float64)

    def op(r):
        Omega = Omega_arr[r]

        # PART 1: GET band_num PRIMARY LATTICE WANNIER FUNCTIONS W_R
        # Make sure same Wannier function implementation is used everywhere to rule out inconsistencies.

        # W_R, D2_W_R = full_calc_Wannier(Omega, x_max, band_num, K)
        W_R, D2_W_R = full_calc_Wannier_Kohn(Omega, (-x_max, x_max), band_num, K, useCached=False)
        # END OF PART 1

        # PART 2
        W_W_overlap_arr[r], HO_exact_overlap_arr[r], HO_calc_overlap_arr[r], W_W_alpha_s_arr[r], HO_exact_alpha_s_arr[r], HO_calc_alpha_s_arr[r] = routine_calc_overlaps(W_R, x_arr, x_span, Omega, N_state, band_num, k_0, single_shift, sepN_len, K)

        chi_HO = create_chi_HO(x_arr, Omega, band_num, K, k_0)
        # plot_Wannier_HO(W_R, chi_HO, x_arr, K, a_0, 1)
        # END OF PART 2

        # PART 3
        W_H_W_arr[r], HO_exact_tunneling_arr[r], chi_H_chi_arr[r] = set_tunnelings(W_R, D2_W_R, V_epsilon[r], x_arr, x_span, Omega, band_num, k_0, sep_len, K)
        # END OF PART 3

    pool = Pool(11)
    pool.map(op, range(R))

    return W_W_overlap_arr, HO_exact_overlap_arr, HO_calc_overlap_arr, W_W_alpha_s_arr, HO_exact_alpha_s_arr, HO_calc_alpha_s_arr, W_H_W_arr, chi_H_chi_arr, HO_exact_tunneling_arr


def calc_1Omega_noHO_overlap_tunneling(Omega, max_sep, N_state, N_band, x_max=10., N_x=10000):
    # DERIVED PARAMETERS
    k_0 = 2.0 * np.pi

    x_min = -x_max

    single_shift = 1. / N_state

    sep_int = int(max_sep)
    sep_len = int(max_sep) + 1
    sepN_int = int(N_state * max_sep)
    sepN_len = int(N_state * max_sep) + 1

    x_span = (x_min, x_max)
    x_arr = np.linspace(*x_span, N_x)

    V_epsilon = -2.0 * Omega * np.cos(k_0 * x_arr)
    # END OF DERIVED PARAMETERS

    # Make sure same Wannier function implementation is used everywhere to rule out inconsistencies.

    # W_R, D2_W_R = full_calc_Wannier(Omega, x_max, N_band, N_x)
    W_R, D2_W_R = full_calc_Wannier_Kohn(Omega, (-x_max, x_max), N_band, N_x, useCached=True)  # (N_band, N_x)
    W_W_overlap_arr = calc_overlap(W_R, x_arr, x_span, single_shift, N_band, sepN_len)  # (N_band, N_band, sepN_len)

    # Tunneling parameters calculated from spectrum are more accurate.
    W_H_W_arr = calc_spectrum_tunnelings(Omega, sep_len, N_band, N_k=501, N_Fourier=200)

    # W_T_W_arr = calc_kinetic(W_R, D2_W_R, x_arr, x_span, k_0, N_band, sep_len)  # (N_band, N_band, sep_len)
    # W_V_W_arr = calc_potential(W_R, V_epsilon, x_arr, x_span, N_band, sep_len)  # (N_band, N_band, sep_len)
    # W_H_W_arr = W_T_W_arr + W_V_W_arr  # (N_band, N_band, sep_len)

    return W_W_overlap_arr, W_H_W_arr


def calc_Wannier_overlaps(Omega, max_sep, N_state, N_band, x_max=10., N_x=10000):
    # DERIVED PARAMETERS
    k_0 = 2.0 * np.pi

    x_min = -x_max

    single_shift = 1. / N_state

    sep_int = int(max_sep)
    sep_len = int(max_sep) + 1
    sepN_int = int(N_state * max_sep)
    sepN_len = int(N_state * max_sep) + 1

    x_span = (x_min, x_max)
    x_arr = np.linspace(*x_span, N_x)
    # END OF DERIVED PARAMETERS

    # Make sure same Wannier function implementation is used everywhere to rule out inconsistencies.

    # W_R, D2_W_R = full_calc_Wannier(Omega, x_max, N_band, N_x)
    W_R, D2_W_R = full_calc_Wannier_Kohn(Omega, (-x_max, x_max), N_band, N_x, useCached=False)  # (N_band, N_x)
    W_W_overlap_arr = calc_overlap(W_R, x_arr, x_span, single_shift, N_band, sepN_len)  # (N_band, N_band, sepN_len)

    return W_W_overlap_arr


def calc_spectrum_tunnelings(Omega, sep_len, N_band, N_k=501, N_Fourier=200):
    """From Edvinas's "Numerical method.pdf", Eq. (35)."""
    # Large values of N_k not required, N_k=250 seems to be plenty.
    # DERIVED PARAMETERS
    k_0 = 2 * pi
    k_arr = np.linspace(-pi, pi, N_k, endpoint=False)
    # END OF DERIVED PARAMETERS

    # CALCULATION OF SPECTRUM TUNNELINGS
    ## E.shape = (N_band, N_k)
    E, g_Fourier = diag_cos_lattice(k_arr, N_band, Omega, k_0, M_2=N_k, N=N_Fourier, N_2=2*N_Fourier+1)

    # E_k, u_Fourier = diag_sin_lattice(Omega, N_Fourier, N_k)
    # E_k = E_k[:, :N_band]
    # E = np.swapaxes(E_k, 0, 1)

    # Non-diagonal matrix elements should be zero.
    W_H_W_arr = np.zeros((N_band, N_band, sep_len), dtype=np.complex128)
    for i in range(N_band):
        # Since the lattice constant a_0 is assumed to be 1., index and separation are the same (sep=index*a_0).
        # for a in range(sep_len):
        for a in range(sep_len):
            exp_ika = np.exp(-1j * k_arr * a)
            W_H_W_arr[i, i, a] = 1./N_k * np.sum(exp_ika[:] * E[i, :])

    check_if_real(W_H_W_arr, "W_H_W_arr")
    W_H_W_arr = W_H_W_arr.real
    # END OF CALCULATION OF SPECTRUM TUNNELINGS

    return W_H_W_arr


def routine_band_flatness(Omega_span, band_num, R, colors, N_k=100, N_spline=1000):
    """Flatness of the i-th band is defined as the bandwidth of the i-th band
    divided by the band gap between the s (i=0) and p (i=1) bands."""

    # DERIVED PARAMETERS
    Omega_arr = np.linspace(*Omega_span, R)
    Omega_ext = np.linspace(*Omega_span, N_spline)  # Used for plot splines. 1000 can be replaced by any large integer.
    # END OF DERIVED PARAMETERS

    band_flatness = np.zeros((R, band_num), dtype=np.float64)
    for idx_Omega, Omega in enumerate(Omega_arr):
        E, g_Fourier = full_diag_cos_lattice(Omega, band_num, N_k)
        sp_gap = calc_indirect_gaps(E)[0]
        for idx_band in range(band_num):
            band_flatness[idx_Omega, idx_band] = calc_bandwidths(E)[idx_band] / sp_gap

    plot_band_flatness(Omega_arr, Omega_ext, band_flatness, colors)


def routine_overlap_tunneling(N_state, Omega_span, a_0, Delta, K, R, band_num, max_sep, x_max, colors):
    # DERIVED PARAMETERS
    Omega_arr = np.linspace(Omega_span[0], Omega_span[1], R)
    Omega_ext = np.linspace(Omega_span[0], Omega_span[1], 1000)  # Used for plot splines. 1000 can be replaced by any large integer.
    # END OF DERIVED PARAMETERS

    W_W_overlap_arr, HO_exact_overlap_arr, HO_calc_overlap_arr, W_W_alpha_s_arr, HO_exact_alpha_s_arr, HO_calc_alpha_s_arr, W_H_W_arr, chi_H_chi_arr, HO_exact_tunneling_arr = \
    overlap_tunneling(N_state, Omega_span, a_0, K, R, band_num, max_sep, x_max)
    # W_W_overlap_arr, HO_exact_overlap_arr, HO_calc_overlap_arr, W_W_alpha_s_arr, HO_exact_alpha_s_arr, HO_calc_alpha_s_arr, W_H_W_arr, chi_H_chi_arr, HO_exact_tunneling_arr = \
    # overlap_tunneling_mp(N_state, Omega_span, a_0, K, R, band_num, max_sep, x_max)

    # PLOTTING
    plot_W_W_overlap(W_W_overlap_arr, HO_exact_overlap_arr, HO_calc_overlap_arr, W_W_alpha_s_arr, HO_exact_alpha_s_arr, HO_calc_alpha_s_arr, Omega_arr, Omega_ext, colors)
    plot_W_tunneling(W_H_W_arr, HO_exact_tunneling_arr, chi_H_chi_arr, Omega_arr, Omega_ext, colors)

    # plot_W_natural_tunnelings(W_H_W_arr, Omega_arr, Omega_ext, colors)
    # END OF PLOTTING


def routine_overlap_tunneling_v2(N_state, Omega_span, Delta, K, R, band_num, max_sep, x_max):
    """Written for subwavelength Haldane phase project."""

    # DERIVED PARAMETERS
    Omega_arr = np.linspace(Omega_span[0], Omega_span[1], R)
    Omega_ext = np.linspace(Omega_span[0], Omega_span[1], 1000)  # Used for plot splines. 1000 can be replaced by any large integer.
    # END OF DERIVED PARAMETERS

    W_W_overlap_arr, W_W_interaction_arr, W_H_W_arr = overlap_interaction_tunneling(N_state, Omega_span, K, R, band_num, max_sep, x_max)

    # OUTPUT
    # for i in range(5):
    #     print("W_W_overlap_arr[:, 0, 0, %i]:" % i, W_W_overlap_arr[:, 0, 0, i])
    #     print("W_W_interaction_arr[:, 0, 0, %i]:" % i, W_W_interaction_arr[:, 0, 0, i])
    # END OF OUTPUT

    # TESTS
    for idx_Omega in range(R):
        # for idx_band in range(band_num):
        for idx_band in range(1):
            assert np.allclose(W_W_overlap_arr[idx_Omega, idx_band, idx_band, 0], 1.0)
            print("W_W_overlap_arr[%i, %i, %i, N_state]" % (idx_Omega, idx_band, idx_band), W_W_overlap_arr[idx_Omega, idx_band, idx_band, N_state])
            assert np.allclose(W_W_overlap_arr[idx_Omega, idx_band, idx_band, N_state], 0.0, atol=1e-3)
    # END OF TESTS

    # PLOTTING
    plot_W_interactions(W_W_interaction_arr, Omega_arr, Omega_ext, N_state)
    plot_W_overlap_v2(W_W_overlap_arr, Omega_arr, Omega_ext, N_state)
    plot_W_tunneling_v2(W_H_W_arr, Omega_arr, Omega_ext, N_state)
    # END OF PLOTTING


def routine_spectrum_tunneling(Omega_span, R, N_band, max_sep, colors):
    # DERIVED PARAMETERS
    sep_len = int(max_sep) + 1

    Omega_arr = np.linspace(Omega_span[0], Omega_span[1], R)
    Omega_ext = np.linspace(Omega_span[0], Omega_span[1], 1000)  # Used for plot splines. 1000 can be replaced by any large integer.
    # END OF DERIVED PARAMETERS

    W_H_W_arr = np.empty((R, N_band, N_band, sep_len), dtype=np.float64)
    for r in range(R):
        Omega = Omega_arr[r]
        W_H_W_arr[r] = calc_spectrum_tunnelings(Omega, sep_len, N_band, N_k=251, N_Fourier=100)

    np.save("W_H_W_arr!Omega_start=%.3f!Omega_end=%.3f!R=%i" % (Omega_span[0], Omega_span[1], R), W_H_W_arr)
    # np.save("ss_natural_tunneling!Omega_start=%.3f!Omega_end=%.3f!R=%i" % (Omega_span[0], Omega_span[1], R), W_H_W_arr[:, 0, 0, 1])
    # np.save("pp_natural_tunneling!Omega_start=%.3f!Omega_end=%.3f!R=%i" % (Omega_span[0], Omega_span[1], R), W_H_W_arr[:, 1, 1, 1])

    # PLOTTING
    plot_W_natural_tunnelings(W_H_W_arr, Omega_arr, Omega_ext, colors)
    # END OF PLOTTING
