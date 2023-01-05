from datetime import datetime
import logging
import math
from numba import njit, prange
import numpy as np
import os
import pathlib
from scipy.interpolate import splrep, splev
import scipy.sparse
import scipy.sparse.linalg
from scipy.special import hermite
import matplotlib.pyplot as plt

from QOPack.Math import abs2
from QOPack.Utility import *


# @fun_time
@njit(cache=True, parallel=True)
def calc_Wannier(eig_i, Bloch_p, x_Wannier_arr, M, N, q_arr, q_step, M_2, N_2, M_nhp, M_php, K_Wannier, K_Wannier_0):
    g = np.zeros((M_2, K_Wannier), dtype=np.complex128)
    Wannier_R = np.zeros(K_Wannier, dtype=np.complex128)

    for i in prange(M_2):
        if Bloch_p[i, 0].real < 0.0:
            Bloch_p[i] *= -1.0
        theta = 0.0
        if eig_i % 2 == 1:
            if i < M:
                theta += np.pi / 2
            elif i > M:
                theta -= np.pi / 2
        Bloch_p[i] *= np.exp(1.j * theta)

        for k in prange(K_Wannier):
            g[i, k] = Bloch_p[i, (k - K_Wannier_0) % N_2]

    for k in range(K_Wannier):
        for l in range(M_nhp, M_php):
            Wannier_R[k] += np.exp(1.j * q_arr[l] * x_Wannier_arr[k]) * g[l, k]
    Wannier_R *= q_step

    W_R_int = np.trapz(abs2(Wannier_R), x_Wannier_arr)
    Wannier_R /= np.sqrt(W_R_int)

    return Bloch_p, Wannier_R


def change_Wannier_discretization(Wannier_R, x_Wannier_arr, x_arr, band_num, K):
    W_R = np.zeros((band_num, K), dtype=np.float64)
    D2_W_R = np.zeros((band_num, K), dtype=np.float64)
    for a in range(band_num):
        tck_real = splrep(x_Wannier_arr, Wannier_R[a].real)
        tck_imag = splrep(x_Wannier_arr, Wannier_R[a].imag)
        W_R_real = splev(x_arr, tck_real, der=0)
        W_R_imag = splev(x_arr, tck_imag, der=0)
        D2_W_real = splev(x_arr, tck_real, der=2)
        D2_W_imag = splev(x_arr, tck_imag, der=2)

        W_R[a] = W_R_real  # + 1.j * W_R_imag
        D2_W_R[a] = D2_W_real  # + 1.j * D2_W_imag

    return W_R, D2_W_R


def diag_cos_lattice(q_arr, band_num, Omega, k_0, M_2, N, N_2):
    E = np.zeros((band_num, M_2))
    g_Fourier = np.zeros((band_num, M_2, N_2), dtype=np.complex128)

    main_diag = np.zeros(N_2)
    off_diag = np.full(N_2, -Omega)

    for i in range(M_2):
        for l in range(-N, N + 1):
            main_diag[l + N] = (l + q_arr[i] / k_0) ** 2
        data = np.array([off_diag, main_diag, off_diag])
        diags = np.array([-1, 0, 1])
        H = scipy.sparse.spdiags(data, diags, N_2, N_2)
        # sigma=-10.0
        eigen_arr = scipy.sparse.linalg.eigsh(H, k=band_num, sigma=-10.)
        # eigen_arr = scipy.sparse.linalg.eigsh(H, k=band_num, sigma=-4.*Omega, which="LM")
        # eigen_arr = scipy.sparse.linalg.eigsh(H, k=band_num, which='SA')
        for j in range(band_num):
            E[j, i] = eigen_arr[0][j].real

        for a in range(band_num):
            for j in range(N_2):
                g_Fourier[a, i, j] = eigen_arr[1][j][a]

    return E, g_Fourier


def calc_Bloch_Wannier(g_Fourier, x_arr, x_p_arr, x_Wannier_arr, band_num, M, N, q_arr, q_step, M_2, N_2, M_nhp, M_php, K, K_Wannier, K_Wannier_0):
    Bloch_p = np.zeros((band_num, M_2, N_2), dtype=np.complex128)
    Wannier_R = np.zeros((band_num, K_Wannier), dtype=np.complex128)

    for a in range(band_num):
        for i in range(M_2):
            Bloch_p[a, i] = np.fft.ifft(np.fft.ifftshift(g_Fourier[a, i]))

            g_int = np.trapz(abs2(Bloch_p[a, i]), x_p_arr)
            Bloch_p[a, i] /= np.sqrt(g_int)

    for a in range(band_num):
        Bloch_p[a], Wannier_R[a] = calc_Wannier(a, Bloch_p[a], x_Wannier_arr, M, N, q_arr, q_step, M_2, N_2, M_nhp, M_php,
                                                K_Wannier, K_Wannier_0)

    W_R, D2_W_R = change_Wannier_discretization(Wannier_R, x_Wannier_arr, x_arr, band_num, K)

    return Bloch_p, W_R, D2_W_R


def full_diag_cos_lattice(Omega, band_num, N_q):
    a_0 = 1.

    # COMPUTATIONAL PARAMETERS
    # M = 1000  # Number of points in q-space
    N = 1000  # Fourier component number
    # END OF COMPUTATIONAL PARAMETERS

    # DERIVED PARAMETERS
    k_0 = 2. * np.pi / a_0

    q_max = np.pi / a_0
    q_min = -q_max

    N_2 = 2 * N + 1

    q_arr = np.linspace(q_min, q_max, N_q)
    # END OF DERIVED PARAMETERS

    E, g_Fourier = diag_cos_lattice(q_arr, band_num, Omega, k_0, N_q, N, N_2)

    return E, g_Fourier


# @fun_time
def full_calc_Wannier(Omega, x_max, band_num, K):
    path_string = "%s/NPY/Wannier" % get_main_dir()
    name_string = "Omega=%.3f!x_max=%.3f!band_num=%i!K=%i.npy" % (Omega, x_max, band_num, K)
    if os.path.exists("%s/W_R!%s" % (path_string, name_string)) and \
    os.path.exists("%s/D2_W_R!%s" % (path_string, name_string)):
        W_R, D2_W_R = read_Wannier_NPY(Omega, x_max, band_num, K)
    else:
        a_0 = 1.

        # COMPUTATIONAL PARAMETERS
        M = 1000  # Number of points in q-space
        N = 1000  # Fourier component number
        # END OF COMPUTATIONAL PARAMETERS

        # DERIVED PARAMETERS
        x_min = -x_max
        x_arr = np.linspace(x_min, x_max, K)

        k_0 = 2. * np.pi / a_0

        q_max = np.pi / a_0
        q_min = -q_max

        N_2 = 2 * N + 1
        M_2 = 2 * M + 1

        q_step = (q_max - q_min) / (2 * M)
        x_p_step = a_0 / (2 * N)

        K_Wannier = 2 * math.floor((x_max - x_min) / x_p_step) // 2 + 1
        K_Wannier_0 = K_Wannier // 2

        M_nhp = math.floor((-np.pi / a_0 - q_min) / q_step)
        M_php = math.floor((np.pi / a_0 - q_min) / q_step)

        q_arr = np.linspace(q_min, q_max, M_2)
        x_p_arr = np.linspace(-0.5 * a_0, 0.5 * a_0, N_2)
        x_Wannier_arr = np.linspace(x_min, x_max, K_Wannier)
        # END OF DERIVED PARAMETERS

        E, g_Fourier = diag_cos_lattice(q_arr, band_num, Omega, k_0, M_2, N, N_2)
        Bloch_p, W_R, D2_W_R = calc_Bloch_Wannier(g_Fourier, x_arr, x_p_arr, x_Wannier_arr, band_num, M, N, q_arr, q_step, M_2, N_2, M_nhp, M_php, K, K_Wannier, K_Wannier_0)

        # check_Bloch_orthonorm(Bloch_p, x_p_arr, band_num, M_2)
        # check_Wannier_sym(W_R, band_num, K, K_0)
        # check_Wannier_orthonorm(W_R, x_arr, band_num)

        write_Wannier_NPY(W_R, D2_W_R, Omega, x_max, band_num, K)

    return W_R, D2_W_R


def create_chi_HO(x_arr, Omega, band_num, K, k_0):
    chi_HO = np.zeros((band_num, K), dtype=np.float64)
    k_x_arr = k_0 * x_arr
    for a in prange(band_num):
        H_alpha = hermite(a)
        chi_HO[a] = (-1.0)**a * np.exp(-Omega**0.5 * k_x_arr**2 / 2.0) * H_alpha(Omega**0.25 * k_x_arr)
        chi2_int = np.trapz(abs2(chi_HO[a]), x_arr)
        chi_HO[a] /= np.sqrt(chi2_int)

    return chi_HO


# @fun_time
def check_Bloch_orthonorm(Bloch_p, x_p_arr, band_num, M_2):
    orthonorm_g_p_arr = np.zeros((band_num, band_num, M_2), dtype=np.float64)
    for a in range(band_num):
        for b in range(band_num):
            for i in range(M_2):
                orthonorm_g_p_arr[a, b, i] = abs2(np.trapz(np.conj(Bloch_p[a, i]) * Bloch_p[b, i], x_p_arr))

    epsilon = 1E-5
    logging.info("orthonorm_g_p_arr:")
    for a in range(band_num):
        for b in range(band_num):
            for i in range(M_2):
                if a == b:
                    if orthonorm_g_p_arr[a, b, i] < 1.0 - epsilon and 1.0 + epsilon < orthonorm_g_p_arr[a, b, i]:
                        logging.warning("WARNING: Bloch functions aren't normalized.")
                else:
                    if epsilon < orthonorm_g_p_arr[a, b, i]:
                        logging.warning("WARNING: Bloch functions aren't orthogonal.")
    logging.info("************")


# @fun_time
def check_Wannier_sym(W_R, band_num, K, K_0):
    sym_W_R_arr = np.zeros(band_num, dtype=np.float64)
    for a in range(band_num):
        for k_plus in range(1, K - K_0):
            if a % 2 == 0:
                sym_W_R_arr[a] += abs2(W_R[a, K_0 + k_plus] - W_R[a, K_0 - k_plus])
            else:
                sym_W_R_arr[a] += abs2(W_R[a, K_0 + k_plus] + W_R[a, K_0 - k_plus])

    epsilon = 1E-5
    logging.info("sym_W_R_arr:")
    for a in range(band_num):
        logging.info("%i: %s" % (a, sym_W_R_arr[a]))
        if epsilon < sym_W_R_arr[a]:
            if a % 2 == 0:
                logging.warning("WARNING: Wannier functions aren't symmetric.")
            else:
                logging.warning("WARNING: Wannier functions aren't antisymmetric.")
    logging.info("************")


# @fun_time
def check_Wannier_orthonorm(W_R, x_arr, band_num):
    orthonorm_W_R_arr = np.zeros((band_num, band_num), dtype=np.float64)
    for a in range(band_num):
        for b in range(band_num):
            orthonorm_W_R_arr[a, b] = abs2(np.trapz(np.conj(W_R[a]) * W_R[b], x_arr))

    epsilon = 1E-5
    logging.info("orthonorm_W_R_arr:")
    for a in range(band_num):
        for b in range(band_num):
            logging.info("%i, %i: %s" % (a, b, orthonorm_W_R_arr[a, b]))
            if a == b:
                if orthonorm_W_R_arr[a, b] < 1.0 - epsilon and 1.0 + epsilon < orthonorm_W_R_arr[a, b]:
                    logging.warning("WARNING: Wannier functions aren't normalized.")
            else:
                if epsilon < orthonorm_W_R_arr[a, b]:
                    logging.warning("WARNING: Wannier functions aren't orthogonal.")
    logging.info("************")


def read_Wannier_NPY(Omega, x_max, band_num, K):
    path_string = "%s/NPY/Wannier" % get_main_dir()
    name_string = "Omega=%.3f!x_max=%.3f!band_num=%i!K=%i.npy" % (Omega, x_max, band_num, K)

    W_R = np.load("%s/W_R!%s" % (path_string, name_string))
    D2_W_R = np.load("%s/D2_W_R!%s" % (path_string, name_string))

    return W_R, D2_W_R


def write_Wannier_NPY(W_R, D2_W_R, Omega, x_max, band_num, K):
    path_string = "%s/NPY/Wannier" % get_main_dir()
    pathlib.Path(path_string).mkdir(parents=True, exist_ok=True)

    name_string = "Omega=%.3f!x_max=%.3f!band_num=%i!K=%i.npy" % (Omega, x_max, band_num, K)

    np.save("%s/W_R!%s" % (path_string, name_string), W_R)
    np.save("%s/D2_W_R!%s" % (path_string, name_string), D2_W_R)


def plot_Wannier(W_R, x_arr, S_band, fig_ax=None):
    fig, ax = get_fig_ax(fig_ax, 'Wannier')

    plt.title("Wannier")
    plt.xlabel("x / $a_0$")
    plt.ylabel("W / $a_0^{-0.5}$")
    plt.plot(x_arr, W_R[S_band].real, color="black")
    plt.fill_between(x_arr, W_R[S_band].real, W_R[S_band].real + W_R[S_band].imag, color="red")

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot("Wannier")
        plt.show()
