from math import pi
import numpy as np
from numba import njit, prange
import scipy.sparse
import scipy.linalg.lapack

from QOPack.Utility import *
from QOPack.Math import *
# from QOPack.RamanTimeDepDetuning.OverlapTunneling import create_chi_HO
from QOPack.RamanTimeDepDetuning.OverlapTunneling import *


def diag_sin_lattice(Omega, N_Fourier, N_k):
    N2_Fourier = 2*N_Fourier + 1

    k_arr = np.linspace(-pi, pi, N_k)

    E_k = np.empty((N_k, N2_Fourier), dtype=np.float64)
    u_Fourier = np.empty((N_k, N2_Fourier, N2_Fourier), dtype=np.complex128)

    off_diag = np.full(N2_Fourier, -Omega)
    for i in range(N_k):
        ##### CHOICE OF ENERGY SHIFT #####
        main_diag = np.zeros(N2_Fourier)
        # main_diag = np.full(N2_Fourier, 2*Omega)
        ##### END OF CHOICE OF ENERGY SHIFT #####

        for j in range(-N_Fourier, N_Fourier+1):
            main_diag[j+N_Fourier] += (j + k_arr[i]/(2*pi))**2

        # E_k[i, :], v, info = scipy.linalg.lapack.dstev(main_diag, off_diag[:-1])
        # u_Fourier[i, :, :] = v.T

        diagonals = [off_diag, main_diag, off_diag]
        offsets = [-1, 0, 1]
        Ham_k = scipy.sparse.diags(diagonals, offsets, shape=(N2_Fourier, N2_Fourier)).toarray()
        E_k[i, :], v = np.linalg.eigh(Ham_k)
        u_Fourier[i, :, :] = v.T

    return E_k, u_Fourier


def sparse_diag_sin_lattice(Omega, N_band, N_Fourier, N_k):
    N2_Fourier = 2*N_Fourier + 1

    k_arr = np.linspace(-pi, pi, N_k)

    E_k = np.empty((N_k, N_band), dtype=np.float64)
    u_Fourier = np.empty((N_k, N_band, N2_Fourier), dtype=np.complex128)

    # off_diag = np.full(N2_Fourier, -Omega/4)
    off_diag = np.full(N2_Fourier, -Omega)

    main_diag = np.empty(N2_Fourier)
    for i in range(N_k):
        for j in range(-N_Fourier, N_Fourier+1):
            main_diag[j+N_Fourier] = (j + k_arr[i]/(2*pi))**2

        diagonals = [off_diag, main_diag, off_diag]
        offsets = [-1, 0, 1]

        Ham_k = scipy.sparse.diags(diagonals, offsets, shape=(N2_Fourier, N2_Fourier)).toarray()

        # E_k[i, :], v = scipy.sparse.linalg.eigsh(Ham_k, N_band, which='SA')
        E_k[i, :], v = scipy.sparse.linalg.eigsh(Ham_k, k=N_band, sigma=-4.*Omega, which="LM")
        u_Fourier[i, :, :] = v.T

    return E_k, u_Fourier


@njit(cache=True)
def calc_u_k(u_Fourier, x_arr, N_Fourier, N_k, N_x):
    N2_Fourier = 2*N_Fourier + 1

    # Assumes u_Fourier.shape is either:
    # 1) (N_k, N2_Fourier, N2_Fourier) for dense implementation;
    # 2) (N_k, N_band, N2_Fourier) for sparse implementation;
    u_k = np.zeros((N_k, u_Fourier.shape[1], N_x), dtype=np.complex128)

    l_arr = np.arange(-N_Fourier, N_Fourier+1)
    exp_ilkx = np.exp(2j * pi * np.expand_dims(l_arr, axis=1) * np.expand_dims(x_arr, axis=0))  # (N2_Fourier, N_x)
    for i in range(N_k):
        for j in range(u_Fourier.shape[1]):
            for k in range(N_x):
                for l in range(N2_Fourier):
                    u_k[i, j, k] += u_Fourier[i, j, l] * exp_ilkx[l, k]

    ##### Slower implementation #####
    # for l in range(N2_Fourier):
    #     for k in range(N_x):
    #         exp_ilkx = np.exp(2j*pi*(l-N_Fourier)*x_arr[k])
    #         for i in range(N_k):
    #             for j in range(u_Fourier.shape[1]):
    #                 u_k[i, j, k] += u_Fourier[i, j, l] * exp_ilkx

    return u_k


@njit(cache=True)
def calc_Bloch_k(u_k, k_arr, x_arr):
    # exp_ikx = np.exp(1j * k_arr[:, np.newaxis] * x_arr[np.newaxis, :])  # (N_k, N_x)
    # Bloch_k = exp_ikx[:, np.newaxis, :] * u_k  # (N_k, N2_Fourier, N_x)
    exp_ikx = np.exp(1j * np.expand_dims(k_arr, axis=1) * np.expand_dims(x_arr, axis=0))  # (N_k, N_x)
    Bloch_k = np.expand_dims(exp_ikx, axis=1) * u_k  # (N_k, N2_Fourier, N_x) or (N_k, N_band, N_x)

    return Bloch_k


def solve_sin_Schrodinger(Omega, x_span, N_band, N_Fourier, N_k, N_x):
    # DERIVED PARAMETERS
    k_arr = np.linspace(-pi, pi, N_k)
    x_arr = np.linspace(*x_span, N_x)
    # END OF DERIVED PARAMETERS

    # E_k, u_Fourier = diag_sin_lattice(Omega, N_Fourier, N_k)
    E_k, u_Fourier = sparse_diag_sin_lattice(Omega, N_band, N_Fourier, N_k)

    u_k = calc_u_k(u_Fourier, x_arr, N_Fourier, N_k, N_x)
    Bloch_k = calc_Bloch_k(u_k, k_arr, x_arr)
    # Bloch_k = normalize_wavefunction(Bloch_k, x_arr)

    return E_k, Bloch_k


def smooth_gauge_Bloch_Kohn(Bloch_k, k_arr, x_arr):
    # ENSURING SMOOTH GAUGE IN MOMENTUM SPACE
    N_x0 = x2index((x_arr[0], x_arr[-1]), len(x_arr), 0.)

    for i, k in enumerate(k_arr):
        for j in range(Bloch_k.shape[1]):
            if j % 2 == 0:
                if Bloch_k[i, j, N_x0].real < 0.:
                    Bloch_k[i, j] *= -1.0
            else:
                if Bloch_k[i, j, N_x0-2].imag < 0.:
                    Bloch_k[i, j] *= 1j
                else:
                    Bloch_k[i, j] *= -1j

    for j in range(Bloch_k.shape[1]):
        if np.real(Bloch_k[0, j, N_x0]*Bloch_k[1, j, N_x0]) < 0.:
            Bloch_k[0, j] *= -1
        if np.real(Bloch_k[-1, j, N_x0]*Bloch_k[-2, j, N_x0]) < 0.:
            Bloch_k[-1, j] *= -1
    # END OF ENSURING SMOOTH GAUGE IN MOMENTUM SPACE

    # *************** OLD ***************
    # ENSURING SMOOTH GAUGE IN MOMENTUM SPACE
    # THIS ONE GIVES THE SAME RESULTS
    # N_x0 = x2index((x_arr[0], x_arr[-1]), len(x_arr), 0.)

    # for i, k in enumerate(k_arr):
    #     for j in range(Bloch_k.shape[1]):
    #         if Bloch_k[i, j, N_x0].real < 0.:
    #             Bloch_k[i, j] *= -1.0

    #         if j % 2 == 1:
    #             if k < 0:
    #                 Bloch_k[i, j] *= 1j
    #             # elif k > 0:
    #             else:
    #                 Bloch_k[i, j] *= -1j

    # for j in range(Bloch_k.shape[1]):
    #     if np.real(Bloch_k[0, j, N_x0]*Bloch_k[1, j, N_x0]) < 0.:
    #         Bloch_k[0, j] *= -1
    #     if np.real(Bloch_k[-1, j, N_x0]*Bloch_k[-2, j, N_x0]) < 0.:
    #         Bloch_k[-1, j] *= -1
    # END OF ENSURING SMOOTH GAUGE IN MOMENTUM SPACE

    # ENSURING SMOOTH GAUGE IN MOMENTUM SPACE
    # N_x0 = x2index((x_arr[0], x_arr[-1]), len(x_arr), 0.)

    # theta = np.empty((N_k, N2_Fourier), dtype=np.float64)
    # for i in range(N2_Fourier):
    #     theta[:, i] = -np.arctan2(np.imag(Bloch_k[:, i, N_x0]), np.real(Bloch_k[:, i, N_x0]))

    #     if i % 2 == 1:
    #         theta[:, i] += pi/2

    # Bloch_k *= np.exp(1j * theta[..., np.newaxis])
    # END OF ENSURING SMOOTH GAUGE IN MOMENTUM SPACE
    # *************** END OF OLD ***************

    return Bloch_k


def smooth_gauge_Bloch_Lowdin(Bloch_k, x_arr, Omega):
    """Based on Marzari2012, Maximally localized Wannier functions: Theory and
    applications, Wannier functions via projection."""
    N_band = Bloch_k.shape[1]
    N_k = Bloch_k.shape[0]
    N_x = len(x_arr)

    ########## TRIAL ORBITALS ##########
    # Trial orbitals shall be the harmonic oscillator wavefunctions
    # (cosinusoidal potential is approximated by a parabolic potential).  Even
    # though approximate orbitals are used, the answers shall be near exact.
    # g_n.shape = (N_band, N_x)
    # g_n = create_chi_HO(x_arr, Omega, N_band, N_x, 2*pi)

    g_n = np.zeros((N_band, N_x), dtype=np.float64)
    for idx_band in range(N_band):
        g_n[idx_band] = (2*pi*x_arr)**(idx_band) * np.exp(-np.abs(2*pi*x_arr))

        norm = np.trapz(abs2(g_n[idx_band]), x_arr)
        g_n[idx_band] /= np.sqrt(norm)
    ########## END OF TRIAL ORBITALS ##########

    Lowdin_Bloch_k = np.empty_like(Bloch_k)
    for idx_k in range(N_k):
        ##### MATRIX OF INNER PRODUCTS #####
        # A_k.shape = (N_band, N_band)
        A_k = np.trapz(np.conj(Bloch_k[idx_k, :, np.newaxis, :]) * g_n[np.newaxis, :, :], x_arr, axis=-1)
        ##### END OF MATRIX OF INNER PRODUCTS #####

        ##### TRANSFORMED BLOCH FUNCTIONS #####
        # phi_k.shape = (N_band, N_x)
        phi_k = np.sum(Bloch_k[idx_k, :, np.newaxis, :] * A_k[:, :, np.newaxis], axis=0)
        ##### END OF TRANSFORMED BLOCH FUNCTIONS #####

        ##### OVERLAP MATRIX #####
        # S_k.shape = (N_band, N_band)
        S_k = np.conj(A_k.T) * A_k
        # check_if_Hermitian(S_k, "S_k")
        ##### END OF OVERLAP MATRIX #####

        ##### OBTAINING S_k^{-1/2} #####
        # Using well-known property:
        # A^{-1/2} = V @ Lambda^{-1/2} @ V^{-1};
        lambda_k, V = np.linalg.eigh(S_k)  # S_k is Hermitian

        lambda_k = lambda_k.astype(np.complex128)  # Done so that np.sqrt appropriately handles square roots of negative numbers.
        inv_sq_Lambda_k = np.diag(np.sqrt(lambda_k))

        # inv_sq_S_k.shape = (N_band, N_band)
        inv_sq_S_k = V @ inv_sq_Lambda_k @ np.conj(V.T)
        ##### END OF OBTAINING S_k^{-1/2} #####

        Lowdin_Bloch_k[idx_k, :, :] = np.sum(phi_k[:, np.newaxis, :] * inv_sq_S_k[:, :, np.newaxis], axis=0)

    return Lowdin_Bloch_k


def calc_Wannier_Kohn(Bloch_k, k_arr, x_arr, Omega, N_k, N_x, N2_Fourier):
    Bloch_k = smooth_gauge_Bloch_Kohn(Bloch_k, k_arr, x_arr)  # (N_k, N2_Fourier, N_x) or (N_k, N_band, N_x)
    ## Lowdin works worse, probably because of suboptimal trial orbitals, e.g.,
    ## harmonic oscillator eigenstates decay as Gaussians, while Wannier decays
    ## exponentially.
    ## Lowdin kinda works for lowest band with N_band=1.
    # Bloch_k = smooth_gauge_Bloch_Lowdin(Bloch_k, x_arr, Omega)

    Wannier = np.trapz(Bloch_k, k_arr, axis=0)  # (N2_Fourier, N_x) or (N_band, N_x)
    Wannier = normalize_wavefunction(Wannier, x_arr)  # (N2_Fourier, N_x) or (N_band, N_x)

    # Maximally localized Wannier function (MLWF) of a 1D sinusoidal lattice should be real.
    check_if_real(Wannier, 'Wannier')
    Wannier = np.real(Wannier)

    return Bloch_k, Wannier


# @fun_time
def full_calc_Wannier_Kohn(Omega, x_span, N_band, N_x=10000, N_k=250, N_Fourier=50, useCached=True):
    """If N_Fourier is too high, it lags the computer (only for dense implementation).
    WARNING: May be inaccurate for high N_band. N_Fourier should always be larger than N_band // 2, preferably by a lot."""
    path_string = "%s/NPY/Wannier" % get_main_dir()
    name_string = "Omega=%.3f!x_span=(%.3f,%.3f)!N_band=%i!N_x=%i.npy" % (Omega, x_span[0], x_span[1], N_band, N_x)

    if useCached and \
    os.path.exists("%s/W_R!%s" % (path_string, name_string)) and \
    os.path.exists("%s/D2_W_R!%s" % (path_string, name_string)):
        Wannier, D2_Wannier = load_Wannier_NPY(Omega, x_span, N_band, N_x)
    else:
        # DERIVED PARAMETERS
        N_Fourier = 50
        N2_Fourier = 2*N_Fourier + 1

        k_arr = np.linspace(-pi, pi, N_k)
        x_arr = np.linspace(*x_span, N_x)
        # END OF DERIVED PARAMETERS

        E_k, Bloch_k = solve_sin_Schrodinger(Omega, x_span, N_band, N_Fourier, N_k, N_x)
        Bloch_k, Wannier = calc_Wannier_Kohn(Bloch_k, k_arr, x_arr, Omega, N_k, N_x, N2_Fourier)  # (N2_Fourier, N_x)

        D2_Wannier = np.empty_like(Wannier, dtype=np.float64)
        for i in range(Wannier.shape[0]):
            D2_Wannier[i] = spl_Dn_y(Wannier[i], x_arr, der=2)
            # D2_Wannier[i] = spectral_Dn_y(Wannier[i], x_arr, der=2)

        # Truncation
        Wannier = Wannier[:N_band]
        D2_Wannier = D2_Wannier[:N_band]

        save_Wannier_NPY(Wannier, D2_Wannier, Omega, x_span, N_band, N_x)

    return Wannier, D2_Wannier


def save_Wannier_NPY(W_R, D2_W_R, Omega, x_span, N_band, N_x):
    path_string = "%s/NPY/Wannier" % get_main_dir()
    pathlib.Path(path_string).mkdir(parents=True, exist_ok=True)

    name_string = "Omega=%.3f!x_span=(%.3f,%.3f)!N_band=%i!N_x=%i.npy" % (Omega, x_span[0], x_span[1], N_band, N_x)

    np.save("%s/W_R!%s" % (path_string, name_string), W_R)
    np.save("%s/D2_W_R!%s" % (path_string, name_string), D2_W_R)


def load_Wannier_NPY(Omega, x_span, N_band, N_x):
    path_string = "%s/NPY/Wannier" % get_main_dir()
    name_string = "Omega=%.3f!x_span=(%.3f,%.3f)!N_band=%i!N_x=%i.npy" % (Omega, x_span[0], x_span[1], N_band, N_x)

    W_R = np.load("%s/W_R!%s" % (path_string, name_string))
    D2_W_R = np.load("%s/D2_W_R!%s" % (path_string, name_string))

    return W_R, D2_W_R


# --------------- PLOTTING ---------------
# SINGLE PLOTS
def plot_u_k_vs_x(x_arr, u_k, S_band, S_k, Omega, fig_ax=None, file_name="u_k_vs_x"):
    fig, ax = get_fig_ax(fig_ax)

    if fig_ax is None:
        ax.set_title(r"$\Omega = %.2fE_R$" % Omega)

    ax.set_xlabel(r"$x$ / $a$")
    ax.set_ylabel(r"$u_k(x)$ / $a^{-1/2}$")
    ax.axhline(0, color='black', ls='--')
    ax.plot(x_arr, np.real(u_k[S_k, S_band, :]), color='blue')
    ax.fill_between(x_arr, np.real(u_k[S_k, S_band, :]), np.real(u_k[S_k, S_band, :])+np.imag(u_k[S_k, S_band, :]), color='red')

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot(file_name)
        plt.show()


def plot_Bloch_k_vs_x(x_arr, Bloch_k, S_band, S_k, Omega, fig_ax=None, file_name="Bloch_k_vs_x"):
    fig, ax = get_fig_ax(fig_ax)

    if fig_ax is None:
        ax.set_title(r"$\Omega = %.2fE_R$" % Omega)

    ax.set_xlabel(r"$x$ / $a$")
    ax.set_ylabel(r"$\Psi_k(x)$ / $a^{-1/2}$")
    ax.axhline(0, color='black', ls='--')
    ax.plot(x_arr, np.real(Bloch_k[S_k, S_band, :]), color='blue')
    ax.fill_between(x_arr, np.real(Bloch_k[S_k, S_band, :]), np.real(Bloch_k[S_k, S_band, :])+np.imag(Bloch_k[S_k, S_band, :]), color='red')

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot(file_name)
        plt.show()


def plot_u_k_vs_k(k_arr, u_k, S_band, S_x, Omega, fig_ax=None, file_name="u_k_vs_x"):
    fig, ax = get_fig_ax(fig_ax)

    if fig_ax is None:
        ax.set_title(r"$\Omega = %.2fE_R$" % Omega)

    ax.set_xlabel(r"$k$ / rad")
    ax.set_ylabel(r"$u_k$ / $a^{-1/2}$")
    ax.axhline(0, color='black', ls='--')
    ax.plot(k_arr, np.real(u_k[:, S_band, S_x]), color='blue')
    ax.fill_between(k_arr, np.real(u_k[:, S_band, S_x]), np.real(u_k[:, S_band, S_x])+np.imag(u_k[:, S_band, S_x]), color='red')

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot(file_name)
        plt.show()


def plot_Bloch_k_vs_k(k_arr, Bloch_k, S_band, S_x, Omega, fig_ax=None, file_name="Bloch_k_vs_k"):
    fig, ax = get_fig_ax(fig_ax)

    if fig_ax is None:
        ax.set_title(r"$\Omega = %.2fE_R$" % Omega)

    ax.set_xlabel(r"$k$ / rad")
    ax.set_ylabel(r"$\Psi_k$ / $a^{-1/2}$")
    ax.axhline(0, color='black', ls='--')
    ax.plot(k_arr, np.real(Bloch_k[:, S_band, S_x]), color='blue')
    ax.fill_between(k_arr, np.real(Bloch_k[:, S_band, S_x]), np.real(Bloch_k[:, S_band, S_x])+np.imag(Bloch_k[:, S_band, S_x]), color='red')

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot(file_name)
        plt.show()


def plot_Wannier_vs_x(x_arr, Wannier, S_band, Omega, fig_ax=None, file_name="Wannier_vs_x"):
    fig, ax = get_fig_ax(fig_ax)

    if fig_ax is None:
        ax.set_title(r"$\Omega = %.2fE_R$" % Omega)

    # ax.set_xlabel(r"$x$ / $a$")
    ax.set_xlabel(r"$x$ / $a_0$")
    # ax.set_ylabel(r"$W_R(x)$ / $a^{-1/2}$")
    ax.set_ylabel(r"$W_R(x)$ / $a_0^{-1/2}$")
    ax.axhline(0, color='black', ls='--')
    ax.plot(x_arr, Wannier[S_band, :], color='black')

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot(file_name)
        plt.show()


def plot_log_abs2_Wannier_vs_x(x_arr, abs2_Wannier, S_band, Omega, fig_ax=None, file_name="log_abs2_Wannier_vs_x"):
    fig, ax = get_fig_ax(fig_ax)

    if fig_ax is None:
        ax.set_title(r"$\Omega = %.2fE_R$" % Omega)

    ax.set_xlabel(r"$x$ / $a$")
    ax.set_ylabel(r"$|W_R(x)|^2$ / $a^{-1}$")
    ax.axhline(0, color='black', ls='--')
    ax.plot(x_arr, abs2_Wannier[S_band, :], color='black')
    ax.set_yscale("log")
    # if x_arr[0] < -2 and x_arr[-1] > 2:
    #     ax.set_xlim(-2, 2)
    ax.set_ylim(1e-8, 9)

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot(file_name)
        plt.show()


def plot_EDR(k_arr, E_k, Omega, N_band=3, fig_ax=None, file_name="EDR"):
    fig, ax = get_fig_ax(fig_ax)
    fig.set_size_inches(8, 5)

    ax.set_title(r"$\Omega = %.2fE_R$" % Omega)
    ax.xlabel(r"$k$ / rad")
    ax.ylabel(r"$E$ / $E_R$")
    for i in range(N_band):
        ax.plot(k_arr, E_k[:, i], color="blue")

    if fig_ax is None:
        set_plot_defaults(fig, ax)
        save_plot(file_name)
        plt.show()
# END OF SINGLE PLOTS


# COMPOSITE PLOTS
def plotN_Wannier_vs_x(x_arr, Wannier, S_band_arr, Omega, file_name="N_Wannier_vs_x"):
    fig, ax = plt.subplots(3, 3)
    fig.set_size_inches(18, 10)

    title = r"$\Omega = %.2fE_R$, S_band= " % Omega
    for S_band in S_band_arr:
        title += r"%i, " % S_band
    title = title[:-2]

    fig.suptitle(title)

    for idx_row in range(3):
        for idx_col in range(3):
            plot_Wannier_vs_x(x_arr, Wannier, S_band_arr[3*idx_row+idx_col], Omega, (fig, ax[idx_row, idx_col]))

    if len(S_band_arr) > 9:
        print("WARNING: Only first 9 Wannier states plotted.")

    set_plot_defaults(fig, ax)
    save_plot(file_name)
    plt.show()


def plotN_log_abs2_Wannier_vs_x(x_arr, abs2_Wannier, S_band_arr, Omega, fig_ax=None, file_name="N_log_abs2_Wannier_vs_x"):
    fig, ax = plt.subplots(3, 3)
    fig.set_size_inches(18, 10)

    title = r"$\Omega = %.2fE_R$, S_band= " % Omega
    for S_band in S_band_arr:
        title += r"%i, " % S_band
    title = title[:-2]

    fig.suptitle(title)

    for idx_row in range(3):
        for idx_col in range(3):
            plot_log_abs2_Wannier_vs_x(x_arr, abs2_Wannier, S_band_arr[3*idx_row+idx_col], Omega, (fig, ax[idx_row, idx_col]))

    if len(S_band_arr) > 9:
        print("WARNING: Only first 9 Wannier states plotted.")

    set_plot_defaults(fig, ax)
    save_plot(file_name)
    plt.show()


def plotN_Wannier_and_log_abs2_Wannier_vs_x(x_arr, Wannier, abs2_Wannier, Omega, fig_ax=None, file_name="N_Wannier_and_log_abs2_Wannier_vs_x"):
    """Plots Wannier functions on the left column and squared absolute value of
    Wannier functions (log scale) on the right column of the four lowest energy
    (Bloch) bands."""
    fig, ax = plt.subplots(4, 2)
    fig.set_size_inches(14, 10)

    fig.suptitle(r"$\Omega = %.2fE_R$" % Omega)

    for idx_band in range(4):
        plot_Wannier_vs_x(x_arr, Wannier, idx_band, Omega, (fig, ax[idx_band, 0]))
        plot_log_abs2_Wannier_vs_x(x_arr, abs2_Wannier, idx_band, Omega, (fig, ax[idx_band, 1]))

    set_plot_defaults(fig, ax)
    save_plot(file_name)
    plt.show()
# END OF COMPOSITE PLOTS
# --------------- END OF PLOTTING ---------------
