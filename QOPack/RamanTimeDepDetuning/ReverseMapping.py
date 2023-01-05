import numpy as np
from scipy.interpolate import splrep, BSpline
from scipy.optimize import root_scalar
from datetime import datetime
import pathlib

from QOPack.Utility import *
from QOPack.RamanTimeDepDetuning.OverlapTunneling import overlap_tunneling, calc_1Omega_noHO_overlap_tunneling


# TODO: General function that prints all arguments in such a manner.
def print_CRM(t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar):
    print('***************')
    print('t_pm:', t_pm)
    print('epsilon:', epsilon)
    print('t_0alpha:', t_0alpha)
    print('tn_alpha:', tn_alpha)
    print('t_bar:', t_bar)
    print('epsilon_bar:', epsilon_bar)
    print('t_0_bar:', t_0_bar)
    print('***************')


def print_Full(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, tn_alpha):
    print('***************')
    print('Omega:', Omega)
    print('omega:', omega)
    print('delta_p:', delta_p)
    print('delta_pm_bar:', delta_pm_bar)
    print('omega_bar:', omega_bar)
    print('delta_0_bar:', delta_0_bar)
    print('tn_alpha (from CRM):', tn_alpha)
    print('***************')


def save_CRM(t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, save_path=None):
    datetime_now = datetime.now()
    if save_path is None:
        save_path = r"%s/ReverseMapping/%s" % (get_main_dir(), datetime_now.strftime("%Y-%m-%d"))
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    file = open(get_main_path(), "r")
    save = open(r"%s/%s!%s.txt" % (save_path, get_main_name(), datetime_now.strftime("%H.%M.%S")), "w")

    save.write("***************\n")
    save.write("t_pm: [%.6f %.6f]\n" % (t_pm[0], t_pm[1]))
    save.write("epsilon: [%.6f %.6f]\n" % (epsilon[0], epsilon[1]))
    save.write("t_0alpha: [%.6f %.6f]\n" % (t_0alpha[0], t_0alpha[1]))
    save.write("tn_alpha: [%.6f %.6f]\n" % (tn_alpha[0], tn_alpha[1]))
    save.write("t_bar: [%.6f %.6f]\n" % (t_bar[0], t_bar[1]))
    save.write("epsilon_bar: [%.6f %.6f]\n" % (epsilon_bar[0], epsilon_bar[1]))
    save.write("t_0_bar: [%.6f %.6f]\n" % (t_0_bar[0], t_0_bar[1]))
    save.write("***************")

    file.close()
    save.close()


def save_Full(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, tn_alpha, save_path=None):
    datetime_now = datetime.now()
    if save_path is None:
        save_path = r"%s/ReverseMapping/%s" % (get_main_dir(), datetime_now.strftime("%Y-%m-%d"))
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    file = open(get_main_path(), "r")
    save = open(r"%s/%s!%s.txt" % (save_path, get_main_name(), datetime_now.strftime("%H.%M.%S")), "w")

    save.write("***************\n")
    save.write("Omega: %.6f\n" % Omega)
    save.write("omega: %.6f\n" % omega)
    save.write("delta_p: [%.6f %.6f %.6f]\n" % (delta_p[-1], delta_p[0], delta_p[1]))
    save.write("delta_pm_bar: [%.6f %.6f]\n" % (delta_pm_bar[0], delta_pm_bar[1]))
    save.write("omega_bar: [%.6f %.6f]\n" % (omega_bar[0], omega_bar[1]))
    save.write("delta_0_bar: [%.6f %.6f]\n" % (delta_0_bar[0], delta_0_bar[1]))
    save.write("tn_alpha (from CRM): [%.6f %.6f]\n" % (tn_alpha[0], tn_alpha[1]))
    save.write("***************")

    file.close()
    save.close()


def CRM2Full(t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, N_state, modulate_parameters, printResult=False, saveResult=False):
    # PARAMETERS
    band_num = 2

    a_0 = 1.
    x_max = 10. * a_0
    # Omega_span = (0.1, 2.5)
    Omega_span = (0.1, 5.0)  # If Omega is out of bounds, program will fail.
    # Omega_span = (5.0, 7.0)
    # Omega_span = (5.0, 10.0)
    max_sep = 1.

    K = 10001
    R = 20
    # R = 40
    # END OF PARAMETERS

    # DERIVED PARAMETERS
    Omega_arr = np.linspace(*Omega_span, R)
    # END OF DERIVED PARAMETERS

    W_W_overlap_arr, HO_exact_overlap_arr, HO_calc_overlap_arr, W_W_alpha_s_arr, HO_exact_alpha_s_arr, HO_calc_alpha_s_arr, W_H_W_arr, chi_H_chi_arr, HO_exact_tunneling_arr = overlap_tunneling(N_state, Omega_span, a_0, K, R, band_num, max_sep, x_max)

    # *************** FINDING STATIC SYSTEM PARAMETERS ***************
    delta_p = np.zeros(3, dtype=np.float64)

    # PART 1: Finding Rabi frequency Omega.
    # (modulate_parameters[2] is False) == False
    # (modulate_parameters[2] == False) == True
    if modulate_parameters[2] == False:
        # If t_0alpha is not modulated

        t, c, k = splrep(Omega_arr, W_W_alpha_s_arr[:, 1])
        alpha_s_spline = BSpline(t, c, k)
        # print(alpha_s_spline(0.1))

        f_Omega = lambda Omega: alpha_s_spline(Omega) - t_0alpha[1] / t_0alpha[0]
        sol = root_scalar(f_Omega, bracket=Omega_span)
        Omega = sol.root
        # print(Omega)
    elif modulate_parameters[1] == False:
        # If epsilon is not modulated

        t, c, k = splrep(Omega_arr, W_H_W_arr[:, 0, 0, 0])
        epsilon_s_spline = BSpline(t, c, k)
        # print(epsilon_s_spline(0.1))

        f_Omega = lambda Omega: epsilon_s_spline(Omega) - epsilon[0]
        sol = root_scalar(f_Omega, bracket=Omega_span)
        Omega = sol.root
    else:
        raise AssertionError(r"Unsupported modulation scheme.")
    # END OF PART 1

    # PART 2: Finding angular frequency omega.
    t, c, k = splrep(Omega_arr, W_H_W_arr[:, 0, 0, 0] - W_H_W_arr[:, 1, 1, 0])
    epsilon_diff_spline = BSpline(t, c, k)
    omega = epsilon[0] - epsilon[1] - epsilon_diff_spline(Omega)
    # END OF PART 2

    # PART 3: Finding delta^(0)
    t, c, k = splrep(Omega_arr, W_W_overlap_arr[:, 0, 0, 1])
    G_ss_spline = BSpline(t, c, k)
    delta_p[0] = t_0alpha[0] / G_ss_spline(Omega)
    # END OF PART 3

    # PART 4: Finding delta^(pm)
    t, c, k = splrep(Omega_arr, W_W_overlap_arr[:, 0, 1, 1])
    # Indices switched because of different notation in code and notes.
    G_ps_spline = BSpline(t, c, k)
    # print(G_ps_spline(0.3))
    G_ps = G_ps_spline(Omega)
    delta_p[1] = t_pm[1] / G_ps
    delta_p[-1] = -t_pm[0] / G_ps
    # END OF PART 4
    # *************** END OF FINDING STATIC SYSTEM PARAMETERS ***************


    # *************** FINDING SYSTEM MODULATION PARAMETERS ***************
    delta_pm_bar = t_bar / G_ps

    omega_bar = np.empty(2, dtype=np.float64)
    omega_bar[0] = epsilon_bar[0] - epsilon_diff_spline(Omega)
    omega_bar[1] = epsilon_bar[1]

    t, c, k = splrep(Omega_arr, W_W_overlap_arr[:, 1, 1, 1])
    G_pp_spline = BSpline(t, c, k)
    G_ss_pp_diff = G_ss_spline(Omega) - G_pp_spline(Omega)
    delta_0_bar = t_0_bar / G_ss_pp_diff
    # *************** END OF FINDING SYSTEM MODULATION PARAMETERS ***************

    # FINDING TN_ALPHA
    # NOTE: If all other CRM parameters are given, then tn_alpha is fixed.
    # In other words, not all possible sets of CRM parameters can be mapped to our Full system.
    # TODO: Spline is less accurate than actually calculating the value.
    t, c, k = splrep(Omega_arr, W_H_W_arr[:, 0, 0, 1])
    tn_s_spline = BSpline(t, c, k)
    t, c, k = splrep(Omega_arr, W_H_W_arr[:, 1, 1, 1])
    tn_p_spline = BSpline(t, c, k)

    tn_s = tn_s_spline(Omega)
    tn_p = tn_p_spline(Omega)
    tn_alpha = np.array([tn_s, tn_p])
    # END OF FINDING TN_ALPHA

    # PRINT RESULT
    if printResult:
        print_Full(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, tn_alpha)
    # END OF PRINT RESULT

    # SAVE RESULT
    if saveResult:
        save_Full(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, tn_alpha)
    # END OF SAVE RESULT

    return Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, tn_alpha


def Full2CRM(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, N_state, printResult=False):
    # PARAMETERS
    N_band = 2

    x_max = 10.
    max_sep = 1.

    N_x = 10001
    # N_x = 30001
    # END OF PARAMETERS

    W_W_overlap_arr, W_H_W_arr = calc_1Omega_noHO_overlap_tunneling(Omega, max_sep, N_state, N_band, x_max=x_max, N_x=N_x)

    # *************** FINDING STATIC SYSTEM PARAMETERS ***************
    # PART 1: Finding intrachain couplings t_pm
    t_pm = np.empty(2, dtype=np.float64)
    t_pm[0] = -delta_p[-1] * W_W_overlap_arr[0, 1, 1]
    t_pm[1] = delta_p[1] * W_W_overlap_arr[0, 1, 1]
    # END OF PART 1

    # PART 2: Finding energy detunings epsilon
    epsilon = np.empty(2, dtype=np.float64)
    epsilon[0] = W_H_W_arr[0, 0, 0]
    epsilon[1] = W_H_W_arr[1, 1, 0] - omega
    # END OF PART 2

    # PART 3: Finding interchain couplings t_0alpha
    t_0alpha = np.empty(2, dtype=np.float64)
    t_0alpha[0] = delta_p[0] * W_W_overlap_arr[0, 0, 1]
    t_0alpha[1] = delta_p[0] * W_W_overlap_arr[1, 1, 1]
    # END OF PART 3

    # PART 4: Finding natural tunneling tn_alpha
    tn_alpha = np.empty(2, dtype=np.float64)
    tn_alpha[0] = W_H_W_arr[0, 0, 1]
    tn_alpha[1] = W_H_W_arr[1, 1, 1]
    # END OF PART 4
    # *************** END OF FINDING STATIC SYSTEM PARAMETERS ***************


    # *************** FINDING SYSTEM MODULATION PARAMETERS ***************
    # PART 1: Finding t_bar
    t_bar = np.empty(2, dtype=np.float64)
    t_bar[0] = delta_pm_bar[0] * W_W_overlap_arr[0, 1, 1]
    t_bar[1] = delta_pm_bar[1] * W_W_overlap_arr[0, 1, 1]
    # END OF PART 1

    # PART 2: Finding epsilon_bar
    epsilon_bar = np.empty(2, dtype=np.float64)
    epsilon_bar[0] = W_H_W_arr[0, 0, 0] - W_H_W_arr[1, 1, 0] + omega_bar[0]
    epsilon_bar[1] = omega_bar[1]
    # END OF PART 2

    # PART 3: Finding t_0_bar
    t_0_bar = np.empty(2, dtype=np.float64)
    t_0_bar[0] = delta_0_bar[0] * (W_W_overlap_arr[0, 0, 1] - W_W_overlap_arr[1, 1, 1])
    t_0_bar[1] = delta_0_bar[1] * (W_W_overlap_arr[0, 0, 1] - W_W_overlap_arr[1, 1, 1])
    # END OF PART 3
    # *************** END OF FINDING SYSTEM MODULATION PARAMETERS ***************

    # PRINT RESULT
    if printResult:
        print_CRM(t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar)
    # END OF PRINT RESULT

    return t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar


def check_mappings_from_Full(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, N_state):
    """Check if g(f(x)) = x, where g(x) = f^{-1}(x)."""

    t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar = Full2CRM(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, N_state)
    Omega1, omega1, delta_p1, delta_pm_bar1, omega_bar1, delta_0_bar1, tn_alpha1 = CRM2Full(t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, N_state)
    print_Full(Omega1, omega1, delta_p1, delta_pm_bar1, omega_bar1, delta_0_bar1, tn_alpha1)
