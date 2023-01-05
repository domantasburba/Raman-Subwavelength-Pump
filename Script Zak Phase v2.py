from math import pi
import numpy as np
import matplotlib.pyplot as plt

from QOPack.Utility import *
from QOPack.RamanTimeDepDetuning.ZakPhase import *

# plt.style.use("./matplotlibrc")
plt.style.use("default")

# @profile
# @fun_time
def main():
    # PARAMETERS
    N_state = 3
    # band_num = 2  # Currently, only band_num=2 is supported.

    # STATIC SYSTEM PARAMETERS
    t_pm = np.array([1.00, 1.00])
    epsilon = np.array([0.25, 0.00])
    t_0alpha = np.array([0.15, 0.00])
    tn_alpha = np.array([0.00, 0.00])

    gamma = np.zeros(3, dtype=np.float64)
    gamma[-1] = 0.  # pi / 2.
    gamma[0] = -pi / 2.  # 0.
    gamma[1] = gamma[-1]
    # END OF STATIC SYSTEM PARAMETERS

    # SYSTEM PLOT PARAMETERS
    x_param_diff_span = np.array([-0.50, 0.50])
    y_param_diff_span = np.array([-0.50, 0.50])
    # y_param_diff_span = np.array([-0.50, 0.50])  # np.array([-0.20, 0.20])

    # t_avg should exceed some critical value. Otherwise, Zak phase will be zero everywhere (trivial case).
    # epsilon_avg and t_0_avg should not affect the results since only the difference matters.
    x_param_avg = 0.20
    y_param_avg = 0.30

    # 2 -> [x, y]; [t_pm, epsilon, t_0alpha]
    select_params = np.array([2, 0])

    N_x = 300  # 200
    N_y = 298  # 198
    # END OF SYSTEM PLOT PARAMETERS

    k_span = np.array([-pi, pi])

    N_k = 1000  # 1000

    # gap_string selects which gap to calculate.
    # Options: "Direct", "Indirect"
    gap_string = "Indirect"

    # withIdentity determines whether to remove the identity part of the Hamiltonian.
    # Basically amounts to a k-dependent energy origin shift.
    withIdentity = True

    # BULK CALCULATION PARAMETERS
    x_param_avg_bulk_span = (-0.5, 0.5)
    y_param_avg_bulk_span = (-0.5, 0.5)

    N_bulk1 = 11
    N_bulk2 = 11
    # END OF BULK CALCULATION PARAMETERS
    # END OF PARAMETERS

    # *************** SINGLE ***************
    # TODO: Figure out if prod_U has to be imaginary. prod_U isn't imaginary, might be a problem.
    # full_energy_gaps(x_param_diff_span, y_param_diff_span, x_param_avg, y_param_avg, t_pm, epsilon, t_0alpha, tn_alpha, gamma, select_params, N_state, N_k, N_x, N_y, gap_string, withIdentity)
    # TODO: Zak phase calculations do not work for (t_1+t_{-1})/2 = 0, figure out why.
    full_Zak_arr(x_param_diff_span, y_param_diff_span, x_param_avg, y_param_avg, t_pm, epsilon, t_0alpha, tn_alpha, gamma, k_span, select_params, N_state, N_k, N_x, N_y, withIdentity=withIdentity)
    # full_A_k(epsilon, t_pm, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k)

    # save_parameter_TXT()
    # *************** END OF SINGLE ***************

    # *************** BULK ***************
    # bulk_gap_arr(x_param_avg_bulk_span, y_param_avg_bulk_span, N_bulk1, N_bulk2, x_param_diff_span, y_param_diff_span, t_pm, epsilon, t_0alpha, tn_alpha, gamma, N_state, N_k, N_x, N_y, withIdentity)
    # bulk_Zak_arr(x_param_avg_bulk_span, y_param_avg_bulk_span, N_bulk1, N_bulk2, x_param_diff_span, y_param_diff_span, t_pm, epsilon, t_0alpha, tn_alpha, gamma, k_span, N_state, N_k, N_x, N_y, withIdentity)
    # *************** END OF BULK ***************


if __name__ == "__main__":
    main()
