from math import pi
import numpy as np

from QOPack.Utility import *
from QOPack.RamanTimeDepDetuning.AdiabaticPumping import *


# @profile
def main():
    # PARAMETERS
    N_state = 3
    # band_num = 2  # Currently, only band_num=2 is supported.

    # STATIC SYSTEM PARAMETERS
    epsilon = np.array([0.25, 0.00])
    t_pm = np.array([1.00, 1.00])
    t_0alpha = np.array([0.15, 0.00])
    tn_alpha = np.array([0.00, 0.00])

    gamma = np.zeros(3, dtype=np.float64)
    gamma[-1] = 0.  # pi / 2.
    gamma[0] = -pi / 2.  # 0.
    gamma[1] = gamma[-1]
    # END OF STATIC SYSTEM PARAMETERS

    # SYSTEM MODULATION PARAMETERS
    # Note that other modulation schemes are possible.
    t_bar = np.array([1.00, 0.25])
    epsilon_bar = np.array([0.15, 0.30])
    t_0_bar = np.array([0.05, 0.05])

    T_pump = 100.

    # [t_pm, epsilon, t_0alpha]
    # modulate_parameters = np.array([True, True, False])
    modulate_parameters = np.array([True, False, True])

    select_params = np.array([2, 0])
    # END OF SYSTEM MODULATION PARAMETERS

    k_span = np.array([-np.pi, np.pi])
    time_span = np.array([-0.1, 0.9])  # np.array([0., 3.])

    bar0_span = np.array([0.00, 0.40])
    bar1_span = np.array([0.00, 0.40])

    # (3, 2); 3 -> [t_pm, epsilon, t_0alpha]; 2 -> [-, +] or [s, p]
    select_bar0 = np.array([1, 0])
    select_bar1 = np.array([1, 1])

    N_k = 100  # 1000
    N_time = 398  # 998

    N_bar0 = 40  # 100
    N_bar1 = 38  # 98

    k_selected = 0.5 * np.pi  # Used for Berry phase
    time_selected = 0.5  # Used for Zak phase

    colors = ['red', 'black', 'blue', 'green', 'purple', 'magenta', 'lime']
    # END OF PARAMETERS

    Chern_numbers = calc_CRM_Chern_numbers(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, (-pi, pi), (-0.1, 0.9), modulate_parameters, 5000, N_time)
    print("Chern numbers:", Chern_numbers)

    # full_Chern_arr(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, k_span, time_span, modulate_parameters, bar0_span, bar1_span, select_bar0, select_bar1, N_k, N_time, N_bar0, N_bar1)
    # full_Berry_arr(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, k_span, time_span, modulate_parameters, bar0_span, bar1_span, select_bar0, select_bar1, N_k, N_time, N_bar0, N_bar1, k_selected)
    # full_exact_CRM_Zak_arr_t(Chern_numbers[0], N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, T_pump, modulate_parameters, select_params, time_selected, colors, N_k, N_time)
    full_multiple_exact_CRM_Zak_arr_t(Chern_numbers[0], N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, T_pump, modulate_parameters, select_params, time_selected, colors, N_k, N_time)
    # full_exact_CRM_Energy_kt(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, T_pump, modulate_parameters, time_selected, colors, N_k, N_time)

    save_parameter_TXT()


if __name__ == "__main__":
    main()
