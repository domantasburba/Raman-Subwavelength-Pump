import numpy as np

from QOPack.Utility import *
# from QOPack.RamanTimeDepDetuning.ZakPhase import full_Wannier_pm
from QOPack.RamanTimeDepDetuning.ReverseMapping import CRM2Full


def main():
    # PARAMETERS
    N_state = 3

    # STATIC SYSTEM PARAMETERS
    t_pm = np.array([1.00, 1.00])
    epsilon = np.array([0.00, 0.00])
    t_0alpha = np.array([0.10, 0.00])
    tn_alpha = np.array([0.15, -0.05])

    gamma = np.zeros(3, dtype=np.float64)
    gamma[-1] = 0.  # np.pi / 2.
    gamma[0] = -np.pi / 2.  # 0.
    gamma[1] = gamma[-1]
    # END OF STATIC SYSTEM PARAMETERS

    # SYSTEM MODULATION PARAMETERS
    t_bar = np.array([1.00, 0.85])
    epsilon_bar = np.array([0.10, 0.40])
    t_0_bar = np.array([0.15, 0.15])
    # END OF SYSTEM MODULATION PARAMETERS

    modulate_parameters = np.array([True, True, False])

    # x_max = 10.

    # N_k = 1000
    # N_x = 10001
    # END OF PARAMETERS

    # [Omega, delta_p, omega], [delta_pm_bar, omega_bar, delta_0_bar], [tn_alpha]
    print(CRM2Full(t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, N_state, modulate_parameters))
    # full_Wannier_pm(epsilon, t_pm, t_0alpha, tn_alpha, gamma, (-np.pi, np.pi), N_state, N_k, x_max, N_x)

    save_parameter_TXT()


if __name__ == "__main__":
    main()
