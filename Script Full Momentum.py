from math import pi
import numpy as np

from QOPack.Utility import *
# from QOPack.RamanTimeDepDetuning.OverlapTunneling import calc_1Omega_noHO_overlap_tunneling
from QOPack.RamanTimeDepDetuning.OverlapTunneling import *
from QOPack.RamanTimeDepDetuning.AdiabaticPumping import calc_CRM_Chern_numbers
from QOPack.RamanTimeDepDetuning.MomentumFull import *


# @fun_time
# @profile
def main():
    # PARAMETERS
    # PHYSICAL PARAMETERS
    # STATIC SYSTEM PARAMETERS
    N_state = 3
    N_band = 2

    # Number of lattice sites (excluding internal degrees of freedom).
    N_lat = 300
    N_lat_Floquet = 400

    Omega = 3.5

    max_sep = 1.0  # Farthest tunneling elements and overlap integrals that are calculated correspond to wavefunctions separated by max_sep
    W_W_overlap_arr, W_H_W_arr = calc_1Omega_noHO_overlap_tunneling(Omega, max_sep, N_state, N_band)

    omega = W_H_W_arr[1, 1, 0] - W_H_W_arr[0, 0, 0]

    delta_p = np.zeros(3, dtype=np.float64)
    delta_p[-1] = -0.256
    delta_p[0] = 0.20
    delta_p[1] = 0.24

    gamma = np.zeros(3, dtype=np.float64)
    gamma[-1] = 0.  # np.pi / 2.
    gamma[0] = -pi / 2.  # 0.
    gamma[1] = gamma[-1]
    # END OF STATIC SYSTEM PARAMETERS

    # TIME EVOLUTION PARAMETERS
    T_pump = 700.  # Must be much larger than other system timescales to ensure adiabaticity (e.g., inverse band gap or inverse omega).

    time_span = (0., 1*T_pump)

    # initial_condition selects which initial condition to use.
    # Options: "Centered Single Sublattice", "Centered Single Lattice", "Real Centered Eigen", "Centered Eigen", "Selected Eigen", "Centered Gaussian", "Plane Constructive".
    initial_condition = "Centered Eigen"

    # If initial condition is a Gaussian, sigma_Gaussian is used as the standard
    # deviation.
    sigma_Gaussian = 5

    delta_pm_bar = np.array([0.25, 0.20])
    t_0_diff = delta_p[0]*(W_W_overlap_arr[0, 0, 1] - W_W_overlap_arr[1, 1, 1])
    omega_bar = np.array([omega, 2.2*t_0_diff])  # omega_bar[1] < omega_bar[0] or solution will not converge.
    # omega_bar = np.array([omega+2.0*t_0_diff, 0.2*t_0_diff])  # omega_bar[1] < omega_bar[0] or solution will not converge.
    delta_0_bar = np.array([-0.24, 0.24])

    # [delta^(pm), omega, delta^0]
    # delta^(pm) should always be modulated, alongside omega or delta^0 or both.
    modulate_parameters = np.array([True, True, False])
    # modulate_parameters = np.array([True, False, True])

    # Only used for time evolution to slowly turn on driving instead of instantaneously.
    adiabaticLaunching = False  # TODO: DO NOT TURN ON, broken.
    tau_adiabatic = T_pump
    # END OF TIME EVOLUTION PARAMETERS

    # CRM PARAMETERS
    # Only used if CRM calculations are made (e.g., comparing Full model
    # with CRM model).
    # Large frequency limit is assumed.
    # First order correction WILL FAIL FOR SMALL FREQUENCIES (small omega or omega_bar[0]).
    addFirstOrder = False  # If True, add first order Floquet correction to Hamiltonian.
    # END OF CRM PARAMETERS
    # END OF PHYSICAL PARAMETERS

    # COMPUTATIONAL PARAMETERS
    # These are not physical parameters, they determine how approximate the
    # numerical calculations shall be. Computational parameters should be carefully
    # chosen to ensure convergence of physical results.
    N_x = 10001
    x_max = 10.

    N_time = 100
    N_period = 51
    # END OF COMPUTATIONAL PARAMETERS

    # PLOTTING PARAMETERS
    # These parameters will not affect the calculations, only the plots.

    # Time spectrum parameters
    # Observe eigenstate S_eigen of system at period_selected.
    period_selected = 1*T_pump  # Must be between 0 and 1 (in the first period).
    time_selected = 1*T_pump
    S_eigen = 0  # N_lat

    # len(colors) >= N_band
    colors = ["red", "black", "blue", "green", "purple", "magenta", "lime"]

    # 2 -> [x, y]; [delta^(pm), omega, delta^0]
    # Should agree with modulate_parameters, used only for plotting parameter path (plot_CRM_parameter_path & plot_Full_parameter_path).
    params_selected = np.array([0, 1])
    # params_selected = np.array([0, 2])
    # END OF PLOTTING PARAMETERS
    # END OF PARAMETERS

    t_pm, epsilon, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar = Full2CRM(Omega, omega, delta_p, delta_pm_bar, omega_bar, delta_0_bar, N_state, printResult=True)
    Chern_numbers = calc_CRM_Chern_numbers(N_state, epsilon, t_pm, t_0alpha, tn_alpha, t_bar, epsilon_bar, t_0_bar, gamma, (-pi, pi), (-0.1, 0.9), modulate_parameters, 5000, N_time)
    print("Chern numbers:", Chern_numbers)

    full_Full_k_Wannier_center(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_x, x_max, max_sep, N_time, time_span, N_period, period_selected, S_eigen, colors, params_selected)
    # compare_Full_momentum_to_real(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, N_x, x_max, max_sep, N_time, time_span, N_period, period_selected, S_eigen, colors, params_selected)
    # compare_Full_k_to_CRM_k(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat, max_sep, N_period, period_selected, colors)
    # Floquet_band_structure(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat_Floquet, max_sep, N_period, period_selected, colors)
    # Floquet_bands_Full_Full2_CRM(Omega, omega, delta_p, gamma, delta_pm_bar, omega_bar, delta_0_bar, T_pump, modulate_parameters, adiabaticLaunching, tau_adiabatic, N_state, N_band, N_lat_Floquet, max_sep, N_period, period_selected, colors)

    save_parameter_TXT()


if __name__ == "__main__":
    main()
