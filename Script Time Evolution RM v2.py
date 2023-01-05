from math import pi
import numpy as np

from QOPack.Math import *
from QOPack.Utility import *
from QOPack.RamanTimeDepDetuning.ZakPhase import *
import QOPack.RamanTimeDepDetuning.TimeEvolutionRM as RM


def main():
    """Janos k. Asboth, Laszlo Oroszlany, Andras Palyi, "A Short Course on
    Topological Insulators: Band Structure and Edge States in One and Two
    Dimensions", Springer.  Taken from SFN.pdf, page 60."""

    # PARAMETERS
    T_pump = 200
    time_span = (0., 1.0*T_pump)
    N_time = 61

    # initial_condition selects which initial condition to use.
    # Options: "Centered Single Sublattice", "Centered Single Lattice", "Real
    # Centered Eigen", "Centered Eigen", "Selected Eigen", "Centered Gaussian",
    # "Plane Constructive".
    # TODO: "Centered Eigen" only works for complex functions.
    initial_condition = "Real Centered Eigen"

    # MODULATION PARAMETERS
    u_bar = 0.
    v_bar = 1.
    w_0 = 1.

    J_0 = pi / (1*T_pump)

    # Options: "Ian", "Book".
    modulation_scheme = "Book"
    # END OF MODULATION PARAMETERS

    N_lat = 500
    N_period = 231

    adiabaticLaunching = False  # DISABLED
    tau_adiabatic = 10.*T_pump

    time_selected = 1.*T_pump

    colors = ["red", "black", "blue", "green", "purple", "magenta", "lime"]
    # END OF PARAMETERS

    # RM.routine_RM_Wannier_center(u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_lat, N_period)
    # RM.routine_RM_time_spectrum(u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_lat, N_period)
    # RM.routine_RM_time_evolution(time_span, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, initial_condition, adiabaticLaunching, tau_adiabatic, time_selected, N_lat, N_time, N_period)
    # RM.Floquet_band_structure(u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, N_lat, N_period)
    RM.routine_RM_all(time_span, u_bar, v_bar, w_0, J_0, T_pump, modulation_scheme, initial_condition, adiabaticLaunching, tau_adiabatic, time_selected, N_lat, N_time, N_period)

    save_parameter_TXT()


if __name__ == "__main__":
    main()
