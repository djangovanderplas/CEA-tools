# This script sizes the throat of an LRE
# Author: Django van der Plas

import numpy as np
import cea

# -------------------------
# Inputs you choose / know
# -------------------------
F = 600.0                      # thrust (N)
eta_cstar = 0.80               # c* efficiency (-)
pc_bar = 34.0                  # chamber pressure (bar)
pamb_bar = 1.01325             # ambient pressure (bar)
eps = 5.5696                   # Ae/At (-)
contraction_ratio = 16.0       # subsonic area ratio (finite combustor contraction)

of_ratio = 2.1                 # mixture ratio (-)

reac_names = [b"C2H5OH(L)", b"H2O", b"N2O"]
fuel_weights = np.array([0.8, 0.2, 0.0])
oxidant_weights = np.array([0.0, 0.0, 1.0])
T_reactant = np.array([298.15, 298.15, 298.15])

# -------------------------
# CEA setup
# -------------------------
reac = cea.Mixture(reac_names)
prod = cea.Mixture(reac_names, products_from_reactants=True)

weights = reac.of_ratio_to_weights(oxidant_weights, fuel_weights, of_ratio)
hc = reac.calc_property(cea.ENTHALPY, weights, T_reactant) / cea.R

solver = cea.RocketSolver(prod, reactants=reac)
sol = cea.RocketSolution(solver)

pi_p = pc_bar / pamb_bar

solver.solve(
    sol,
    weights,
    pc_bar,
    pi_p,
    subar=[contraction_ratio],
    supar=[eps],
    hc=hc,
    iac=True
)

# -------------------------
# Pick station closest to ambient pressure
# -------------------------
P = np.array(sol.P)  # bar
idx = int(np.argmin(np.abs(P - pamb_bar)))

Cf = float(np.array(sol.coefficient_of_thrust)[idx])
cstar_ideal = float(np.array(sol.c_star)[idx])  # m/s

# -------------------------
# Throat sizing
# -------------------------
pc_Pa = pc_bar * 1e5  # bar -> Pa

At = F / (Cf * pc_Pa)                 # m^2
dt = np.sqrt(4.0 * At / np.pi)        # m

# Mass flow needed to achieve Pc given c* efficiency
mdot = pc_Pa * At / (eta_cstar * cstar_ideal)  # kg/s

print(f"Exit station index: {idx}, P_exit ~ {P[idx]:.4f} bar")
print(f"Cf = {Cf:.4f}")
print(f"c* (ideal) = {cstar_ideal:.1f} m/s, c* (eff) = {eta_cstar*cstar_ideal:.1f} m/s")
print(f"At = {At*1e6:.2f} mm^2")
print(f"dt = {dt*1e3:.2f} mm")
print(f"mdot = {mdot:.3f} kg/s")