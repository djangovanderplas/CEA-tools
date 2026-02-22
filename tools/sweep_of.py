# This script sweeps the mixture ratio to find an optimal
# Author: Django van der Plas

import numpy as np
import matplotlib.pyplot as plt
import cea

# -------------------------
# User engine / propellant setup
# -------------------------
reac_names = [b"C2H5OH(L)", b"H2O", b"N2O"]

fuel_weights = np.array([0.8, 0.2, 0.0])      # 80% ethanol, 20% water
oxidant_weights = np.array([0.0, 0.0, 1.0])   # 100% N2O

T_reactant = np.array([298.15, 298.15, 298.15])  # K

pc_bar = 34.0   # Chamber pressure (bar)
p_amb_bar = 1.01325     # Ambient pressure (bar)

contraction_ratio = 16.0   # subsonic area ratio (-)
expansion_ratio = 6.94     # expansion ratio (Ae/At)

pi_p = pc_bar / p_amb_bar
subar = [contraction_ratio]
supar = [expansion_ratio]

g0 = 9.80665  # (m/s^2)

# -------------------------
# Build mixtures & solver once
# -------------------------
reac = cea.Mixture(reac_names)
prod = cea.Mixture(reac_names, products_from_reactants=True)

solver = cea.RocketSolver(prod, reactants=reac)
solution = cea.RocketSolution(solver)

# -------------------------
# Sweep O/F
# -------------------------
of_grid = np.linspace(0.1, 8.0, 76)
isp_s = np.full_like(of_grid, np.nan, dtype=float)
tc = np.full_like(of_grid, np.nan, dtype=float)  # chamber temperature

for i, of_ratio in enumerate(of_grid):
    weights = reac.of_ratio_to_weights(oxidant_weights, fuel_weights, of_ratio)
    hc = reac.calc_property(cea.ENTHALPY, weights, T_reactant) / cea.R

    solver.solve(
        solution,
        weights,
        pc_bar,
        pi_p,
        subar=subar,
        supar=supar,
        hc=hc,
        iac=True
    )

    P = np.array(solution.P)         # bar
    Isp_ms = np.array(solution.Isp)  # m/s
    T = np.array(solution.T)         # K

    # Station closest to ambient pressure
    idx = int(np.argmin(np.abs(P - p_amb_bar)))

    isp_s[i] = Isp_ms[idx] / g0
    tc[i] = T[0]  # chamber temperature (typically first station)

# -------------------------
# Find optimal O/F
# -------------------------
k = int(np.nanargmax(isp_s))
of_opt = float(of_grid[k])
isp_opt = float(isp_s[k])
tc_opt = float(tc[k])

print(f"Optimal O/F (max Isp at ~Pamb): {of_opt:.4f}")
print(f"Max Isp: {isp_opt:.2f} s")
print(f"Chamber T at optimum: {tc_opt:.1f} K")

# -------------------------
# Plot (dual axis) + optimum marker
# -------------------------
fig, ax1 = plt.subplots()

# Isp curve
ax1.plot(of_grid, isp_s)
ax1.set_xlabel("O/F (mass)")
ax1.set_ylabel("Isp (s)")
ax1.grid(True)

# Mark optimum O/F
ax1.axvline(of_opt, linestyle="--")
ax1.plot(of_opt, isp_opt, marker="o")
ax1.text(of_opt, isp_opt, f"  opt O/F={of_opt:.2f}\n  Isp={isp_opt:.1f}s", va="top")

# Temperature on right axis (red)
ax2 = ax1.twinx()
ax2.plot(of_grid, tc, color="red")
ax2.set_ylabel("Chamber Temperature (K)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("O/F Sweep: Isp and Combustion Temperature")
plt.show()