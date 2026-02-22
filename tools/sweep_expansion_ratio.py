# This file sweeps expansion ratio to find optimum O/F
# Author: Django van der Plas

import numpy as np
import matplotlib.pyplot as plt
import cea

# -------------------------
# Standard atmosphere (ISA) pressure vs altitude
# -------------------------
def isa_pressure_pa(h_m: float) -> float:
    """
    Standard atmosphere pressure (Pa) vs altitude (m)

    :param h_m: altitude (m):
    :return: pressure (Pa)
    """
    g0 = 9.80665
    R = 287.05287  # J/(kg·K)

    # Layer base values (1976 std atmosphere)
    # Each tuple: (h_base, T_base, p_base, lapse_rate)
    # Units: m, K, Pa, K/m
    layers = [
        (0.0,     288.15, 101325.0,   -0.0065),
        (11000.0, 216.65, 22632.06,    0.0),
        (20000.0, 216.65, 5474.889,    0.001),
        (32000.0, 228.65, 868.0187,    0.0028),
        (47000.0, 270.65, 110.9063,    0.0),
        (51000.0, 270.65, 66.93887,   -0.0028),
        (71000.0, 214.65, 3.956420,   -0.002),
    ]

    h = float(np.clip(h_m, 0.0, 86000.0))

    # Find layer
    for i in range(len(layers) - 1):
        h_b, T_b, p_b, L_b = layers[i]
        h_next = layers[i + 1][0]
        if h < h_next:
            break
    else:
        h_b, T_b, p_b, L_b = layers[-1]

    dh = h - h_b

    if abs(L_b) < 1e-12:
        # Isothermal
        return p_b * np.exp(-g0 * dh / (R * T_b))
    else:
        # Gradient
        T = T_b + L_b * dh
        return p_b * (T / T_b) ** (-g0 / (R * L_b))


# -------------------------
# User / engine setup
# -------------------------
altitude_m = 0.0             # altitude (m)
pc_bar = 34.0                # chamber pressure (bar)
contraction_ratio = 16.0     # subsonic area ratio (-)
of_ratio = 4.31              # choose O/F (-)
eta_cstar = 0.80             # c* efficiency (-)

# Propellants: 80% ethanol + 20% water fuel, N2O oxidizer
reac_names = [b"C2H5OH(L)", b"H2O", b"N2O"]
fuel_weights = np.array([0.8, 0.2, 0.0])
oxidant_weights = np.array([0.0, 0.0, 1.0])
T_reactant = np.array([298.15, 298.15, 298.15])  # K

# Expansion ratio sweep
eps_grid = np.linspace(2.0, 40.0, 78)  # <-- adjust range/resolution

# -------------------------
# Build mixtures & solver once
# -------------------------
reac = cea.Mixture(reac_names)
prod = cea.Mixture(reac_names, products_from_reactants=True)

solver = cea.RocketSolver(prod, reactants=reac)
sol = cea.RocketSolution(solver)

# Convert altitude -> ambient pressure
pamb_pa = isa_pressure_pa(altitude_m)
pamb_bar = pamb_pa / 1e5

# Convert O/F to weights and compute chamber enthalpy (normalized by R)
weights = reac.of_ratio_to_weights(oxidant_weights, fuel_weights, of_ratio)
hc = reac.calc_property(cea.ENTHALPY, weights, T_reactant) / cea.R

g0 = 9.80665

# Storage
Isp_amb_s = np.full_like(eps_grid, np.nan, dtype=float)
Cf_amb = np.full_like(eps_grid, np.nan, dtype=float)
T_chamber = np.full_like(eps_grid, np.nan, dtype=float)

# -------------------------
# Sweep eps
# -------------------------
for i, eps in enumerate(eps_grid):
    # Solve with this expansion ratio; use pressure-ratio consistent with ambient
    # (Pc / Pexit) based on pamb. This gives a consistent back-pressure target.
    pi_p = pc_bar / pamb_bar

    solver.solve(
        sol,
        weights,
        pc_bar,
        pi_p,
        subar=[contraction_ratio],
        supar=[float(eps)],
        hc=hc,
        iac=True
    )

    P = np.array(sol.P)  # bar
    ae_at = np.array(sol.ae_at)  # Ae/At per station
    Cf = np.array(sol.coefficient_of_thrust)
    cstar = np.array(sol.c_star)
    T = np.array(sol.T)

    # Pick the station whose area ratio is closest to this eps
    j = int(np.argmin(np.abs(ae_at - eps)))

    # Ambient-correct thrust coefficient:
    # Cf_amb = Cf_vac - (pamb/pc)*eps
    cf_amb = float(Cf[j] - (pamb_bar / pc_bar) * eps)

    # "Ambient Isp" proxy from Cf_amb and c*
    isp_amb = cf_amb * float(cstar[j]) / g0

    Cf_amb[i] = cf_amb
    Isp_amb_s[i] = isp_amb
    T_chamber[i] = float(T[0])  # chamber station is usually first

# Find optimum
k = int(np.nanargmax(Isp_amb_s))
eps_opt = float(eps_grid[k])
isp_opt = float(Isp_amb_s[k])

print(f"Altitude: {altitude_m:.0f} m")
print(f"Pamb: {pamb_bar:.4f} bar")
print(f"Pc: {pc_bar:.2f} bar, O/F: {of_ratio:.3f}")
print(f"Optimal eps (max Isp_amb): {eps_opt:.3f}")
print(f"Max Isp_amb (ideal, from Cf_amb*c*/g0): {isp_opt:.2f} s")
print(f"Real Isp using c* efficiency {eta_cstar:.2f}: "
      f"{isp_opt * eta_cstar:.2f} s")

# -------------------------
# Plot
# -------------------------
fig, ax1 = plt.subplots()
ax1.plot(eps_grid, Isp_amb_s, label="Isp_amb (s)")
ax1.set_xlabel("Expansion ratio eps = Ae/At")
ax1.set_ylabel("Ambient Isp (s)")
ax1.grid(True)

# Mark optimum
ax1.axvline(eps_opt, linestyle="--")
ax1.text(eps_opt, isp_opt, f"  opt eps={eps_opt:.2f}", va="bottom")

plt.title(f"Expansion Ratio Sweep @ {altitude_m:.0f} m (Pamb={pamb_bar:.3f} bar)")
plt.show()