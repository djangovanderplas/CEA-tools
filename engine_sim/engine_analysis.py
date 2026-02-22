import warnings

import astropy.units as u
import cea
import numpy as np
from astropy.units import Quantity
from astropy.units import imperial as i

warnings.filterwarnings("ignore")

try:
    from ambiance import Atmosphere

    amb_working = True
except Exception:
    amb_working = False


class EngineAnalysis:
    """Rocket engine performance analysis using CEA v3 (`cea` package)."""

    def __init__(
        self,
        injector,
        throat_diameter: Quantity,
        exit_diameter: Quantity,
        c_star_efficiency: float,
        cf_efficiency: float,
        nom_ox_massflow: Quantity,
        nom_fuel_massflow: Quantity,
        contraction_ratio: Quantity,
    ):
        self.injector = injector
        self.throat_diameter = throat_diameter
        self.exit_diameter = exit_diameter
        self.c_star_efficiency = c_star_efficiency * u.dimensionless_unscaled
        self.cf_efficiency = cf_efficiency * u.dimensionless_unscaled
        self.nom_ox_massflow = nom_ox_massflow
        self.nom_fuel_massflow = nom_fuel_massflow
        self.SL_pressure = 1.01325 * u.bar
        self.contraction_ratio = contraction_ratio

        self.reac, self.prod, self.solver, self.solution = self.create_cea()
        self.reac_names = [b"O2(L)", b"C2H5OH(L)", b"H2O"]
        self.oxidizer_weights = np.array([1.0, 0.0, 0.0])
        self.fuel_weights = np.array([0.0, 0.8, 0.2])
        self.reactant_temperature = np.array([90.0, 298.15, 298.15])

    def create_cea(self):
        reac = cea.Mixture([b"O2(L)", b"C2H5OH(L)", b"H2O"])
        prod = cea.Mixture([b"O2(L)", b"C2H5OH(L)", b"H2O"], products_from_reactants=True)
        solver = cea.RocketSolver(prod, reactants=reac)
        return reac, prod, solver, cea.RocketSolution(solver)

    @property
    def nom_total_massflow(self):
        return self.nom_ox_massflow + self.nom_fuel_massflow

    @property
    def nom_of_ratio(self):
        return self.nom_ox_massflow / self.nom_fuel_massflow

    @property
    def throat_area(self):
        return np.pi * (self.throat_diameter / 2) ** 2

    @property
    def exit_area(self):
        return np.pi * (self.exit_diameter / 2) ** 2

    @property
    def area_expansion_ratio(self):
        return (self.exit_area / self.throat_area).decompose()

    @property
    def total_efficiency(self):
        return self.c_star_efficiency * self.cf_efficiency

    def _solve(self, p_chamber: Quantity, mixture_ratio: Quantity, eps: float | None = None):
        of_ratio = mixture_ratio.decompose().value
        weights = self.reac.of_ratio_to_weights(self.oxidizer_weights, self.fuel_weights, of_ratio)
        hc = self.reac.calc_property(cea.ENTHALPY, weights, self.reactant_temperature) / cea.R

        eps_value = float(self.area_expansion_ratio.value if eps is None else eps)
        pi_p = p_chamber.to(u.bar).value / self.SL_pressure.to(u.bar).value

        self.solver.solve(
            self.solution,
            weights,
            p_chamber.to(u.bar).value,
            pi_p,
            subar=[float(self.contraction_ratio.value)],
            supar=[eps_value],
            hc=hc,
            iac=True,
        )
        return self.solution

    def get_cstar(self, p_chamber: Quantity, mixture_ratio: Quantity) -> Quantity:
        sol = self._solve(p_chamber, mixture_ratio)
        c_star = float(np.array(sol.c_star)[0]) * (u.m / u.s)
        return (c_star * self.c_star_efficiency).to(u.m / u.s)

    def get_chamber_pressure(self, massflow: Quantity, mixture_ratio: Quantity, threshold=1e-5) -> Quantity:
        delta = 1.0
        p_chamber = 50 * u.bar
        while delta > threshold:
            c_star = self.get_cstar(p_chamber, mixture_ratio)
            new_p_chamber = (c_star * massflow) / self.throat_area
            delta = abs(p_chamber.to(u.bar).value - new_p_chamber.to(u.bar).value)
            p_chamber = new_p_chamber
        return p_chamber.to(u.bar)

    def get_optimal_ambient_pressure(self, chamber_pressure: Quantity):
        self._solve(chamber_pressure, self.nom_of_ratio)
        p = np.array(self.solution.P)
        ae_at = np.array(self.solution.ae_at)
        idx = int(np.argmin(np.abs(ae_at - self.area_expansion_ratio.decompose().value)))
        return float(p[idx]) * u.bar

    def get_optimal_alt(self, opt_exit_pressure: Quantity):
        atmos = Atmosphere.from_pressure(opt_exit_pressure.to(u.Pa).value)
        return atmos.h[0] * u.m

    def get_injection_pressure(self, chamber_pressure: Quantity, mixture_ratio: Quantity):
        self._solve(chamber_pressure, mixture_ratio)
        gamma = float(np.array(self.solution.gamma)[0])
        cr = float(self.contraction_ratio.value)

        # Approximate injector total-pressure ratio from chamber Mach at finite contraction.
        mach_chamber = (2.0 / (gamma - 1.0) * (cr ** ((gamma - 1.0) / gamma) - 1.0)) ** 0.5
        p0_over_p = (1.0 + 0.5 * (gamma - 1.0) * mach_chamber**2) ** (gamma / (gamma - 1.0))
        return (chamber_pressure * p0_over_p).to(u.bar)

    def print_nominal_properties(self):
        print("-" * 40)
        print(" " * 3 + "#### NOMINAL ENGINE PROPERTIES ####")
        print("-" * 40)
        print("--- Input parameters ---")
        print(f"Total Massflow:         {self.nom_total_massflow:.2f}")
        print(f"O/F Ratio:              {self.nom_of_ratio:.2f}")
        print(f"Throat Area:            {self.throat_area:.2f}")
        print(f"Exit Area:              {self.exit_area:.2f}")
        print(f"Area Expansion Ratio:   {self.area_expansion_ratio:.2f}")
        print(f"C* Efficiency:          {self.c_star_efficiency:.2f}")
        print(f"CF Efficiency:          {self.cf_efficiency:.2f}")
        print(f"OX Massflow:            {self.nom_ox_massflow:.4f}")
        print(f"Fuel Massflow:          {self.nom_fuel_massflow:.4f}")

        chamber_pressure = self.get_chamber_pressure(self.nom_total_massflow, self.nom_of_ratio)
        sol = self._solve(chamber_pressure, self.nom_of_ratio)

        p = np.array(sol.P)
        idx_sl = int(np.argmin(np.abs(p - self.SL_pressure.to(u.bar).value)))

        isp_vac = float(np.nanmax(np.array(sol.Isp))) * u.s
        t_comb = float(np.array(sol.T)[0]) * u.K
        c_star = self.get_cstar(chamber_pressure, self.nom_of_ratio)
        opt_amb_pressure = self.get_optimal_ambient_pressure(chamber_pressure)
        if amb_working:
            opt_altitude = self.get_optimal_alt(opt_amb_pressure)

        cf = np.array(sol.coefficient_of_thrust)
        cstar = np.array(sol.c_star)

        opt_idx = int(np.argmin(np.abs(np.array(sol.ae_at) - self.area_expansion_ratio.value)))
        opt_isp = float(cf[opt_idx] * cstar[opt_idx] / 9.80665) * u.s * self.total_efficiency
        sl_isp = float(cf[idx_sl] * cstar[idx_sl] / 9.80665) * u.s * self.total_efficiency

        vac_cf = float(np.nanmax(cf)) * u.dimensionless_unscaled * self.cf_efficiency
        opt_cf = float(cf[opt_idx]) * u.dimensionless_unscaled * self.cf_efficiency
        sl_cf = float(cf[idx_sl]) * u.dimensionless_unscaled * self.cf_efficiency

        p_inj = self.get_injection_pressure(chamber_pressure, self.nom_of_ratio)

        print("--- Calculated parameters ---")
        print(f"Nominal Cham. Pressure: {chamber_pressure:.3f}")
        print(f"Injection Pressure:     {p_inj:.3f}")
        print(f"Optimal Exit Pressure:  {opt_amb_pressure:.3f}")
        if amb_working:
            print(f"Optimal Altitude:       {opt_altitude.to(u.km):.3f}")
        print(f"C*:                     {c_star:.2f}")
        print(f"Combustion Temp:        {t_comb:.2f}")
        print(f"Vac Isp:                {isp_vac:.3f}")
        print(f"Opt Isp:                {opt_isp:.3f}")
        print(f"SL Isp:                 {sl_isp:.3f}")
        print(f"Vac Cf:                 {vac_cf:.4f}")
        print(f"Opt Cf:                 {opt_cf:.4f}")
        print(f"SL Cf :                 {sl_cf:.4f}")
        print("-" * 40)

    def find_combustion_equilibrium(
        self,
        ox_manifold_pressure,
        fuel_manifold_pressure,
        threshold=1e-2,
        limit=100,
        verbose=False,
    ):
        ox_density = self.injector.get_ox_density(ox_manifold_pressure)
        fuel_density = self.injector.get_fuel_density(fuel_manifold_pressure)
        of_ratio = self.nom_of_ratio
        chamber_pressure = self.get_chamber_pressure(self.nom_total_massflow, self.nom_of_ratio)
        relaxation_factor = 0.3
        buffer_size = 5
        buffer = [chamber_pressure.to(u.bar).value] * buffer_size

        j = 0
        delta = 1
        if verbose:
            print("Starting Combustion Equilibrium Iteration...")

        while delta > threshold:
            if j > limit:
                raise ValueError(f"Iteration did not converge after {j} iterations")

            chamber_pressure = np.average(buffer, weights=[0.5, 0.25, 0.15, 0.03, 0.02]) * u.bar
            injection_pressure = self.get_injection_pressure(chamber_pressure, of_ratio)

            ox_pressure_drop = ox_manifold_pressure - injection_pressure
            fuel_pressure_drop = fuel_manifold_pressure - injection_pressure

            ox_massflow = self.injector.get_ox_massflow(ox_pressure_drop, ox_density)
            fuel_massflow = self.injector.get_fuel_massflow(fuel_pressure_drop, fuel_density)

            of_ratio = ox_massflow / fuel_massflow
            total_massflow = ox_massflow + fuel_massflow

            new_chamber_pressure = self.get_chamber_pressure(total_massflow, of_ratio)
            new_chamber_pressure = (
                chamber_pressure * (1 - relaxation_factor) + new_chamber_pressure * relaxation_factor
            )

            delta = abs(new_chamber_pressure.to(u.bar).value - chamber_pressure.to(u.bar).value)

            buffer.insert(0, new_chamber_pressure.to(u.bar).value)
            buffer.pop(-1)

            unique_values = len(set(round(x, 3) for x in buffer))
            if unique_values <= 2 and j > buffer_size:
                relaxation_factor = 0.1
            else:
                relaxation_factor = 0.3

            sol = self._solve(chamber_pressure, of_ratio)
            p = np.array(sol.P)
            cf = np.array(sol.coefficient_of_thrust)
            cstar = np.array(sol.c_star)
            idx_sl = int(np.argmin(np.abs(p - self.SL_pressure.to(u.bar).value)))
            sl_isp = float(cf[idx_sl] * cstar[idx_sl] / 9.80665) * u.s * self.total_efficiency
            sl_thrust = sl_isp * total_massflow
            if verbose:
                print(
                    f"{j}. Pc: {chamber_pressure:.3f}, o/f: {of_ratio:.3f}, "
                    f"mdot: {total_massflow:.3f}, sl_isp: {sl_isp:.3f}, "
                    f"thrust: {sl_thrust.value * 9.81 / 1000:.3f} kN, delta: {delta:.3f}"
                )

            j += 1

        chamber_pressure = new_chamber_pressure
        injection_pressure = self.get_injection_pressure(chamber_pressure, of_ratio)
        ox_pressure_drop = ox_manifold_pressure - injection_pressure
        fuel_pressure_drop = fuel_manifold_pressure - injection_pressure

        ox_massflow = self.injector.get_ox_massflow(ox_pressure_drop, ox_density)
        fuel_massflow = self.injector.get_fuel_massflow(fuel_pressure_drop, fuel_density)
        total_massflow = ox_massflow + fuel_massflow
        of_ratio = ox_massflow / fuel_massflow
        thrust = (sl_isp * total_massflow * 9.81).value * u.N

        self.of_ratio = of_ratio
        self.chamber_pressure = chamber_pressure
        self.thrust = thrust

        new_chamber_pressure = self.get_chamber_pressure(total_massflow, of_ratio)
        if abs(new_chamber_pressure.to(u.bar).value - chamber_pressure.to(u.bar).value) >= threshold:
            raise ValueError("Iteration converged but end result does not match threshold")

        if verbose:
            print("-" * 45)
            print(" " * 3 + "#### INJECTOR EQUILIBRIUM PROPERTIES ####")
            print("-" * 45)
            print(f"Converged after {j} iterations...")
            print(f"Ox Pressure Drop:   {ox_pressure_drop:.3f}")
            print(f"Fuel Pressure Drop: {fuel_pressure_drop:.3f}")
            print(f"Ox Massflow:        {ox_massflow:.5f}")
            print(f"Fuel Massflow:      {fuel_massflow:.5f}")
            print(f"OF:                 {of_ratio:.3f}")
            print(
                f"Ox dP Inj/Pc%:      {(ox_pressure_drop / chamber_pressure) * 100:.3f} %"
                "        [Should be > 20%]"
            )
            print(
                f"Fuel dP Inj/Pc%:    {(fuel_pressure_drop / chamber_pressure) * 100:.3f} %"
                "        [Should be > 20%]"
            )

        self.nom_ox_massflow = ox_massflow
        self.nom_fuel_massflow = fuel_massflow
