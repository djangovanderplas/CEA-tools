import astropy.units as u

from engine_sim.engine_analysis import EngineAnalysis
from engine_sim.injector import Injector


injector = Injector(
    ox_area=Injector.calc_circular_orifice_area(1.5 * u.mm, 68),
    fuel_area=Injector.calc_annular_orifice_area(30.45 * u.mm, 31.68 * u.mm),
    ox_cd=0.6 * u.dimensionless_unscaled,
    fuel_cd=1 * u.dimensionless_unscaled,
    ox_temp=90 * u.K,
    fuel_temp=305 * u.K,
)

engine = EngineAnalysis(
    injector=injector,
    throat_diameter=50.83 * u.mm,
    exit_diameter=153.7 * u.mm,
    c_star_efficiency=0.88,
    cf_efficiency=1.03,
    nom_ox_massflow=2.246 * u.kg / u.s,
    nom_fuel_massflow=1.604 * u.kg / u.s,
    contraction_ratio=6.86 * u.dimensionless_unscaled,
)

engine.find_combustion_equilibrium(
    ox_manifold_pressure=36 * u.bar,
    fuel_manifold_pressure=37 * u.bar,
    verbose=True,
)
engine.print_nominal_properties()
