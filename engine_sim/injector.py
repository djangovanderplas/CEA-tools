import numpy as np
import astropy.units as u
from astropy.units import Quantity
from CoolProp.CoolProp import PropsSI


class Injector:
    """Injector model used for pressure-drop-driven mass flow estimates."""

    def __init__(self, ox_area, fuel_area, ox_cd, fuel_cd, ox_temp, fuel_temp):
        self.ox_area = ox_area
        self.fuel_area = fuel_area
        self.ox_cd = ox_cd
        self.fuel_cd = fuel_cd
        self.ox_temp = ox_temp
        self.fuel_temp = fuel_temp

    @staticmethod
    def calc_annular_orifice_area(d_inner: Quantity, d_outer: Quantity) -> Quantity:
        return np.pi * (d_outer**2 - d_inner**2) / 4

    @staticmethod
    def calc_circular_orifice_area(d: Quantity, n_holes: int = 1) -> Quantity:
        return np.pi * (d / 2) ** 2 * n_holes

    def get_ox_density(self, pressure: Quantity, ox_name: str = "Oxygen") -> Quantity:
        return (
            PropsSI("D", "P", pressure.to(u.Pa).value, "T", self.ox_temp.to(u.K).value, ox_name)
            * u.kg
            / (u.m**3)
        )

    def get_fuel_density(
        self,
        pressure: Quantity,
        fuel_name: str = "HEOS::Ethanol[0.8]&H2O[0.2]",
    ) -> Quantity:
        return (
            PropsSI("D", "P", pressure.to(u.Pa).value, "T", self.fuel_temp.to(u.K).value, fuel_name)
            * u.kg
            / (u.m**3)
        )

    def get_ox_massflow(self, pressure_drop: Quantity, ox_density: Quantity) -> Quantity:
        return (self.ox_cd * self.ox_area * np.sqrt(2 * ox_density * pressure_drop)).decompose()

    def get_fuel_massflow(self, pressure_drop: Quantity, fuel_density: Quantity) -> Quantity:
        return (self.fuel_cd * self.fuel_area * np.sqrt(2 * fuel_density * pressure_drop)).decompose()
