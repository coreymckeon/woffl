"""Out Flow for an Oil Well

Performs the calculations that occur inside a vertical, inclined or horizontal wellbore.
Predominate equations are two phase flow correlations. The most common being Beggs and Brill
"""

import math

import numpy as np

from flow import singlephase as sp
from flow import twophase as tp
from geometry.pipe import Annulus, Pipe
from geometry.wellprofile import WellProfile
from pvt.resmix import ResMix


def homo_diff_press(
    pin: float, tin: float, inn_dia: float, abs_ruff: float, length: float, height: float, qoil_std: float, prop: ResMix
) -> float:
    """Homogenous Differential Pressure

    Calculate Homogenous Differential Pressure Across Wellbore / Pipe.
    Note, could have positive / negative length and height, depending on which node you
    are starting with. EG, starting at bottom and going up or starting at top and going down.

    Args:
        pin (float): Inlet Pressure, psig
        tin (float): Inlet Temperature, deg F
        inn_dia (float): Inner Diameter of Pipe, inches
        abs_ruff (float): Absolute Roughness of Pipe, inches
        length (float): Length of Pipe, feet, Positive is with flow direction
        height (float): Height Difference of Pipe, feet, Positive is upwards
        qoil_std (float): Oil Rate, STBOPD
        prop (ResMix): Properties of Fluid Mixture in Wellbore, ResMix

    Returns:
        dp (float): Differential Pressure, psid
        nslh (float): No Slip Liquid Holdup, unitless
    """
    prop = prop.condition(pin, tin)
    qtot = sum(prop.insitu_volm_flow(qoil_std))
    rho_mix = prop.rho_mix()
    visc_mix = prop.visc_mix()

    area = sp.cross_area(inn_dia)
    vmix = sp.velocity(qtot, area)
    rey = sp.reynolds(rho_mix, vmix, inn_dia, visc_mix)
    rel_ruff = sp.relative_roughness(inn_dia, abs_ruff)
    ff = sp.ffactor_darcy(rey, rel_ruff)

    dp_fric = sp.diff_press_friction(ff, rho_mix, vmix, inn_dia, length)
    dp_stat = sp.diff_press_static(rho_mix, height)
    dp = dp_fric + dp_stat
    return dp


# how should I definre where the jetpump is exactly?
# you don't need to look past the location of the jump pump discharge
class OutFlow:
    def __init__(
        self, oil_rate: float, surf_press: float, surf_temp: float, prop_wb: ResMix, tubing: Pipe, wellprof: WellProfile
    ) -> None:
        """Out Flow from JetPump to Surface

        Calculations for the upward flow through the tubing.

        Args:
            oil_rate (float): Oil Rate, BOPD
            surf_press (float): Production Pressure at Wellhead, PSIG
            surf_temp (float): Production Temperature at Wellhead, deg F
            prop_wb (FormWater): Wellbore fluid to pull properties from
            tubing (Pipe): Tubing dimensions that the flow is inside
            wellprofile (WellProfile): survey dimensions and location of jet pump

        Returns:
            Self
        """
        self.oil_rate = oil_rate
        self.surf_press = surf_press
        self.surf_temp = surf_temp
        self.prop = prop_wb
        self.tubing = tubing
        self.wellprof = wellprof

    def __repr__(self):
        return f"{self.oil_rate} BOPD flowing inside a {self.tubing.inn_dia} inch pipe"


# I even really need a class here? Should I just make a segment and then piece those segments
# together?

# start with filtered wellbore and homogenous flow parameters
# the following function takes a pre-broken wellbore segment and calcs out the
