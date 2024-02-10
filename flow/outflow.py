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
