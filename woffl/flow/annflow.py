import math

import numpy as np

import woffl.flow.singlephase as sp
from woffl.geometry.pipe import Annulus
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.formwat import FormWater


class AnnFlow:
    def __init__(
        self,
        ann_rate: float,
        surf_press: float,
        surf_temp: float,
        prop_an: FormWater,
        ann_dim: Annulus,
        wellprof: WellProfile,
    ) -> None:
        """Annular Flow

        Calculations that constrain the downward flow through the annulus.

        Args:
            ann_rate (float): Water Lift Rate, BWPD
            surf_press (float): Water Pressure at Wellhead, PSIG
            surf_temp (float): Water Temperature at Wellhead, deg F
            fluid (FormWater): Fluid to pull properties from
            ann_dim (Annulus): Annular dimensions that is flowed inside
            wellprofile (WellProfile): survey dimensions and location of jet pump

        Returns:
            Self
        """
        self.ann_rate = ann_rate
        self.surf_press = surf_press
        self.surf_temp = surf_temp
        self.prop = prop_an
        self.ann_dim = ann_dim
        self.wellprof = wellprof

        def __repr__(self):
            return f"{self.ann_rate} BWPD flowing inside a {self.ann_dim.out_pipe.inn_dia} annulus"
