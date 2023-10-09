import math

import numpy as np
from geometry.pipe import Annulus
from geometry.wellprofile import WellProfile
from pvt.formwat import FormWater

# this and tubing flow could be a singular class most likely
# would require you to have a fluid be a ResMix
# define whether the flow was in the tubing or it was in the annulus
# but I also think that it could get messy with the single phase vs three phase calls?
# can I add an addition clause to ResMix that if you add water it recalcs and updates wc?


class AnnFlow:
    def __init__(
        self,
        ann_rate: float,
        surf_press: float,
        surf_temp: float,
        fluid: FormWater,
        ann_dim: Annulus,
        wellprofile: WellProfile,
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
        self.fluid = fluid
        self.ann_dim = ann_dim
        self.wellprofile = wellprofile

        def __repr__(self):
            return f"{self.ann_rate} BWPD flowing inside a {self.ann_dim.out_pipe.inn_dia} annulus"
