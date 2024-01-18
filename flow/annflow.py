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

    @staticmethod
    def bpd_to_ft3s(q_bpd: float) -> float:
        """Convert liquid BPD to ft3/s

        Args:
            q_bpd (float): Volumetric Flow, BPD

        Returns:
            q_ft3s (float): Volumetric Flow, ft3/s
        """
        # 42 gal/bbl, 7.48052 gal/ft3, 24 hr/day, 60min/hour, 60sec/min
        q_ft3s = q_bpd * 42 / (24 * 60 * 60 * 7.48052)
        q_ft3s = round(q_ft3s, 3)
        return q_ft3s

    @staticmethod
    def static_press(rho: float, height: float) -> float:
        """Static Fluid Column Pressure

        Args:
            rho (float): Fluid Density, lbm/ft3
            height (float): Fluid Height, feet

        returns:
            prs_st (float): Static Pressure, psi
        """
        prs_st = rho * height / 144  # psi, gravity cancels each other out with US Units
        return round(prs_st, 2)

    @staticmethod
    def velocity(q_ft3s: float, area: float) -> float:
        """Flow Velocity

        Args:
            q_ft3s (float): Volumetric Flow, ft3/s
            area (float): Pipe Cross Sectional Area, ft2

        Returns:
            vel (float): Flow Velocity, ft/s
        """
        vel = q_ft3s / area
        return round(vel, 3)

    @staticmethod
    def reynolds(rho: float, vel: float, dhyd: float, visc: float) -> float:
        """Reynolds Number

        Args:
            rho (float): Fluid density, lbm/ft3
            vel (float): Flow velocity, ft/s
            dhyd (float): Hydraulic Diameter, inches
            visc (float): Dynamic Viscosity, cP

        Returns:
            reynolds (float): Reynolds number, unitless
        """
        # convert inch to feet
        dhyd = dhyd / 12  # feet

        # convert cp to lbm/(ft*s)
        visc = visc / 1488.2  # lbm/(ft*s)

        reynolds = rho * vel * dhyd / visc
        return round(reynolds, 2)

    @staticmethod
    def ffactor(reynolds: float, rel_ruff: float) -> float:
        """Darcy Friction Factor of Piping

        Args:
            reynolds (float): Reynolds Number, Re unitless
            rel_ruff (float): Relative Roughness, e/D unitless

        Returns:
            ff (float): Darcy Friction Factor of Piping, unitless

        References:
            - Cranes Technical Paper No. 410 Equation 1-21
        """
        # laminar / transistional flow
        if reynolds < 4000:
            ff = 64 / reynolds

        else:
            # serghide equation Cranes TP410 Equation 1-21
            # log base 10 is correct, not natural log
            a = -2 * math.log10((rel_ruff / 3.7) + (12 / reynolds))
            b = -2 * math.log10((rel_ruff / 3.7) + (2.51 * a / reynolds))
            c = -2 * math.log10((rel_ruff / 3.7) + (2.51 * b / reynolds))
            ff = (a - (b - a) ** 2 / (c - 2 * b + a)) ** -2

        return round(ff, 4)

    @staticmethod
    def dynamic_press(ff: float, length: float, rho: float, vel: float, dhyd: float) -> float:
        """Dynamic Pressure Loss

        Calculate the frictional pressure loss in a piping system.
        Commonly referred to as dynamic pressure

        Args:
            ff (float): darcy friction factor of the pipe, unitless
            length (float): length of piping, feet
            rho (float): fluid density, lbm/ft3
            vel (float): fluid velocity, ft/s
            dhyd (float): hydraulic diameter, inches

        Returns:
            dp_fric (float): pressure loss from friction, psi
        """
        g = 32.174  # 1lbf equals 32.174 lbm*ft/s2

        # convert inch to feet
        dhyd = dhyd / 12  # feet

        dp_fric = ff * rho * vel**2 * length / (2 * dhyd * g)  # lbf/ft2

        # convert lbf/ft2 to lbf/in2
        dp_fric = dp_fric / 144  # lbf/in2
        return round(dp_fric, 2)
