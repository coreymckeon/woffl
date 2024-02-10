"""Out Flow for an Oil Well

Performs the calculations that occur inside a vertical, inclined or horizontal wellbore.
Predominate equations are two phase flow correlations. The most common being Beggs and Brill
"""

import math

import numpy as np

from flow import jetflow as jf
from flow import jetplot as jplt
from flow import singlephase as sp

# from flow.inflow import InFlow
# from geometry.jetpump import JetPump
from geometry.pipe import Annulus, Pipe

# from pvt.blackoil import BlackOil
# from pvt.formgas import FormGas
# from pvt.formwat import FormWater
from pvt.resmix import ResMix
