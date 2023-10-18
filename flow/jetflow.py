import math

import numpy as np
from inflow import InFlow

from hilcorpak.jetpump import JetPump
from hilcorpak.pvt.blackoil import BlackOil
from hilcorpak.pvt.formgas import FormGas
from hilcorpak.pvt.formwat import FormWater
from hilcorpak.pvt.resmix import ResMix

"""This file is nothing more than a collection of Kaelin trying to make
some jet pump theory work in a practical method. At this point I don't
really like Bobs write up on pto vs pte and I will attempt my own method.
I'm going to just make static functions in an attempt to make this easier
"""

"""
Needs the following:
IPR, Reservoir Fluid with WC / GOR, Jet Pump with dimensions of nozzle/throat.
"""

# def throat_entry():
