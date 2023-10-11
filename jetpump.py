import math

import numpy as np


class JetPump:
    # lookup table to find the nozzle diameter in inches
    # Kermit Brown Vol 2b Table 6.1
    kermit_dict = {
        "1": 0.06869,
        "2": 0.07680,
        "3": 0.08587,
        "4": 0.09600,
        "5": 0.10733,
        "6": 0.12000,
        "7": 0.13416,
        "8": 0.15000,
        "9": 0.16771,
        "10": 0.18750,
        "11": 0.20963,
        "12": 0.23438,
        "13": 0.26218,  # kermit brown pump
        "14": 0.29297,
        "15": 0.32755,
        "16": 0.36621,
        "17": 0.40944,
        "18": 0.45776,
        "19": 0.51180,
        "20": 0.57220,
    }

    # national jet pump sizes
    # from Champion X Benji
    nozz_dict = {
        "1": 0.0553,
        "2": 0.0628,
        "3": 0.0705,
        "4": 0.0798,
        "5": 0.0903,  # matches 1 throat
        "6": 0.1016,  # matches 2 throat
        "7": 0.1145,  # doesn't match the 3 throat
        "8": 0.1291,  # matches 4 throat
        "9": 0.1458,  # matches 5 throat
        "10": 0.1643,
        "11": 0.1858,
        "12": 0.2099,
        "13": 0.237,
        "14": 0.2675,
        "15": 0.3017,
        "16": 0.3404,
        "17": 0.3841,
        "18": 0.4335,
        "19": 0.4981,
        "20": 0.5519,
    }

    # national jet pump throat sizes
    # from Champion X Benji
    thrt_dict = {
        "1": 0.0903,
        "2": 0.1016,
        "3": 0.1151,
        "4": 0.1291,
        "5": 0.1458,
        "6": 0.1643,
        "7": 0.1858,
        "8": 0.2099,
        "9": 0.237,
        "10": 0.2675,
        "11": 0.3017,
        "12": 0.3404,
        "13": 0.3841,
        "14": 0.4335,
        "15": 0.4981,
        "16": 0.5519,
        "17": 0.6228,
        "18": 0.7027,
        "19": 0.7929,
        "20": 0.8947,
    }

    # lookup table to define the area ratio
    area_dict = {
        "A": 0.410,
        "B": 0.328,
        "C": 0.262,
        "D": 0.210,
        "E": 0.168,
    }

    def __init__(self, nozzle_no: str, area_ratio: str) -> None:
        """Jet Pump Initialization

        Create a Jet Pump object.

        Args:
            nozzle_no (string): Represents the Nozzle Size
            area_ratio (string): Represents the Area Ratio

        Returns:
            None
        """
        # nozzle number is a string
        # area ratio is a string
        self.noz_no = str(nozzle_no)  # nozzle number, make sure it is a string
        self.rat_ar = area_ratio  # area ratio

        # define friction factors
        # these values won't be changed in functional calls
        self.knoz = 0.1  # nozzle friction factor
        self.kthr = 0.3  # throat / diffuser friction factor

        # lookup the nozzle diameter
        # do error handling if the nozzle is not in the dictionary
        try:
            self.dnoz = JetPump.nozz_dict[self.noz_no]  # inches
        except KeyError:
            print(f"Nozzle Size {self.noz_no} not recognized")

        # calculate area of the nozzle
        anoz = 3.1415 * self.dnoz**2 / 4  # in^2
        self.anoz = round(anoz, 5)  # in^2

        # calculate throat area from area ratio / nozzle area
        # do error handling if the area ratio is not in the dictionary
        try:
            athr = self.anoz / JetPump.area_dict[self.rat_ar]  # in^2
            self.athr = round(athr, 5)  # in^2
        except KeyError:
            print(f"Area Ratio {self.rat_ar} not recognized")

        # calculate throat diameter
        dthr = math.sqrt(4 * self.athr / 3.1415)  # inches
        self.dthr = round(dthr, 5)  # inches

    def __str__(self):
        return f"{self.noz_no+self.rat_ar} Jet Pump, Nozzle: {self.dnoz} inches, Throat: {self.dthr} inches"

    def p_jt(self, q_pf, p_pf, d_pf):
        """Jet Pressure
        Input:  q_pf - power fluid flowrate, ft^3/s
                p_pf - power fluid pressure, psig (At pump conditions)
                d_pf - power fluid density,  lbm/ft^3
        Output: p_jt - jet pressure, psia (Throat Enterance)
        """

        self.q_pf = q_pf
        self.p_pf = p_pf
        self.d_pf = d_pf

        # put the area of the nozzle and area of the throat in
        # the dunder init function...self.anoz
        anoz = 3.1415 * self.dnoz**2 / (4 * 144)  # ft^2, area of nozzle
        vnoz = q_pf / anoz  # ft/s, velocity of nozzle

        # nozzle kinetic energy in bernoulli equation
        # convert from lbm*ft/s^2 to lbf/in^2
        # psi, kinetic energy of nozzle
        ke_noz = (1 / 2) * d_pf * vnoz**2 / (144 * 32.174)
        ke_noz = round(ke_noz, 2)

        # equation 1 cunningham, bernoulli application
        p_jt = p_pf - ke_noz * (1 + self.knoz)  # psi
        p_jt = round(p_jt, 2)  # psi
        print(f"Nozzle Vel: {round(vnoz,2)} ft/s, Nozzle KE: {ke_noz} psi, Jet Pressure: {p_jt} psi")

        return p_jt
