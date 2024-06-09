"""Batch Jet Pump Runs

Contains code that is used to run multiple pumps at once to understand the
current conditions. Code outputs a formatted list of python dictionaries that
can be converted to a Pandas Dataframe or equivalent for analysis.
"""

from dataclasses import dataclass
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import woffl.assembly.sysops as so
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Annulus, Pipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.resmix import ResMix


@dataclass(frozen=True)
class BatchResult:
    """Code is currently not used"""

    wellname: str
    nozzle: str
    throat: str
    choked: bool
    qoil_std: float
    fwat_bwpd: float
    mach_te: float
    total_water: float
    total_wc: float


class BatchPump:
    "Used for running different nozzle and throat combinations and selecting the correct one."

    def __init__(
        self,
        pwh: float,
        tsu: float,
        rho_pf: float,
        ppf_surf: float,
        wellbore: Pipe,
        wellprof: WellProfile,
        ipr_su: InFlow,
        prop_su: ResMix,
        wellname: str = "na",
    ) -> None:
        """Batch Pump Solver

        Used for iterating across a wide range of different pumps. An adjacent dataclass will
        be made for storing the results. Could add an optional wellname here if desired?

        Args:
            pwh (float): Pressure Wellhead, psig
            tsu (float): Temperature Suction, deg F
            rho_pf (float): Density of the power fluid, lbm/ft3
            ppf_surf (float): Pressure Power Fluid Surface, psig
            wellbore (Pipe): Pipe Class of the Wellbore
            wellprof (WellProfile): Well Profile Class
            ipr_su (InFlow): Inflow Performance Class
            prop_su (ResMix): Reservoir Mixture Conditions
        """
        self.pwh = pwh
        self.tsu = tsu
        self.rho_pf = rho_pf
        self.ppf_surf = ppf_surf
        self.wellbore = wellbore
        self.wellprof = wellprof
        self.ipr_su = ipr_su
        self.prop_su = prop_su
        self.wellname = wellname

    def update_pressure_wellhead(self, pwh: float) -> None:
        """Update the Wellhead Pressure

        Used to update the wellhead pressure instead of re-initializing everything.

        Args:
            pwh (float): Pressure Wellhead, psig
        """
        self.pwh = pwh

    def update_pressure_powerfluid(self, ppf_surf: float) -> None:
        """Update the Powerfluid Surface Pressure

        Used to update the powerfluid surface pressure instead of re-initializing everything.

        Args:
            ppf_surf (float): Pressure Power Fluid at Surface, psig
        """
        self.ppf_surf = ppf_surf

    def update_pressure_reservoir(self, pres: float) -> None:
        """Update the Reservoir Pressure in IPR

        Used to update reservoir pressure instead of re-initalizing everything.

        Args:
            pres (float): Pressure Reservoir, psig
        """
        self.ipr_su.pres = pres

    @staticmethod
    def jetpump_list(
        nozzles: list[str],
        throats: list[str],
        knz: float = 0.01,
        ken: float = 0.03,
        kth: float = 0.3,
        kdi: float = 0.4,
    ) -> list[JetPump]:
        """Create a list of Jet Pumps

        Automatically generate a list of different types of jet pumps. The list is fed into
        the batch run to understand performance. Could move this into batch run if desired.

        Args:
            nozzles (list): List of Nozzle Strings
            throats (list): List of Throat Ratios
            knz (float): Nozzle Friction Factor
            ken (float): Enterance Friction Factor
            kth (float): Throat Friction Factor
            kdi (float): Diffuser Friction Factor

        Returns:
            jp_list (list): List of JetPumps
        """
        jp_list = []
        for nozzle, throat in product(nozzles, throats):
            jp_list.append(JetPump(nozzle, throat, knz, ken, kth, kdi))
        return jp_list

    def batch_run(
        self,
        jetpumps: list[JetPump],
    ) -> list:
        """Batch Run of Jet Pumps

        Run through multiple different types of jet pumps. Results will be stored in
        a data class where the results can be graphed and selected for the optimal pump.

        Args:
            jetpump_list (list): List of JetPumps

        Returns:
            results (list): List of Dictionaries of Jet Pump Results
        """
        results = []
        for jetpump in jetpumps:
            try:
                psu_solv, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = so.jetpump_solver(
                    self.pwh,
                    self.tsu,
                    self.rho_pf,
                    self.ppf_surf,
                    jetpump,
                    self.wellbore,
                    self.wellprof,
                    self.ipr_su,
                    self.prop_su,
                )
                result = {
                    "wellname": self.wellname,
                    "res_pres": self.ipr_su.pres,
                    "pf_pres": self.ppf_surf,
                    "nozzle": jetpump.noz_no,
                    "throat": jetpump.rat_ar,
                    "psu_solv": psu_solv,
                    "sonic_status": sonic_status,
                    "qoil_std": qoil_std,
                    "fwat_bwpd": fwat_bwpd,
                    "qnz_bwpd": qnz_bwpd,
                    "mach_te": mach_te,
                    "total_water": fwat_bwpd + qnz_bwpd,
                    "total_wc": fwat_bwpd + qnz_bwpd / (fwat_bwpd + qnz_bwpd + qoil_std),
                    "error": "na",
                }
            except Exception as exc:
                result = {
                    "wellname": self.wellname,
                    "res_pres": self.ipr_su.pres,
                    "pf_pres": self.ppf_surf,
                    "nozzle": jetpump.noz_no,
                    "throat": jetpump.rat_ar,
                    "psu_solv": np.nan,
                    "sonic_status": np.nan,
                    "qoil_std": np.nan,
                    "fwat_bwpd": np.nan,
                    "qnz_bwpd": np.nan,
                    "mach_te": np.nan,
                    "total_water": np.nan,
                    "total_wc": np.nan,
                    "error": exc,
                }
            results.append(result)  # add some progress bar code here?
        return results

    def batch_debug(
        self,
        jetpumps: list[JetPump],
    ) -> list:
        """Batch Run of Jet Pumps

        Run through multiple different types of jet pumps. Results will be stored in
        a data class where the results can be graphed and selected for the optimal pump.
        Debugging method omits the try statement so places that the code fails can be
        seen and a method of fixing them can attempt to be put in place.

        Args:
            jetpump_list (list): List of JetPumps

        Returns:
            results (list): List of Dictionaries of Jet Pump Results
        """
        results = []
        for jetpump in jetpumps:
            print(f"Nozzle {jetpump.noz_no} and Throat: {jetpump.rat_ar} started")
            psu_solv, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = so.jetpump_solver(
                self.pwh,
                self.tsu,
                self.rho_pf,
                self.ppf_surf,
                jetpump,
                self.wellbore,
                self.wellprof,
                self.ipr_su,
                self.prop_su,
            )
            result = {
                "wellname": self.wellname,
                "res_pres": self.ipr_su.pres,
                "pf_pres": self.ppf_surf,
                "nozzle": jetpump.noz_no,
                "throat": jetpump.rat_ar,
                "psu_solv": psu_solv,
                "sonic_status": sonic_status,
                "qoil_std": qoil_std,
                "fwat_bwpd": fwat_bwpd,
                "qnz_bwpd": qnz_bwpd,
                "mach_te": mach_te,
                "total_water": fwat_bwpd + qnz_bwpd,
                "total_wc": fwat_bwpd + qnz_bwpd / (fwat_bwpd + qnz_bwpd + qoil_std),
            }
            results.append(result)  # add some progress bar code here?
        return results


# create a couple small functions that could be used across a pandas dataframe later
# need graphing, cleaning, and dropping variables, calculating gradients, finalized picking?
# could make these static methods inside the class if desired?


def batch_results_mask(
    qoil_std: list[float] | np.ndarray | pd.Series,
    qwat_tot: list[float] | np.ndarray | pd.Series,
) -> list[bool]:
    """Batch Results Mask

    Create a mask of booleans for the batch results that can be passed into either a dataframe,
    numpy array or list to filter out the unnecessary data. The unnecessary data are points where
    either a np.nan exists, or the oil rate is lower for a higher amount of water. The filtered
    points can then be passed to another function to calculate the gradient.

    Args:
        qoil_std (list): Oil Prod. Rate, BOPD
        qwat_tot (list): Total Water Rate, BWPD

    Returns:
        mask (list): True is a point to calc gradient, False means a point exists with better oil and less water
    """
    mask = []
    oil_copy = qoil_std.copy()
    wat_copy = qwat_tot.copy()

    for oil_main, wat_main in zip(qoil_std, qwat_tot):
        if oil_main == np.nan:
            status = False
        else:
            for oil_comp, wat_comp in zip(oil_copy, wat_copy):  # oil and water compare
                oil_diff = oil_comp - oil_main
                wat_diff = wat_comp - wat_main
                if (oil_diff > 0) and (wat_diff < 0):  # if another point has more oil for less water
                    status = False  # trash that point and exit the first loop
                    break
                else:
                    status = True  # has to maintain the True status for the entire loop through
        mask.append(status)
    return mask


def batch_results_plot(
    qoil_std: list[float] | np.ndarray | pd.Series,
    qwat_tot: list[float] | np.ndarray | pd.Series,
    nozzles: list[str] | np.ndarray | pd.Series,
    throats: list[str] | np.ndarray | pd.Series,
    wellname: str = "na",
    mask: list[bool] = [],
) -> None:
    """Batch Results Plot

    Create a plot to view the results from the batch run.

    Args:
        qoil_std (list): Oil Prod. Rate, BOPD
        qwat_tot (list): Total Water Rate, BWPD
        nozzles (list): Nozzle Numbers
        throats (list): Throat Ratios
        wellname (str): Wellname String
        mask (list): List of Booleans used for filtering data
    """
    jp_names = [noz + thr for noz, thr in zip(nozzles, throats)]  # create a list of all the jetpump names
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if not mask:  # if the mask is an empty list
        mask = [False for noz in nozzles]  # create a list of falses

    for oil, water, jp, status in zip(qoil_std, qwat_tot, jp_names, mask):
        if oil == np.nan:
            pass
        else:
            if status:  # booleaan true of false
                ax.plot(water, oil, marker="o", linestyle="", color="r")  # one of the main point
            else:
                ax.plot(water, oil, marker="o", linestyle="", color="b")  # a one optimized point

            ax.annotate(jp, xy=(water, oil), xycoords="data", xytext=(1.5, 1.5), textcoords="offset points")

    ax.set_xlabel("Total Water Rate, BWPD")
    ax.set_ylabel("Produced Oil Rate, BOPD")

    if wellname == "na":
        ax.title.set_text("Jet Pump Performance")
    else:
        ax.title.set_text(f"{wellname} Jet Pump Performance")

    plt.show()
