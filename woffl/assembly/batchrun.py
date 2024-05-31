"""Batch Jet Pump Runs

Contains code that is used to run multiple pumps at once to understand the
current conditions. Currently Scott's code is here with a simple nested for
loop. Will eventually update this to more of a class style system. Runs the
analysis and sends the results to a .csv file.
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
                    "total water": fwat_bwpd + qnz_bwpd,
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
                    "total water": np.nan,
                    "total_wc": np.nan,
                    "error": exc,
                }
            results.append(result)
        return results


# create a couple small functions that could be used across a pandas dataframe later
# need graphing, cleaning, and dropping variables, calculating gradients, finalized picking?


results = []
dfjet = pd.DataFrame(results)
dfjet = dfjet.sort_values(by="psu_solv", ascending=True)
# df_sorted.to_csv("modelrun_output B-35.csv")
print(dfjet)

plt.plot(dfjet["total_water"], dfjet["qoil_std"], marker="o", linestyle="")
plt.xlabel("Total Water, BWPD")
plt.ylabel("Oil Rate, BOPD")
plt.show()
