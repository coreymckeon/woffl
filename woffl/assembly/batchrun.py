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
from scipy import optimize as opt

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
                    "sonic_status": False,
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
    nozzles: list[str] | np.ndarray | pd.Series,
) -> list[bool]:
    """Batch Results Mask

    Create a mask of booleans for the batch results that can be passed into either a dataframe,
    numpy array or list as a filter. The initial filter is selecting the throat with the highest
    oil rate for each nozzle size. Any points where the oil rate is lower for a higher amount of
    water are then next removed. The filtered points can be passed to a function to calculate the gradient.

    Args:
        qoil_std (list): Oil Prod. Rate, BOPD
        qwat_tot (list): Total Water Rate, BWPD
        nozzles (list): List of the Nozzles, strings

    Returns:
        mask (list): True is a point to calc gradient, False means a point exists with better oil and less water
    """
    # convert all the lists into numpy arrays
    if isinstance(qoil_std, list):
        qoil_std = np.array(qoil_std)
    if isinstance(qwat_tot, list):
        qwat_tot = np.array(qwat_tot)
    if isinstance(nozzles, list):
        nozzles = np.array(nozzles)

    mask = np.zeros(len(qoil_std), dtype=bool)

    unique_nozzles = np.unique(nozzles)  # unique nozzles in the list
    for noz in unique_nozzles:
        idxs = np.where(nozzles == noz)[0]  # indicies where the nozzle is a specific nozzle
        max_idx = idxs[np.argmax(qoil_std[idxs])]
        mask[max_idx] = True

    # compare the points to themselves to look for where oil is higher for less water
    for idx_main in np.where(mask)[0]:
        for idx_sub in np.where(mask)[0]:
            if idx_main != idx_sub:
                if qoil_std[idx_main] < qoil_std[idx_sub] and qwat_tot[idx_main] > qwat_tot[idx_sub]:
                    mask[idx_main] = False
                    break

    return mask.tolist()


def exp_model(x, a, b, c):
    """Exponential Curve Fit

    Args:
        x (float): Input Value
        a (float): Asymptote of the Curve
        b (float): Constant
        c (float): Constant

    Returns
        y (float): Value at point x
    """
    return a - b * np.exp(-c * x)


def exp_deriv(x, b, c):
    """Derivative of Exponential Curve Fit

    Args:
        x (float): Input Value
        b (float): Constant
        c (float): Constant

    Returns
        s (float): Slope at point x
    """
    return c * b * np.exp(-c * x)


def rev_exp_deriv(s, b, c):
    """Derivative of Exponential Curve Fit, solve for x

    Args:
        s (float): Slope of the curve
        b (float): Constant
        c (float): Constant

    Returns
        x (float): Output x value
    """
    x = -1 / c * np.log(s / (c * b))
    x = max(x, 0)  # make sure s doesn't drop below zero
    return x


def batch_curve_fit(
    qoil_filt: list[float] | np.ndarray | pd.Series, qwat_filt: list[float] | np.ndarray | pd.Series
) -> tuple[float, float, float]:
    """Batch Curve Fit

    Curve fit the filtered datapoints from the Batch Results

    Args:
        qoil_filt (list): Filtered Oil Array, bopd
        qwat_filt (list): Filtered Water Array, bwpd

    Returns:
        coeff (float): a, b and c coefficients for curve fit
    """
    initial_guesses = [max(qoil_filt), max(qoil_filt), 0.001]
    coeff, _ = opt.curve_fit(exp_model, qwat_filt, qoil_filt, p0=initial_guesses)
    return coeff


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
    Add an additional argument that would be the coefficients for
    the curve fit of the upper portion of the data.

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


def gradient_back(oil_rate: np.ndarray, water_rate: np.ndarray) -> list:
    """Gradient Calculations Feed Backwards

    Completes a numerical gradient calculation between jet pumps. Adds a zero on
    the oil and water arrays that are passed to it. These ensures points exists for
    doing a reverse gradient calculation.

    Args:
        qoil_filt (list): Filtered Oil Prod. Rate, BOPD
        qwat_filt (list): Filtered Total Water Rate, BWPD

    Returns:
        gradient (list): Gradient of Oil-Rate / Water-Rate, bbl/bbl
    """
    oil_rate = np.append(0, oil_rate)  # add a zero to the start
    water_rate = np.append(0, water_rate)  # add a zero to the start

    if len(oil_rate) != len(water_rate):
        raise ValueError("Oil and Water Arrays Must be the same length")

    grad = []
    for i in range(len(oil_rate)):
        if i != 0:
            grad.append((oil_rate[i] - oil_rate[i - 1]) / (water_rate[i] - water_rate[i - 1]))
        else:
            pass
    return grad


def batch_fit_plot(
    qoil_filt: list[float] | np.ndarray | pd.Series,
    qwat_filt: list[float] | np.ndarray | pd.Series,
    coeff: tuple[float, float, float],
) -> None:
    """Batch Fit Plot

    Create a plot to view the analytical curve fit and derivative.
    Used for QC of Data and Information.


    Args:
        qoil_filt (list): Filtered Oil Prod. Rate, BOPD
        qwat_filt (list): Filtered Total Water Rate, BWPD
        coeff (tuple): Tuple of Curve Fit Coefficients
    """
    a, b, c = coeff  # parse out the coefficients for easier understanding

    fit_water = np.linspace(0, np.nanmax(qwat_filt), 1000)

    fit_oil = [exp_model(wat, a, b, c) for wat in fit_water]
    fit_grad = [exp_deriv(wat, b, c) for wat in fit_water]
    num_grad = gradient_back(qoil_filt, qwat_filt)  # type: ignore

    # Plotting the original data and the fitted curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    ax1.scatter(qwat_filt, qoil_filt, label="Original Data")
    ax1.plot(fit_water, fit_oil, color="red", linestyle="--", label="Exponential Fit")
    ax1.set_title("f(x) = a-b*exp(-c*x)")
    ax1.set_xlabel("Water Rate BWPD")
    ax1.set_ylabel("Oil Rate, BOPD")
    ax1.legend()

    ax2.plot(qwat_filt, num_grad, marker="o", linestyle="", label="Numerical Derivative")
    ax2.plot(fit_water, fit_grad, color="r", linestyle="--", label="Derivative Fit")
    ax2.set_title("df/dx = c*b*exp(-c*x)")
    ax2.set_xlabel("Water Rate BWPD")
    ax2.set_ylabel("Marginal Oil Water Ratio, BBL/BBL")
    ax2.legend()

    fig.suptitle(f"Model Coeff: a = {round(a, 1)}, b = {round(b, 1)}, c = {round(c, 5)}")

    plt.show()
