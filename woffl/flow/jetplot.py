"""Jet Plotting and Storage

A series of functions that are predominately built for variable storage and plotting.
Classes that store values and results are referred to as Books. A single book class is
built that stores calculations for the throat entry book and a diffuser. The books possess
the following.

Returns:
        prs_ray (np array): Pressure Array, psig
        vel_ray (np array): Velocity Array, ft/s
        rho_ray (np array): Density Array, lbm/ft3
        snd_ray (np array): Speed of Sound Array, ft/s
        mach_ray (np array): Mach Number, unitless
        kde_ray (np array): Kinetic Differential Energy, ft2/s2
        ede_ray (np array): Expansion Differential Energy, ft2/s2
        tde_ray (np array): Total Differntial Energy, ft2/s2
        grad_ray (np array): Gradient of tde/dp Array, ft2/(s2*psig)
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes

from woffl.flow import jetflow as jf  # legacy
from woffl.flow import singlephase as sp
from woffl.flow.inflow import InFlow
from woffl.pvt.resmix import ResMix


class JetBook:
    def __init__(self, prs: float, vel: float, rho: float, snd: float, kde: float):
        """Book for storing Jet Pump Enterance / Diffuser Calculations

        Create a book for storing results values from the throat enterance and diffuser.
        Can be used later for graphing and other analysis.

        Args:
            prs (float): Pressure, psig
            vel (float): Velocity, ft/s
            rho (float): Density, lbm/ft3
            snd (float): Speed of Sound, ft/s
            kde (float): Kinetic Differential Energy, ft2/s2
        """
        self.prs_ray = np.array([prs])
        self.vel_ray = np.array([vel])
        self.rho_ray = np.array([rho])
        self.snd_ray = np.array([snd])

        self.kde_ray = np.array([kde])

        ede = 0
        tde = kde + ede

        self.ede_ray = np.array([ede])  # expansion energy array
        self.tde_ray = np.array([tde])  # total differential energy
        self.mach_ray = np.array([vel / snd])  # mach number
        self.grad_ray = np.array([np.nan])  # gradient of tde vs prs

    # https://docs.python.org/3/library/string.html#formatspec
    def __repr__(self):
        """Creates a fancy table to see some of the stored data"""
        sformat = "{:>8} | {:>8} | {:>7} | {:>6} | {:>6} \n"  # string format
        nformat = "{:>8.1f} | {:>8.1f} | {:>7.1f} | {:>6.1f} | {:>6.2f} \n"  # number format
        spc = 48 * "-" + "\n"  # spacing
        pout = sformat.format("pressure", "velocity", "density", "sound", "mach")
        pout = pout + sformat.format("psig", "ft/s", "lbm/ft3", "ft/s", "no_u") + spc
        for prs, vel, rho, snd, mach in zip(self.prs_ray, self.vel_ray, self.rho_ray, self.snd_ray, self.mach_ray):
            pout += nformat.format(prs, vel, rho, snd, mach)
        return pout

    def append(self, prs: float, vel: float, rho: float, snd: float, kde: float):
        """Append Values onto the Throat Entry Book

        Appended values are added onto the existing throat entry arrays.

        Args:
            prs (float): Pressure, psig
            vel (float): Velocity, ft/s
            rho (float): Density, lbm/ft3
            snd (float): Speed of Sound, ft/s
            kde (float): Kinetic Differential Energy, ft2/s2
        """
        self.prs_ray = np.append(self.prs_ray, prs)
        self.vel_ray = np.append(self.vel_ray, vel)
        self.rho_ray = np.append(self.rho_ray, rho)
        self.snd_ray = np.append(self.snd_ray, snd)
        self.kde_ray = np.append(self.kde_ray, kde)

        ede = self.ede_ray[-1] + jf.incremental_ee(self.prs_ray[-2:], self.rho_ray[-2:])
        tde = kde + ede

        self.ede_ray = np.append(self.ede_ray, ede)
        self.tde_ray = np.append(self.tde_ray, tde)
        self.mach_ray = np.append(self.mach_ray, vel / snd)  # mach number

        grad = (self.tde_ray[-2] - self.tde_ray[-1]) / (self.prs_ray[-2] - self.prs_ray[-1])
        self.grad_ray = np.append(self.grad_ray, grad)  # gradient of tde vs prs

    def plot_te(self) -> None:
        """Throat Entry Plots

        Create a series of graphs to use for visualization of the results
        in the book from the throat entry area.
        """
        self._throat_entry_graphs(
            self.prs_ray,
            self.vel_ray,
            self.rho_ray,
            self.snd_ray,
            self.kde_ray,
            self.ede_ray,
            self.tde_ray,
            self.grad_ray,
        )
        return None

    def plot_di(self) -> None:
        """Diffuser Plots

        Create a series of graphs to use for visualization of the results in
        the book from the diffuser area.
        """
        self._diffuser_graphs(
            self.prs_ray,
            self.vel_ray,
            self.rho_ray,
            self.snd_ray,
            self.kde_ray,
            self.ede_ray,
            self.tde_ray,
        )
        return None

    def dete_zero(self) -> tuple[float, float, float, float]:
        """Throat Entry Parameters at Zero Total Differential Energy

        Args:
            None

        Return:
            pte (float): Throat Entry Pressure, psig
            vte (float): Throat Entry Velocity, ft/s
            rho_te (float): Throat Entry Density, lbm/ft3
            mach_te (float): Mach Throat Entry, unitless
        """
        return self._dete_zero(self.prs_ray, self.vel_ray, self.rho_ray, self.tde_ray, self.mach_ray)

    def dedi_zero(self) -> float:
        """Diffuser Discharge Pressure at Zero Total Differential Energy

        Args:
            None

        Return:
            pdi (float): Diffuser Discharge Pressure, psig
        """
        return np.interp(0, self.tde_ray, self.prs_ray)  # type: ignore

    @staticmethod
    def _dete_zero(
        prs_ray: np.ndarray,
        vel_ray: np.ndarray,
        rho_ray: np.ndarray,
        tde_ray: np.ndarray,
        mach_ray: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Throat Entry Parameters at Zero Total Differential Energy

        Calculate the throat entry pressure, density, and velocity where dEte is zero

        Args:
            prs_ray (np array): Pressure Array, psig
            vel_ray (np array): Velocity Array, ft/s
            rho_ray (np array): Density Array, lbm/ft3
            tde_ray (np array): Total Differential Energy, ft2/s2

        Return:
            pte (float): Throat Entry Pressure, psig
            vte (float): Throat Entry Velocity, ft/s
            rho_te (float): Throat Entry Density, lbm/ft3
            mach_te (float): Mach Throat Entry, unitless
        """
        dtdp = np.gradient(tde_ray, prs_ray)  # uses central limit thm, so same size
        mask = dtdp >= 0  # only points where slope is greater than or equal to zero

        tde_flipped = np.flip(tde_ray[mask])

        pte = np.interp(0, tde_flipped, np.flip(prs_ray[mask]))
        vte = np.interp(0, tde_flipped, np.flip(vel_ray[mask]))
        rho_te = np.interp(0, tde_flipped, np.flip(rho_ray[mask]))
        mach_te = np.interp(0, tde_flipped, np.flip(mach_ray[mask]))

        return pte, vte, rho_te, mach_te  # type: ignore

    @staticmethod
    def _throat_entry_graphs(
        prs_ray: np.ndarray,
        vel_ray: np.ndarray,
        rho_ray: np.ndarray,
        snd_ray: np.ndarray,
        kde_ray: np.ndarray,
        ede_ray: np.ndarray,
        tde_ray: np.ndarray,
        grad_ray: np.ndarray,
    ) -> None:

        mach_ray = vel_ray / snd_ray
        # grad_ray = np.gradient(tde_ray, prs_ray)
        psu = prs_ray[0]
        pmo = float(np.interp(1, mach_ray, prs_ray))  # interpolate for pressure at mach 1, pmo
        pgo = float(np.interp(0, np.flip(grad_ray), np.flip(prs_ray)))  # find point where gradient is zero
        pte, vte, rho_te, mach_te = JetBook._dete_zero(prs_ray, vel_ray, rho_ray, tde_ray, mach_ray)

        fig, axs = plt.subplots(4, sharex=True)
        plt.rcParams["mathtext.default"] = "regular"
        fig.suptitle(f"Suction at {round(psu,0)} psi, Mach 1 at {round(pmo,0)} psi")

        axs[0].scatter(prs_ray, 1 / rho_ray)
        axs[0].set_ylabel("Specific Volume, ft3/lbm")

        axs[1].scatter(prs_ray, vel_ray, label="Mixture Velocity")
        axs[1].scatter(prs_ray, snd_ray, label="Speed of Sound")
        axs[1].set_ylabel("Velocity, ft/s")
        axs[1].legend()

        axs[2].scatter(prs_ray, ede_ray, label="Expansion")
        axs[2].scatter(prs_ray, kde_ray, label="Kinetic")
        axs[2].axhline(y=0, color="black", linestyle="--", linewidth=1)
        axs[2].set_ylabel("Energy, ft2/s2")
        axs[2].legend()

        ycoord = (max(tde_ray) + min(tde_ray)) / 2
        axs[3].scatter(prs_ray, tde_ray)
        axs[3].axhline(y=0, color="black", linestyle="--", linewidth=1)
        axs[3].axvline(x=pmo, color="black", linestyle="--", linewidth=1)
        axs[3].annotate(text="Mach 1", xy=(pmo, ycoord), rotation=90)
        axs[3].axvline(x=pte, color="black", linestyle="--", linewidth=1)
        axs[3].annotate(text="TEE 0", xy=(pte, ycoord), rotation=90)
        axs[3].axvline(x=pgo, color="black", linestyle="--", linewidth=1)
        axs[3].annotate(text="Grad 0", xy=(pgo, 2 * ycoord / 8), rotation=90)
        axs[3].set_ylabel("$dE_{te}$, ft2/s2")
        axs[3].set_xlabel("Throat Entry Pressure, psig")

        plt.show()
        return None

    @staticmethod
    def _diffuser_graphs(
        prs_ray: np.ndarray,
        vel_ray: np.ndarray,
        rho_ray: np.ndarray,
        snd_ray: np.ndarray,
        kde_ray: np.ndarray,
        ede_ray: np.ndarray,
        tde_ray: np.ndarray,
    ) -> None:

        ptm = prs_ray[0]

        fig, axs = plt.subplots(4, sharex=True)
        plt.rcParams["mathtext.default"] = "regular"

        axs[0].scatter(prs_ray, 1 / rho_ray)
        axs[0].set_ylabel("Specific Volume, ft3/lbm")

        axs[1].scatter(prs_ray, vel_ray, label="Diffuser Outlet")
        axs[1].scatter(prs_ray, snd_ray, label="Speed of Sound")
        # axs[1].scatter(ptm, vtm, label="Diffuser Inlet")
        axs[1].set_ylabel("Velocity, ft/s")
        axs[1].legend()

        axs[2].scatter(prs_ray, ede_ray, label="Expansion")
        axs[2].scatter(prs_ray, kde_ray, label="Kinetic")
        axs[2].axhline(y=0, color="black", linestyle="--", linewidth=1)
        axs[2].set_ylabel("Energy, ft2/s2")
        axs[2].legend()

        axs[3].scatter(prs_ray, tde_ray)
        axs[3].axhline(y=0, color="black", linestyle="--", linewidth=1)
        axs[3].set_ylabel("$dE_{di}$, ft2/s2")
        axs[3].set_xlabel("Diffuser Outlet Pressure, psig")

        if max(tde_ray) >= 0 and min(tde_ray) <= 0:  # make sure a solution exists
            pdi = np.interp(0, tde_ray, prs_ray)
            vdi = np.interp(pdi, prs_ray, vel_ray)
            ycoord = (min(vel_ray) + max(snd_ray)) / 2
            axs[1].axvline(x=pdi, color="black", linestyle="--", linewidth=1)
            axs[1].annotate(text=f"{round(vdi, 1)} ft/s", xy=(pdi, ycoord), rotation=90)  # type: ignore

            ycoord = min(tde_ray)
            axs[3].axvline(x=pdi, color="black", linestyle="--", linewidth=1)
            axs[3].annotate(text=f"{int(pdi)} psi", xy=(pdi, ycoord), rotation=90)
            fig.suptitle(f"Diffuser Inlet and Outlet at {round(ptm,0)} and {round(pdi,0)} psi")  # type: ignore
        else:
            fig.suptitle(f"Diffuser Inlet at {round(ptm,0)} psi")
        plt.show()


# this goes all the way down to 200 psig
def throat_entry_book(
    psu: float, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> tuple[float, JetBook]:
    """Create Throat Entry Book

    Create a Book of Throat Entry Values that can be used for visualization.
    Shows what is occuring inside the throat entry while pressure is dropped.
    Keeps all the values, even when pte velocity is greater than Mach 1.

    Args:
        psu (float): Suction Pressure, psig
        tsu (float): Suction Temp, deg F
        ken (float): Enterance Friction Factor, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        qoil_std (float): Oil Rate, STBOPD
        te_book (JetBook): Book of values for inside the throat entry
    """
    qoil_std = ipr_su.oil_flow(psu, method="pidx")  # oil standard flow, bopd

    prop_su = prop_su.condition(psu, tsu)
    qtot = sum(prop_su.insitu_volm_flow(qoil_std))
    vte = sp.velocity(qtot, ate)

    te_book = JetBook(psu, vte, prop_su.rho_mix(), prop_su.cmix(), jf.enterance_ke(ken, vte))

    ray_len = 50  # number of elements in the array
    pte_ray = np.linspace(200, psu, ray_len)  # throat entry pressures
    pte_ray = np.flip(pte_ray, axis=0)  # start with high pressure and go low

    for pte in pte_ray[1:]:  # start with the second value, psu is the first and is used to create array

        prop_su = prop_su.condition(pte, tsu)
        qtot = sum(prop_su.insitu_volm_flow(qoil_std))
        vte = sp.velocity(qtot, ate)

        te_book.append(pte, vte, prop_su.rho_mix(), prop_su.cmix(), jf.enterance_ke(ken, vte))

    return qoil_std, te_book


def diffuser_book(
    ptm: float, ttm: float, ath: float, kdi: float, adi: float, qoil_std: float, prop_tm: ResMix
) -> tuple[float, JetBook]:
    """Create a Diffuser Book

    Create a book of diffuser arrays. The arrays are used to find where the diffuser
    pressure crosses the energy equilibrium mark and find discharge pressure.

    Args:
        ptm (float): Throat Mixture Pressure, psig
        ttm (float): Throat Mixture Temp, deg F
        ath (float): Throat Area, ft2
        kdi (float): Diffuser Friction Loss, unitless
        adi (float): Diffuser Area, ft2
        qoil_std (float): Oil Rate, STD BOPD
        prop_tm (ResMix): Properties of Throat Mixture

    Returns:
        vtm (float): Throat Mixture Velocity, ft/s
        di_book (JetBook): Book of values of what is occuring inside throat entry
    """
    prop_tm = prop_tm.condition(ptm, ttm)
    qtot = sum(prop_tm.insitu_volm_flow(qoil_std))
    vtm = sp.velocity(qtot, ath)
    vdi = sp.velocity(qtot, adi)

    di_book = JetBook(ptm, vdi, prop_tm.rho_mix(), prop_tm.cmix(), jf.diffuser_ke(kdi, vtm, vdi))

    ray_len = 30
    pdi_ray = np.linspace(ptm, ptm + 1500, ray_len)  # throat entry pressures

    for pdi in pdi_ray[1:]:  # start with 2nd value, ptm already used previously

        prop_tm = prop_tm.condition(pdi, ttm)
        qtot = sum(prop_tm.insitu_volm_flow(qoil_std))
        vdi = sp.velocity(qtot, adi)

        di_book.append(pdi, vdi, prop_tm.rho_mix(), prop_tm.cmix(), jf.diffuser_ke(kdi, vtm, vdi))

    return vtm, di_book


def multi_throat_entry_books(
    psu_ray: list | np.ndarray, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> tuple[list, list]:
    """Multiple Throat Entry Arrays

    Calculate throat entry arrays at different suction pressures. Used to
    graph later. Similiar to Figure 5 from Robert Merrill Paper.

    Args:
        psu_ray (list): List or Array of Suction Pressures, psig
        tsu (float): Suction Temp, deg F
        ken (float): Throat Entry Friction, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        qoil_list (list): List of Oil Rates
        book_list (list): List of Jet Book Results
    """
    if max(psu_ray) >= ipr_su.pres:
        raise ValueError("Max suction pressure must be less than reservoir pressure")
    # 200 is arbitary and has been hard coded into the throat entry array calcs
    if min(psu_ray) <= 200:
        raise ValueError("Min suction pressure must be greater than 200 psig")

    book_list = list()  # create empty list to fill up with results
    qoil_list = list()

    for psu in psu_ray:
        qoil_std, te_book = throat_entry_book(psu, tsu, ken, ate, ipr_su, prop_su)

        qoil_list.append(qoil_std)
        book_list.append(te_book)

    return qoil_list, book_list


def te_tde_subsonic_plot(qoil_std: float, te_book: JetBook, color: str) -> Axes:
    """Throat Entry (TE) Total Differential Energy (TDE) for Subsonic Values Plot

    Args:
        qoil_std (float): Oil Rate, BOPD
        te_book (JetBook): Throat Entry Results Book
        color (string): Matplotlib Recognized Color Description

    Return:
        ax (Axis): Matplotlib Axis to be used later
    """
    psu = te_book.prs_ray[0]
    pmo = np.interp(1, te_book.mach_ray, te_book.prs_ray)
    tde_pmo = np.interp(pmo, np.flip(te_book.prs_ray), np.flip(te_book.tde_ray))

    pte_ray = te_book.prs_ray[te_book.prs_ray >= pmo]
    tee_ray = te_book.tde_ray[te_book.prs_ray >= pmo]

    ax = plt.gca()
    ax.scatter(pte_ray, tee_ray, color=color, label=f"{int(qoil_std)} bopd, {int(psu)} psi")
    ax.scatter(pmo, tde_pmo, marker="v", color=color)  # type: ignore
    return ax


def multi_suction_graphs(qoil_list: list, book_list: list) -> None:
    """Throat Entry Graphs for Multiple Suction Pressures

    Create a graph that shows throat entry equation solutions for multiple suction pressures

    Args:
        qoil_list (list): List of Oil Rates at different suction pressures
        book_list (list): List of throat entry books at various suction pressures

    Returns:
        Graphs
    """
    plt.rcParams["mathtext.default"] = "regular"
    prop_cycle = plt.rcParams["axes.prop_cycle"]()  # convert to iterator
    fig, ax = plt.subplots(figsize=(8, 6))

    for qoil_std, te_book in zip(qoil_list, book_list):
        color = next(prop_cycle)["color"]  # new color method
        te_tde_subsonic_plot(qoil_std, te_book, color)  # need parent and children classes

    ax.set_xlabel("Throat Entry Pressure, psig")
    ax.set_ylabel("$dE_{te}$, ft2/s2")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Figure 5 of SPE-202928-MS, Mach 1 at \u25BC")
    ax.legend()

    plt.show()
