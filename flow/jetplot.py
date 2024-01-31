from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid, trapezoid

from flow import jetflow as jf  # legacy
from flow.inflow import InFlow
from pvt.resmix import ResMix

# functions that are built predominately for visualizations


def throat_entry_arrays(psu: float, tsu: float, ate: float, ipr_su: InFlow, prop_su: ResMix):
    """Throat Entry Raw Arrays

    Create a series of throat entry arrays. The arrays can be graphed to visualize.
    What is occuring inside the throat entry while pressure is dropped. Keeps all the
    values, even where pte velocity is greater than mach 1.

    Args:
        psu (float): Suction Pressure, psig
        tsu (float): Suction Temp, deg F
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        qoil_std (float): Oil Rate, STD BOPD
        pte_ray (np array): Press Throat Entry Array, psig
        rho_ray (np array): Density Throat Entry Array, lbm/ft3
        vel_ray (np array): Velocity Throat Entry Array, ft/s
        snd_ray (np array): Speed of Sound Array, ft/s
    """
    qoil_std = ipr_su.oil_flow(psu, method="pidx")  # oil standard flow, bopd

    ray_len = 30  # number of elements in the array

    # create empty arrays to fill later
    vel_ray = np.empty(ray_len)
    rho_ray = np.empty(ray_len)
    snd_ray = np.empty(ray_len)

    pte_ray = np.linspace(200, psu, ray_len)  # throat entry pressures
    pte_ray = np.flip(pte_ray, axis=0)  # start with high pressure and go low

    for i, pte in enumerate(pte_ray):
        prop_su = prop_su.condition(pte, tsu)
        qtot = sum(prop_su.insitu_volm_flow(qoil_std))

        vel_ray[i] = qtot / ate
        rho_ray[i] = prop_su.pmix()
        snd_ray[i] = prop_su.cmix()

    return qoil_std, pte_ray, rho_ray, vel_ray, snd_ray


def throat_entry_energy(ken, pte_ray, rho_ray, vel_ray):
    """Energy Arrays Specific for Throat Entry

    Calculate the reservoir fluid kinetic energy and expansion energy. Return
    arrays that can be graphed for visualization.

    Args:
        ken (float): Nozzle Enterance Friction Loss, unitless
        pte_ray (np array): Press Throat Entry Array, psig
        rho_ray (np array): Density Throat Entry Array, lbm/ft3
        vel_ray (np array): Velocity Throat Entry Array, ft/s

    Returns:
        ke_ray (np array): Kinetic Energy, ft2/s2
        ee_ray (np array): Expansion Energy, ft2/s2
    """

    # convert from psi to lbm/(ft*s2)
    plbm = pte_ray * 144 * 32.174
    ee_ray = cumulative_trapezoid(1 / rho_ray, plbm, initial=0)  # ft2/s2 expansion energy
    ke_ray = (1 + ken) * (vel_ray**2) / 2  # ft2/s2 kinetic energy
    return ke_ray, ee_ray


def throat_entry_graphs(ken, pte_ray, rho_ray, vel_ray, snd_ray) -> None:
    """Throat Entry Graphs

    Create a graph to visualize what is occuring inside the throat entry section

    Args:
        ken (float): Throat Enterance Friction, unitless
        pte_ray (np array): Press Throat Entry Array, psig
        rho_ray (np array): Density Throat Entry Array, lbm/ft3
        vel_ray (np array): Velocity Throat Entry Array, ft/s
        snd_ray (np array): Speed of Sound Array, ft/s

    Returns:
        Graphs of Specific Volume, Velocity, Specific Energy, and TEE
    """
    mach_ray = vel_ray / snd_ray
    pmo = np.interp(1, mach_ray, pte_ray)  # interpolate for pressure at mach 1, pmo
    # fix this later for entrance friction loss
    kse_ray, ese_ray = throat_entry_energy(ken, pte_ray, rho_ray, vel_ray)
    tee_ray = kse_ray + ese_ray
    psuc = pte_ray[0]

    pte, rho_te, vte = jf.tee_zero(pte_ray, rho_ray, vel_ray, tee_ray)

    fig, axs = plt.subplots(4, sharex=True)
    fig.suptitle(f"Suction at {round(psuc,0)} psi, Mach 1 at {round(pmo,0)} psi")

    axs[0].scatter(pte_ray, 1 / rho_ray)
    axs[0].set_ylabel("Specific Volume, ft3/lbm")

    axs[1].scatter(pte_ray, vel_ray, label="Mixture Velocity")
    axs[1].scatter(pte_ray, snd_ray, label="Speed of Sound")
    axs[1].set_ylabel("Velocity, ft/s")
    axs[1].legend()

    axs[2].scatter(pte_ray, ese_ray, label="Expansion")
    axs[2].scatter(pte_ray, kse_ray, label="Kinetic")
    axs[2].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axs[2].set_ylabel("Specific Energy, ft2/s2")
    axs[2].legend()

    ycoord = (max(tee_ray) + min(tee_ray)) / 2
    axs[3].scatter(pte_ray, tee_ray)
    axs[3].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axs[3].axvline(x=pmo, color="black", linestyle="--", linewidth=1)
    axs[3].annotate(text="Mach 1", xy=(pmo, ycoord), rotation=90)
    axs[3].axvline(x=pte, color="black", linestyle="--", linewidth=1)
    axs[3].annotate(text="TEE 0", xy=(pte, ycoord), rotation=90)
    axs[3].set_ylabel("TEE, ft2/s2")
    axs[3].set_xlabel("Throat Entry Pressure, psig")
    plt.show()
    return None


@dataclass
class throat_entry_result:
    """Class for storing throat entry equation results, to be graphed later.

    Args:
        psu (float): Suction Pressure, psig
        qsu (float): Suction Oil Flow, bopd
        pmo (float): Mach One Pressure, psig
        pte_ray (np.ndarray): Throat Entry Array, psig
        tee_ray (np.ndarray): Throat Entry Eqn Array, ft2/s2
    """

    psu: float
    qsu: float
    pmo: float
    pte_ray: np.ndarray
    tee_ray: np.ndarray


def throat_entry_mach_one(pte_ray: np.ndarray, vel_ray: np.ndarray, snd_ray: np.ndarray) -> float:
    """Throat Entry Mach One

    Calculates the pressure where the throat entry flow hits sonic velocity, mach = 1

    Args:
        pte_ray (np array): Press Throat Entry Array, psig
        vel_ray (np array): Velocity Throat Entry Array, ft/s
        snd_ray (np array): Speed of Sound Array, ft/s

    Returns:
        pmo (float): Pressure Mach One, psig
    """
    mach_ray = vel_ray / snd_ray
    # check that the mach array has values that span one for proper interpolation
    if np.max(mach_ray) <= 1:  # type: ignore
        raise ValueError("Max value in Mach array is less than one, increase pte")
    if np.min(mach_ray) >= 1:  # type: ignore
        raise ValueError("Min value in Mach array is greater than one, decrease pte")
    pmo = np.interp(1, mach_ray, pte_ray)
    return pmo  # type: ignore


def multi_throat_entry_arrays(
    psu_min: float, psu_max: float, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> list:
    """Multiple Throat Entry Arrays

    Calculate throat entry arrays at different suction pressures. Used to
    graph later. Similiar to Figure 5 from Robert Merrill Paper.

    Args:
        psu_min (float): Suction Pressure Min, psig
        psu_max (float): Suction Pressure Max, psig, less than reservoir pressure
        tsu (float): Suction Temp, deg F
        ken (float): Throat Entry Friction, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        res_list (list): List of throat_entry_result class at various suction pressures
    """
    if psu_max >= ipr_su.pres:
        raise ValueError("Max suction pressure must be less than reservoir pressure")
    # 200 is arbitary and has been hard coded into the throat entry array calcs
    if psu_min <= 200:
        raise ValueError("Min suction pressure must be greater than 200 psig")

    ray_len = 5  # number of elements in the array

    # create empty list to fill up with results
    res_lis = list()

    psu_ray = np.linspace(psu_min, psu_max, ray_len)

    for psu in psu_ray:
        qoil_std, pte_ray, rho_ray, vel_ray, snd_ray = throat_entry_arrays(psu, tsu, ate, ipr_su, prop_su)
        pmo = throat_entry_mach_one(pte_ray, vel_ray, snd_ray)
        kse_ray, ese_ray = throat_entry_energy(ken, pte_ray, rho_ray, vel_ray)
        tee_ray = kse_ray + ese_ray

        # story the results as a class
        res_lis.append(
            throat_entry_result(
                psu=psu,
                qsu=qoil_std,
                pmo=pmo,
                pte_ray=pte_ray,  # drop suction pressure
                tee_ray=tee_ray,
            )
        )
    return res_lis


def multi_suction_graphs(res_lis: list) -> None:
    """Throat Entry Graphs for Multiple Suction Pressures

    Create a graph that shows throat entry equation solutions for multiple suction pressures

    Args:
        res_lis (list): List of throat_entry_result class at various suction pressures

    Returns:
        Graphs
    """
    # ax = plt.gca() # old matplotlib code
    for res in res_lis:
        tee_pmo = np.interp(res.pmo, np.flip(res.pte_ray), np.flip(res.tee_ray))
        # color = next(ax._get_lines.prop_cycler)["color"] # old matplotlib code
        prop_cycle = plt.rcParams["axes.prop_cycle"]  # new color method
        color = next(prop_cycle.by_key()["color"])  # new color method
        # only graph values where the mach number is under one
        pte_ray = res.pte_ray[res.pte_ray >= res.pmo]
        tee_ray = res.tee_ray[res.pte_ray >= res.pmo]
        plt.scatter(pte_ray, tee_ray, color=color, label=f"{int(res.qsu)} bopd, {int(res.psu)} psi")
        plt.scatter(res.pmo, tee_pmo, marker="v", color=color)  # type: ignore

    plt.xlabel("Throat Entry Pressure, psig")
    plt.ylabel("Throat Entry Equation, ft2/s2")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.title("Figure 5 of SPE-202928-MS, Mach 1 at \u25BC")
    plt.legend()
    plt.show()
    return None


def diffuser_arrays(ptm: float, ttm: float, ath: float, adi: float, qoil_std: float, prop_tm: ResMix):
    """Diffuser Raw Arrays

    Create diffuser arrays. The arrays are used to find where the diffuser
    pressure crosses the energy equilibrium mark and find discharge pressure.

    Args:
        ptm (float): Throat Mixture Pressure, psig
        ttm (float): Throat Mixture Temp, deg F
        ath (float): Throat Area, ft2
        adi (float): Diffuser Area, ft2
        qoil_std (float): Oil Rate, STD BOPD
        prop_tm (ResMix): Properties of Throat Mixture

    Returns:
        vtm (float): Throat Mixture Velocity, ft/s
        pdi_ray (np array): Press Diffuser Array, psig
        rho_ray (np array): Density Diffuser Array, lbm/ft3
        vdi_ray (np array): Velocity Diffuser Array, ft/s
        snd_ray (np array): Speed of Sound Array, ft/s
    """
    vtm = None
    ray_len = 30  # number of elements in the array

    # create empty arrays to fill later
    vdi_ray = np.empty(ray_len)
    rho_ray = np.empty(ray_len)
    snd_ray = np.empty(ray_len)

    pdi_ray = np.linspace(ptm, ptm + 1500, ray_len)  # throat entry pressures

    for i, pdi in enumerate(pdi_ray):
        prop_tm = prop_tm.condition(pdi, ttm)
        qtot = sum(prop_tm.insitu_volm_flow(qoil_std))

        vdi_ray[i] = qtot / adi
        rho_ray[i] = prop_tm.pmix()
        snd_ray[i] = prop_tm.cmix()
        if i == 0:
            vtm = qtot / ath

    return vtm, pdi_ray, rho_ray, vdi_ray, snd_ray


def diffuser_energy(vtm, kdi, pdi_ray, rho_ray, vdi_ray):
    """Specific Energy Arrays for Diffuser

    Calculate the jet pump fluid kinetic energy and expansion energy.
    Return arrys that can be graphed for visualization.

    Args:
        vtm (float): Velocity of throat mixture, ft/s
        kdi (float): Diffuser Friction Loss, unitless
        pdi_ray (np array): Press Diffuser Array, psig
        rho_ray (np array): Density Diffuser Array, lbm/ft3
        vdi_ray (np array): Velocity Diffuser Array, ft/s

    Returns:
        ke_ray (np array): Kinetic Energy, ft2/s2
        ee_ray (np array): Expansion Energy, ft2/s2
    """
    # convert from psi to lbm/(ft*s2)
    plbm = pdi_ray * 144 * 32.174
    ee_ray = cumulative_trapezoid(1 / rho_ray, plbm, initial=0)  # ft2/s2 expansion energy
    ke_ray = (vdi_ray**2 - (1 - kdi) * vtm**2) / 2  # ft2/s2 kinetic energy
    return ke_ray, ee_ray


def diffuser_graphs(vtm, kdi, pdi_ray, rho_ray, vdi_ray, snd_ray) -> None:
    """Diffuser Graphs

    Create a graph to visualize what is occuring in the diffuser section

    Args:
        vtm (float): Velocity of throat mixture, ft/s
        kdi (float): Diffuser Friction Factor, unitless
        pdi_ray (np array): Press Diffuser Array, psig
        rho_ray (np array): Density Diffuser Array, lbm/ft3
        vdi_ray (np array): Velocity Diffuser Array, ft/s
        snd_ray (np array): Speed of Sound in Diffuser Array, ft/s

    Returns:
        Graphs of Specific Volume, Velocity, Specific Energy, and DEE
    """
    kse_ray, ese_ray = diffuser_energy(vtm, kdi, pdi_ray, rho_ray, vdi_ray)
    dee_ray = kse_ray + ese_ray
    ptm = pdi_ray[0]

    fig, axs = plt.subplots(4, sharex=True)

    axs[0].scatter(pdi_ray, 1 / rho_ray)
    axs[0].set_ylabel("Specific Volume, ft3/lbm")

    axs[1].scatter(pdi_ray, vdi_ray, label="Diffuser Outlet")
    axs[1].scatter(pdi_ray, snd_ray, label="Speed of Sound")
    axs[1].scatter(ptm, vtm, label="Diffuser Inlet")
    axs[1].set_ylabel("Velocity, ft/s")
    axs[1].legend()

    axs[2].scatter(pdi_ray, ese_ray, label="Expansion")
    axs[2].scatter(pdi_ray, kse_ray, label="Kinetic")
    axs[2].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axs[2].set_ylabel("Specific Energy, ft2/s2")
    axs[2].legend()

    axs[3].scatter(pdi_ray, dee_ray)
    axs[3].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axs[3].set_ylabel("DEE, ft2/s2")
    axs[3].set_xlabel("Diffuser Outlet Pressure, psig")

    if max(dee_ray) >= 0 and min(dee_ray) <= 0:  # make sure a solution exists
        pdi = np.interp(0, dee_ray, pdi_ray)
        vdi = np.interp(pdi, pdi_ray, vdi_ray)
        ycoord = (min(vdi_ray) + max(snd_ray)) / 2
        axs[1].axvline(x=pdi, color="black", linestyle="--", linewidth=1)
        axs[1].annotate(text=f"{round(vdi, 1)} ft/s", xy=(pdi, ycoord), rotation=90)

        ycoord = min(dee_ray)
        axs[3].axvline(x=pdi, color="black", linestyle="--", linewidth=1)
        axs[3].annotate(text=f"{int(pdi)} psi", xy=(pdi, ycoord), rotation=90)
        fig.suptitle(f"Diffuser Inlet and Outlet at {round(ptm,0)} and {round(pdi,0)} psi")
    else:
        fig.suptitle(f"Diffuser Inlet at {round(ptm,0)} psi")
    plt.show()
