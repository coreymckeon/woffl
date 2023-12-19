import math
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz

from flow.inflow import InFlow
from pvt.resmix import ResMix


# update code so JetPump is an input, for ate, atm and friction values
# currently friction values of the JetPump are nested, not great...
def actual_flow(
    oil_rate: float, poil_std: float, poil: float, yoil: float, ywat: float, ygas: float
) -> tuple[float, float, float]:
    """Actual Flow of Mixture

    Calculate the actual flow rates of the oil, water and gas in ft3/s

    Args:
        oil_rate (float): Oil Rate, BOPD
        poil_std (float): Density Oil at Std Cond, lbm/ft3
        poil (float): Density Oil Act. Cond, lbm/ft3
        yoil (float): Volm Fraction Oil Act. Cond, ft3/ft3
        ywat (float): Volm Fraction Water Act. Cond, ft3/ft3
        ygas (float): Volm Fraction Gas Act. Cond, ft3/ft3

    Returns:
        qoil (float): Oil rate, actual ft3/s
        qwat (float): Water rate, actual ft3/s
        qgas (float): Gas Rate, actual ft3/s
    """
    # 42 gal/bbl, 7.48052 gal/ft3, 24 hr/day, 60min/hour, 60sec/min
    qoil_std = oil_rate * 42 / (24 * 60 * 60 * 7.48052)  # ft3/s at standard conditions
    moil = qoil_std * poil_std
    qoil = moil / poil

    qtot = qoil / yoil  # oil flow divided by oil total fraction
    qwat = ywat * qtot
    qgas = ygas * qtot

    return qoil, qwat, qgas


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
    if np.max(mach_ray) <= 1:
        raise ValueError("Max value in Mach array is less than one, increase pte")
    if np.min(mach_ray) >= 1:
        raise ValueError("Min value in Mach array is greater than one, decrease pte")
    pmo = np.interp(1, mach_ray, pte_ray)
    return pmo


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
    rho_oil_std = prop_su.oil.condition(0, 60).density  # oil standard density
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

        rho_oil = prop_su.oil.density  # oil density
        yoil, ywat, ygas = prop_su.volm_fract()
        qoil, qwat, qgas = actual_flow(qoil_std, rho_oil_std, rho_oil, yoil, ywat, ygas)
        qtot = qoil + qwat + qgas

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
    ee_ray = cumtrapz(1 / rho_ray, plbm, initial=0)  # ft2/s2 expansion energy
    ke_ray = (1 + ken) * (vel_ray**2) / 2  # ft2/s2 kinetic energy
    return ke_ray, ee_ray


def tee_near_pmo(psu: float, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix) -> float:
    """Throat Entry Equation near pmo, mach 1 pressure

    Find the value of the throat entry equation near the mach 1 pressure. The following
    function will be iterated across to minimize the throat entry equation.

    Args:
        psu (float): Suction Press, psig
        tsu (float): Suction Temp, deg F
        ken (float): Throat Entry Friction, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        tee_pmo (float): Throat Entry Value near pmo, psig"""

    qoil_std, pte_ray, rho_ray, vel_ray, snd_ray = throat_entry_arrays(psu, tsu, ate, ipr_su, prop_su)
    pmo = throat_entry_mach_one(pte_ray, vel_ray, snd_ray)
    mask = pte_ray >= pmo  # only use values where pte_ray is greater than pmo, haven't hit mach 1
    # note: do we even have to calc  and filter pmo? TEE vs pte is a parabola which anyway...?
    # discontinuities with mask function might screw all this up...
    kse_ray, ese_ray = throat_entry_energy(ken, pte_ray[mask], rho_ray[mask], vel_ray[mask])
    tee_ray = kse_ray + ese_ray
    tee_pmo = min(tee_ray)  # find the smallest value of tee where mach <=1
    return tee_pmo


def minimize_tee(tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix) -> float:
    """Minimize Throat Entry Equation at pmo

    Find that psu that minimizes the throat entry equation for where Mach = 1 (pmo).
    Secant method for iteration, starting point is Res Pres minus 200 and 300 psig.
    Boundary equation is the starting point that Bob Merrill uses in his paper.

    Args:
        tsu (float): Suction Temp, deg F
        ken (float): Throat Entry Friction, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        psu (float): Suction Pressure, psig"""
    # how can we guarentee mach values are reached???
    psu_list = [ipr_su.pres - 300, ipr_su.pres - 400]
    # store values of tee near mach=1 pressure
    tee_list = [
        tee_near_pmo(psu_list[0], tsu, ken, ate, ipr_su, prop_su),
        tee_near_pmo(psu_list[1], tsu, ken, ate, ipr_su, prop_su),
    ]
    # criteria for when you've converged to an answer
    psu_diff = 5
    n = 0  # loop counter
    while abs(psu_list[-2] - psu_list[-1]) > psu_diff:
        # use secant method to calculate next guess value for psu to use
        psu_nxt = psu_list[-1] - tee_list[-1] * (psu_list[-2] - psu_list[-1]) / (tee_list[-2] - tee_list[-1])
        tee_nxt = tee_near_pmo(psu_nxt, tsu, ken, ate, ipr_su, prop_su)
        psu_list.append(psu_nxt)
        tee_list.append(tee_nxt)
        n = n + 1
        if n == 10:
            print("TEE Minimization did not converge")
            break
    # print(psu_list)
    # print(tee_list)
    return psu_list[-1]


def cross_zero_tee(
    ken: float, pte_ray: np.ndarray, rho_ray: np.ndarray, vel_ray: np.ndarray, snd_ray: np.ndarray
) -> tuple[float, float, float]:
    """Throat Entry Parameters with a zero TEE

    Calculate the throat entry pressure, density, and velocity where TEE crosses zero.
    Valid for one suction pressure  of the pump / reservoir.

    Args:
        ken (float): Throat Entry Friction, unitless
        pte_ray (np array): Press Throat Entry Array, psig
        rho_ray (np array): Density Throat Entry Array, lbm/ft3
        vel_ray (np array): Velocity Throat Entry Array, ft/s
        snd_ray (np array): Speed of Sound Array, ft/s

    Return:
        pte (float): Throat Entry Pressure, psig
        rho_te (float): Throat Entry Density, lbm/ft3
        vte (float): Throat Entry Velocity, ft/s
    """
    pmo = throat_entry_mach_one(pte_ray, vel_ray, snd_ray)
    mask = pte_ray >= pmo  # only use values where pte_ray is greater than pmo, haven't hit mach 1
    kse_ray, ese_ray = throat_entry_energy(ken, pte_ray[mask], rho_ray[mask], vel_ray[mask])
    tee_ray = kse_ray + ese_ray
    # is there a way to speed up all these interpolations?
    pte = np.interp(0, np.flip(tee_ray), np.flip(pte_ray[mask]))
    rho_te = np.interp(0, np.flip(tee_ray), np.flip(rho_ray[mask]))
    vte = np.interp(0, np.flip(tee_ray), np.flip(vel_ray[mask]))
    return pte, rho_te, vte


def pf_press_depth(fld_dens: float, prs_surf: float, pump_tvd: float) -> float:
    """Power Fluid Pressure at Depth

    Calculate the Power Fluid Pressure at Depth.

    Args:
        fld_dens (float): Density of Fluid, lbm/ft3
        prs_surf (float): Power Fluid Surface Pressure, psig
        pump_tvd (float): Pump True Vertical Depth, feet

    Returns:
        prs_dpth (float): Power Fluid Depth Pressure, psig
    """
    prs_dpth = prs_surf + fld_dens * pump_tvd / 144
    return prs_dpth


def nozzle_velocity(pni: float, pte: float, knz: float, rho_nz: float) -> float:
    """Nozzle Velocity

    Solve Bernoulli's Equation to calculate the nozzle velocity in ft/s.

    Args:
        pni (float): Nozzle Inlet Pressure, psig
        pte (float): Throat Entry Pressure, psig
        knz (float): Friction of Nozzle, unitless
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3

    Returns:
        vnz (float): Nozzle Velocity, ft/s
    """
    vnz = math.sqrt(2 * 32.2 * 144 * (pni - pte) / (rho_nz * (1 + knz)))
    return vnz


def nozzle_rate(vnz: float, anz: float) -> tuple[float, float]:
    """Nozzle Flow Rate

    Find Nozzle / Power Fluid Flowrate in ft3/s and BPD

    Args:
        vnz (float): Nozzle Velocity, ft/s
        anz (float): Area of Nozzle, ft2

    Returns:
        qnz_ft3s (float): Nozzle Flowrate ft3/s
        qnz_bpd (float): Nozzle Flowrate bpd
    """
    qnz_ft3s = anz * vnz
    qnz_bpd = qnz_ft3s * (7.4801 * 60 * 60 * 24 / 42)
    return qnz_ft3s, qnz_bpd


def throat_dp(kth: float, vnz: float, anz: float, rho_nz: float, vte: float, ate: float, rho_te: float):
    """Throat Differential Pressure

    Solves the throat mixture equation of the jet pump. Calculates throat differntial pressure.
    Use the throat entry pressure and differential pressure to calculate throat mix pressure.
    ptm = pte - dp_th

    Args:
        kth (float): Friction of Throat Mix, Unitless
        vnz (float): Velocity of Nozzle, ft/s
        anz (float): Area of Nozzle, ft2
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3
        vte (float): Velocity of Throat Entry Mixture, ft/s
        ate (float): Area of Throat Entry, ft2
        rho_te (float): Density of Throat Entry Mixture, lbm/ft3

    Returns:
        dp_th (float): Throat Differential Pressure, psid
    """

    mnz = vnz * anz * rho_nz  # mass flow of the mozzle
    qnz = vnz * anz  # volume flow of the nozzle

    mte = vte * ate * rho_te  # mass flow of the throat entry
    qte = vte * ate  # volume flow of the throat entry

    ath = anz + ate  # area of the throat

    mtm = mnz + mte  # mass flow of total mixture
    vtm = (vnz * anz + vte * ate) / ath  # velocity of total mixture
    rho_tm = (mnz + mte) / (qnz + qte)  # density of total mixture

    # units of lbm/(s2*ft)
    dp_tm = 0.5 * kth * rho_tm * vtm**2 + mtm * vtm / ath - mnz * vnz / ath - mte * vte / ath
    # convert to lbf/in2
    dp_tm = dp_tm / (32.174 * 144)
    return dp_tm, vtm


def throat_wc(qoil_std: float, wc_su: float, qwat_nz: float) -> float:
    """Throat Watercut

    Calculate watercut inside jet pump throat. This is the new watercut
    after the power fluid and reservoir fluid have mixed together

    Args:
        qoil_std (float): Oil Rate, STD BOPD
        wc_su (float): Watercut at pump suction, decimal
        qwat_nz (float): Powerfluid Flowrate, BWPD

    Returns:
        wc_tm (float): Watercut at throat, decimal"""

    qwat_su = qoil_std * wc_su / (1 - wc_su)
    qwat_tot = qwat_nz + qwat_su
    wc_tm = qwat_tot / (qwat_tot + qoil_std)
    return wc_tm


# other option is to just make a while loop that does something similiar
# instead of calculating all the "extra" values
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
    rho_oil_std = prop_tm.oil.condition(0, 60).density  # oil standard density
    vtm = None

    ray_len = 30  # number of elements in the array

    # create empty arrays to fill later
    vdi_ray = np.empty(ray_len)
    rho_ray = np.empty(ray_len)
    snd_ray = np.empty(ray_len)

    pdi_ray = np.linspace(ptm, ptm + 1000, ray_len)  # throat entry pressures

    for i, pdi in enumerate(pdi_ray):
        prop_tm = prop_tm.condition(pdi, ttm)

        rho_oil = prop_tm.oil.density  # oil density
        yoil, ywat, ygas = prop_tm.volm_fract()
        qoil, qwat, qgas = actual_flow(qoil_std, rho_oil_std, rho_oil, yoil, ywat, ygas)
        qtot = qoil + qwat + qgas

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
    ee_ray = cumtrapz(1 / rho_ray, plbm, initial=0)  # ft2/s2 expansion energy
    ke_ray = (vdi_ray**2 - (1 - kdi) * vtm**2) / 2  # ft2/s2 kinetic energy
    return ke_ray, ee_ray
