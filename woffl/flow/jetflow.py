"""Jet Flow Equations

Functions that are used for solving the fluid dynamics inside a jet pump. The actual
jet pump geometry is accomplished in a seperate module.
"""

import math

import numpy as np
from scipy.integrate import trapezoid

from woffl.flow import jetplot as jp
from woffl.flow import singlephase as sp
from woffl.flow.inflow import InFlow
from woffl.pvt.resmix import ResMix


def enterance_ke(ken: float, vte: float) -> float:
    """Throat Enterance Kinetic Energy

    Calculate the kinetic energy in the throat enterance.
    Add the energy lost due to friction to balance out with EE.

    Args:
        ken: Throat Entry Friction Factor, unitless
        vte: Velocity at Throat Entry, ft/s

    Returns:
        ke_te: Throat Enterance Kinetic Energy, ft2/s2
    """
    ke_te = (1 + ken) * (vte**2) / 2
    return ke_te


def incremental_ee(prs_ray: np.ndarray, rho_ray: np.ndarray) -> float:
    """Fluid Incremental Expansion Energy

    Calculate the incremental change in expansion energy for a fluid over a
    pressure change. Uses the trapezoid rule to sum the area under the density
    curve for a difference in pressure. Equal to Sum (dp/ρ). The length of the
    arrays must match and be equal or greater than length 2.

    Args:
        prs_ray (np.ndarray): Array of pressures, psig
        rho_ray (np.ndarray): Array of densitys, lbm/ft3

    Returns:
        ee_inc (float): Expansion Energy Incremental, ft2/s2
    """
    ee_inc = trapezoid(1 / rho_ray, 144 * 32.174 * prs_ray)
    return ee_inc


# change this to a function that just creates the book?
# this only goes past crossing the zero tde line
def throat_entry_zero_tde(
    psu: float, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> tuple[float, jp.JetBook]:
    """Throat Entry Differential Energy at Zero

    Create a throat entry book where the differential energy crosses zero.
    Throat entry book can
    Use IPR to find the expected well production rate at the specific psu.

    Args:
        psu (float): Suction Pressure, psig
        tsu (float): Suction Temp, deg F
        ken (float): Throat Entry Friction, unitless
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

    te_book = jp.JetBook(psu, vte, prop_su.rho_mix(), prop_su.cmix(), enterance_ke(ken, vte))

    pdec = 25  # pressure decrease
    pmin = 50

    while (te_book.tde_ray[-1] > 0) and (te_book.prs_ray[-1] > pmin):
        pte = te_book.prs_ray[-1] - pdec

        prop_su = prop_su.condition(pte, tsu)
        qtot = sum(prop_su.insitu_volm_flow(qoil_std))
        vte = sp.velocity(qtot, ate)

        te_book.append(pte, vte, prop_su.rho_mix(), prop_su.cmix(), enterance_ke(ken, vte))

        # re-evaluate criteria for this...
        # ensures crossing zero tde while below mach limit
        # if (te_book.mach_ray[-1] > 1) and (te_book.tde_ray[-2] > 100):
        # raise ValueError(f"Suction Pressure of {psu} psig is too low. Select higher Psu.")

    # pte, vte, rho_te, mach_te = te_book.dete_zero()
    return qoil_std, te_book


# this goes until the mach number equals one
def throat_entry_mach_one(
    psu: float, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> tuple[float, float, jp.JetBook]:
    """Throat Entry Differential Energy at Mach One

    Find the numerical value of the dEte (differential energy throat entry) equation
    when the gradient (slope?) is a zero value. The physical meaning of the when the slope
    is zero is when the transistion from subsonic (mach < 1) to sonic (mach > 1) flow. A
    detailed derivation that shows the difference relationship to dEte and Ma value can
    be produced by Kaelin Ellis upon request

    Args:
        psu (float): Suction Pressure, psig
        tsu (float): Suction Temp, deg F
        ken (float): Enterance Friction Factor, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        tde_fin (float): Total Differential Energy at Mach 1, ft2/s2
        qoil_std (float): Oil Produced at psu with set IPR, bopd
        te_book (JetBook): Book of values for inside the throat entry
    """
    qoil_std = ipr_su.oil_flow(psu, method="pidx")  # oil standard flow, bopd

    prop_su = prop_su.condition(psu, tsu)
    qtot = sum(prop_su.insitu_volm_flow(qoil_std))
    vte = sp.velocity(qtot, ate)

    te_book = jp.JetBook(psu, vte, prop_su.rho_mix(), prop_su.cmix(), enterance_ke(ken, vte))

    pdec = 25  # pressure decrease
    pmin = 50  # minimum pressure

    # keep mach under one, and pte above pmin, so it doesn't go negative
    while (te_book.mach_ray[-1] <= 1) and (te_book.prs_ray[-1] > pmin):
        pte = te_book.prs_ray[-1] - pdec

        prop_su = prop_su.condition(pte, tsu)
        qtot = sum(prop_su.insitu_volm_flow(qoil_std))
        vte = sp.velocity(qtot, ate)

        te_book.append(pte, vte, prop_su.rho_mix(), prop_su.cmix(), enterance_ke(ken, vte))

    if te_book.mach_ray[-1] >= 1:  # return nearest value instead of interpolating
        tde_fin = te_book.tde_ray[-2]
    else:
        tde_fin = te_book.tde_ray[-1]

    return tde_fin, qoil_std, te_book  # type: ignore


def psu_minimize(
    tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> tuple[float, float, jp.JetBook]:
    """Minimize psu

    Find the smallest psu possible where the throat is choked. (Ma = 1)
    This psu is the theoretically smallest psu possible for a set jetpump and ipr combo.
    Even with an infinite amount of power fluid, you could not get below this psu.

    Args:
        tsu (float): Suction Temp, deg F
        ken (float): Throat Entry Friction, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        psu_min (float): Suction Pressure Minimized, psig
        qoil_std (float): Oil Rate, STBOPD
        pte (float): Throat Entry Pressure, psig
        rho_te (float): Throat Entry Density, lbm/ft3
        vte (float): Throat Entry Velocity, ft/s
    """
    psu_list = [ipr_su.pres - 300, ipr_su.pres - 400]
    # store values of tee near mach=1 pressure
    tee_list = [
        throat_entry_mach_one(psu_list[0], tsu, ken, ate, ipr_su, prop_su)[0],
        throat_entry_mach_one(psu_list[1], tsu, ken, ate, ipr_su, prop_su)[0],
    ]

    psu_diff = 5  # criteria for when you've converged to an answer
    n = 0  # loop counter
    while abs(psu_list[-2] - psu_list[-1]) > psu_diff:
        psu_nxt = psu_secant(psu_list[-2], psu_list[-1], tee_list[-2], tee_list[-1])
        tee_nxt, qoil_std, te_book = throat_entry_mach_one(psu_nxt, tsu, ken, ate, ipr_su, prop_su)
        psu_list.append(psu_nxt)
        tee_list.append(tee_nxt)
        n = n + 1
        if n == 10:
            raise ValueError("Suction Pressure for Minimization did not converge")
    # pte, vte, rho_te, mach_te = te_book.dete_zero()
    return psu_list[-1], qoil_std, te_book


def psu_secant(psu1: float, psu2: float, dete1: float, dete2: float) -> float:
    """Next Suction Pressure with Secant Method

    Uses the secant method to calculate the next psu to use to find a zero dEte at Ma = 1.

    Args:
        psu1 (float): Suction Pressure One, psig
        psu2 (float): Suction Pressure Two, psig
        dete1 (float): Differential Energy at psu1 and Ma = 1, ft2/s2
        dete2 (float): Differential Energy at psu1 and Ma = 1, ft2/s2

    Return:
        psu3 (float): Suction Pressure Three, psig
    """
    psu3 = psu2 - dete2 * (psu1 - psu2) / (dete1 - dete2)
    return psu3


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
    vnz = math.sqrt(2 * 32.174 * 144 * (pni - pte) / (rho_nz * (1 + knz)))
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
    qnz_bpd = sp.ft3s_to_bpd(qnz_ft3s)
    return qnz_ft3s, qnz_bpd


def throat_inlet_momentum(
    vnz: float, anz: float, rho_nz: float, vte: float, ate: float, rho_te: float
) -> tuple[float, float]:
    """Throat Inlet Momentum

    Calculate the inlet momentum of the throat in lbm/(s2*ft). The units
    are actually pressure, but referred to as momentum to help differentiate.

    Args:
        vnz (float): Velocity of Nozzle, ft/s
        anz (float): Area of Nozzle, ft2
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3
        vte (float): Velocity of Throat Entry Mixture, ft/s
        ate (float): Area of Throat Entry, ft2
        rho_te (float): Density of Throat Entry Mixture, lbm/ft3

    Returns:
        mom_nz (float): Nozzle Momentum, lbm*ft/s2
        mom_te (float): Entry Momentum, lbm*ft/s2
    """
    mom_nz = sp.momentum(rho_nz, vnz, anz)  # momentum of the nozzle
    mom_te = sp.momentum(rho_te, vte, ate)  # momentum of the throat entry
    return mom_nz, mom_te


def throat_outlet_momentum(kth: float, vtm: float, ath: float, rho_tm: float) -> tuple[float, float]:
    """Throat Outlet Momentum

    Calculate the outlet momentum of the throat in lbm/(s2*ft). The units
    are actually pressure, but referred to as momentum to help differentiate.

    Args:
        kth (float): Throat Friction Factor, unitless
        vtm (float): Velocity of Throat Mixture, ft/s
        ath (float): Area of the Throat, ft2
        rho_tm (float): Density of Throat Mixture, lbm/ft3

    Returns:
        mom_tm (float): Throat Mixting Momentum, lbm*ft/s2
        mom_fr (float): Throat Friction Momemum, lbm*ft/s2
    """
    mom_tm = sp.momentum(rho_tm, vtm, ath)
    mom_fr = 1 / 2 * kth * mom_tm
    return mom_tm, mom_fr


def throat_discharge(
    pte: float,
    tte: float,
    kth: float,
    vnz: float,
    anz: float,
    rho_nz: float,
    vte: float,
    ate: float,
    rho_te: float,
    prop_tm: ResMix,
):
    """Throat Discharge Pressure

    Solves the throat mixture equation of the jet pump. Calculates throat differntial pressure.
    Use the throat entry pressure and differential pressure to calculate throat mix pressure.
    Account for the discharge pressure is greater than the inlet pressure. Loops through the
    calculated discharge pressure until a converged answer occurs.

    Args:
        pte (float): Pressure of Throat Entry, psig
        tte (float): Temperature of Throat Entry, deg F
        kth (float): Friction of Throat Mix, Unitless
        vnz (float): Velocity of Nozzle, ft/s
        anz (float): Area of Nozzle, ft2
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3
        vte (float): Velocity of Throat Entry Mixture, ft/s
        ate (float): Area of Throat Entry, ft2
        rho_te (float): Density of Throat Entry Mixture, lbm/ft3
        prop_tm (ResMix): Properties of the Throat Mixture

    Returns:
        ptm (float): Throat Discharge Pressure, psig
    """
    mom_nz, mom_te = throat_inlet_momentum(vnz, anz, rho_nz, vte, ate, rho_te)

    mnz = sp.massflow(rho_nz, vnz, anz)  # mass flow of the nozzle
    mte = sp.massflow(rho_te, vte, ate)  # mass flow of the throat entry
    ath = anz + ate  # area of the throat
    mtm = mnz + mte  # mass flow of total mixture

    rho_tm = prop_tm.condition(pte, tte).rho_mix()  # density of total mixture
    vtm = sp.velocity(mtm / rho_tm, ath)

    mom_tm, mom_fr = throat_outlet_momentum(kth, vtm, ath, rho_tm)
    mom_tot = mom_fr + mom_tm - mom_nz - mom_te
    dp_tm = sp.mom_to_psi(mom_tot, ath)  # lbf/in2

    ptm = pte - dp_tm
    ptm_list = [pte, ptm]

    ptm_diff = 5
    n = 0
    while abs(ptm_list[-2] - ptm_list[-1]) > ptm_diff:
        ptm = ptm_list[-1]

        rho_tm = prop_tm.condition(ptm, tte).rho_mix()  # density of total mixture
        vtm = sp.velocity(mtm / rho_tm, ath)

        mom_tm, mom_fr = throat_outlet_momentum(kth, vtm, ath, rho_tm)
        mom_tot = mom_fr + mom_tm - mom_nz - mom_te
        dp_tm = sp.mom_to_psi(mom_tot, ath)  # lbf/in2
        ptm = pte - dp_tm
        ptm_list.append(ptm)
        n += 1
        if n == 10:
            raise ValueError("Throat Mixture did not converge")
    return ptm_list[-1]


def throat_wc(qoil_std: float, wc_su: float, qwat_nz: float) -> tuple[float, float]:
    """Throat Watercut and Formation Water Rate

    Calculate watercut and formation water rate into the jet pump throat.
    New watercut after the power fluid and reservoir fluid have mixed together.

    Args:
        qoil_std (float): Oil Rate, STD BOPD
        wc_su (float): Watercut at pump suction, decimal
        qwat_nz (float): Powerfluid Flowrate, BWPD

    Returns:
        wc_tm (float): Watercut at throat, decimal
        qwat_su (float): Formation Water at Suction, bwpd"""

    qwat_su = qoil_std * wc_su / (1 - wc_su)
    qwat_tot = qwat_nz + qwat_su
    wc_tm = qwat_tot / (qwat_tot + qoil_std)
    return wc_tm, qwat_su


def diffuser_ke(kdi: float, vtm: float, vdi: float) -> float:
    """Diffuser Kinetic Energy

    Calculate the kinetic energy in the diffuser.
    Substract the energy lost due to friction from inlet fluid.

    Args:
        kdi: Diffuser Friction Factor, unitless
        vtm: Velocity at Throat Mixture, ft/s
        vdi: Velocity at Diffuser Discharge, ft/s

    Returns:
        ke_di: Diffuser Kinetic Energy, ft2/s2
    """
    ke_di = (vdi**2 - (1 - kdi) * vtm**2) / 2
    return ke_di


def diffuser_discharge(
    ptm: float, ttm: float, kdi: float, ath: float, adi: float, qoil_std: float, prop_tm: ResMix
) -> tuple[float, float]:
    """Diffuser Discharge Pressure

    Directly calculate the diffuser discharge pressure. Only loops until the diffuser total
    energy is greater than 0. Then uses numpy interpolation for diffuser discharge pressure.

    Args:
        ptm (float): Throat Mixture Pressure, psig
        ttm (float): Throat Mixture Temp, deg F
        kdi (float): Diffuser Friction Factor, unitless
        ath (float): Throat Area, ft2
        adi (float): Diffuser / Tubing Area, ft2
        qoil_std (float): Oil Rate, STD BOPD
        prop_tm (ResMix): Properties of Throat Mixture

    Returns:
        vtm (float): Throat Mixture Velocity
        pdi (float): Diffuser Discharge Pressure, psig
    """
    prop_tm = prop_tm.condition(ptm, ttm)
    qtot = sum(prop_tm.insitu_volm_flow(qoil_std))
    vdi = sp.velocity(qtot, adi)
    vtm = sp.velocity(qtot, ath)

    di_book = jp.JetBook(ptm, vdi, prop_tm.rho_mix(), prop_tm.cmix(), diffuser_ke(kdi, vtm, vdi))

    pinc = 100  # pressure increase

    while di_book.tde_ray[-1] < 0:
        pdi = di_book.prs_ray[-1] + pinc

        prop_tm = prop_tm.condition(pdi, ttm)
        qtot = sum(prop_tm.insitu_volm_flow(qoil_std))
        vdi = sp.velocity(qtot, adi)

        di_book.append(pdi, vdi, prop_tm.rho_mix(), prop_tm.cmix(), diffuser_ke(kdi, vtm, vdi))

    pdi = di_book.dedi_zero()
    return vtm, pdi  # type: ignore


def jetpump_overall(
    psu: float,
    tsu: float,
    pni: float,
    rho_ni: float,
    ken: float,
    knz: float,
    kth: float,
    kdi: float,
    ath: float,
    anz: float,
    adi: float,
    ipr_su: InFlow,
    prop_su: ResMix,
) -> tuple[float, float, float, float, float, float, float, ResMix]:
    """Jet Pump Overall Equations

    Solve the jetpump equations, calculating out the expected discharge conditions.
    Function dete_zero() will raise a ValueError if the selected psu is too low.

    Args:
        psu (float): Suction Pressure, psig
        tsu (float): Suction Temp, deg F
        pni (float): Nozzle Inlet Pressure, psig
        rho_ni (float): Nozzle Inlet Density, lbm/ft3
        ken (float): Enterance Friction Factor, unitless
        knz (float): Nozzle Friction Factor, unitless
        kth (float): Throat Friction Factor, unitless
        kdi (float): Diffuser Friction Factor, unitless
        ath (float): Throat Area, ft2
        anz (float): Nozzle Area, ft2
        adi (float): Diffuser Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        pte (float): Throat Entry Pressure, psig
        ptm (float): Throat Mixture Pressure, psig
        pdi (float): Diffuser Discharge Pressure, psig
        qoil_std (float): Oil Rate, STBOPD
        fwat_bpd (float): Formation Water Rate, BWPD
        qnz_bpd (float): Power Fluid Rate, BWPD
        mach_te (float): Throat Entry Mach, unitless
        prop_tm (ResMix): Properties of Discharge Fluid
    """
    ate = ath - anz
    qoil_std, te_book = throat_entry_zero_tde(psu=psu, tsu=tsu, ken=ken, ate=ate, ipr_su=ipr_su, prop_su=prop_su)
    pte, vte, rho_te, mach_te = te_book.dete_zero()

    vnz = nozzle_velocity(pni, pte, knz, rho_ni)

    qnz_ft3s, qnz_bwpd = nozzle_rate(vnz, anz)
    wc_tm, fwat_bwpd = throat_wc(qoil_std, prop_su.wc, qnz_bwpd)

    prop_tm = ResMix(wc_tm, prop_su.fgor, prop_su.oil, prop_su.wat, prop_su.gas)
    ptm = throat_discharge(pte, tsu, kth, vnz, anz, rho_ni, vte, ate, rho_te, prop_tm)
    vtm, pdi = diffuser_discharge(ptm, tsu, kdi, ath, adi, qoil_std, prop_tm)
    return pte, ptm, pdi, qoil_std, fwat_bwpd, qnz_bwpd, mach_te, prop_tm
