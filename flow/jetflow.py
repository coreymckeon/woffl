import math

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

from flow.inflow import InFlow
from pvt.resmix import ResMix


# update code so JetPump is an input, for ate, atm and friction values
# tee_final or tee_mach_one
def tee_final(
    psu: float, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Throat Enterance Final Energy

    Calculate the amount of energy in the throat enterance when the flow
    hits sonic velocity, mach = 1 or when the pte is low on pressure (< pdecrease).
    The suction pressure, psu and final enterance energy are fed into a secant
    solver that finds psu that gives a zero final energy.

    Args:
        psu (float): Suction Pressure, psig
        tsu (float): Suction Temp, deg F
        ken (float): Enterance Friction Factor, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        tee_fin (float): Final Throat Energy Equation Value, ft2/s2
        qoil_std (float): Oil flow from reservoir, stbopd
        pte_ray
        rho_ray
        vte_ray
        tee_ray
    """
    # rho_oil_std = prop_su.oil.condition(0, 60).density  # oil standard density # legacy

    qoil_std = ipr_su.oil_flow(psu, method="pidx")  # oil standard flow, bopd
    prop_su = prop_su.condition(psu, tsu)
    qtot = sum(prop_su.insitu_volm_flow(qoil_std))

    # qtot = total_actual_flow(qoil_std, rho_oil_std, prop_su) # legacy
    vte = qtot / ate

    pte_ray = np.array([psu])
    vte_ray = np.array([vte])
    rho_ray = np.array([prop_su.pmix()])
    # snd_ray = np.array([prop_su.cmix()])
    mach_ray = np.array([vte / prop_su.cmix()])

    kse_ray = np.array([(1 + ken) * (vte**2) / 2])
    ese_ray = np.array([0])
    tee_ray = np.array([kse_ray + ese_ray])

    pdec = 50  # pressure decrease
    # keep mach under one, and pte above pdec, so it doesn't go negative
    while mach_ray[-1] <= 1 and pte_ray[-1] > pdec:
        pte = pte_ray[-1] - pdec

        prop_su = prop_su.condition(pte, tsu)
        # qtot = total_actual_flow(qoil_std, rho_oil_std, prop_su) # legacy
        qtot = sum(prop_su.insitu_volm_flow(qoil_std))
        vte = qtot / ate

        vte_ray = np.append(vte_ray, vte)
        kse_ray = np.append(kse_ray, (1 + ken) * (vte**2) / 2)

        # snd_ray = np.append(snd_ray, prop_su.cmix())
        mach_ray = np.append(mach_ray, vte / prop_su.cmix())

        pte_ray = np.append(pte_ray, pte)
        rho_ray = np.append(rho_ray, prop_su.pmix())

        sv_int = trapezoid(1 / rho_ray[-2:], 144 * 32.174 * pte_ray[-2:])
        ese_ray = np.append(ese_ray, ese_ray[-1] + sv_int)
        tee_ray = np.append(tee_ray, kse_ray[-1] + ese_ray[-1])

    if mach_ray[-1] >= 1:
        tee_fin = np.interp(1, mach_ray, tee_ray)
    else:
        tee_fin = tee_ray[-1]

    return tee_fin, qoil_std, pte_ray, rho_ray, vte_ray, tee_ray


def tee_zero(
    pte_ray: np.ndarray, rho_ray: np.ndarray, vte_ray: np.ndarray, tee_ray: np.ndarray
) -> tuple[float, float, float]:
    """Throat Entry Parameters with zero TEE

    Calculate the throat entry pressure, density, and velocity where TEE crosses zero.
    Valid for one suction pressure  of the pump / reservoir.

    Args:
        pte_ray (np array): Press Throat Entry Array, psig
        rho_ray (np array): Density Throat Entry Array, lbm/ft3
        vte_ray (np array): Velocity Throat Entry Array, ft/s
        tee_ray (np array): Throat Entry Equation Array, ft/s

    Return:
        pte (float): Throat Entry Pressure, psig
        rho_te (float): Throat Entry Density, lbm/ft3
        vte (float): Throat Entry Velocity, ft/s
    """

    pte = np.interp(0, np.flip(tee_ray), np.flip(pte_ray))
    rho_te = np.interp(0, np.flip(tee_ray), np.flip(rho_ray))
    vte = np.interp(0, np.flip(tee_ray), np.flip(vte_ray))

    return pte, rho_te, vte


def tee_minimize(
    tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> tuple[float, float, float, float, float]:
    """Minimize Throat Entry Equation at pmo

    Find that psu that minimizes the throat entry equation for where Mach = 1 (pmo).
    Secant method for iteration, starting point is Res Pres minus 200 and 300 psig.

    Args:
        tsu (float): Suction Temp, deg F
        ken (float): Throat Entry Friction, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        psu (float): Suction Pressure, psig"""
    psu_list = [ipr_su.pres - 300, ipr_su.pres - 400]
    # store values of tee near mach=1 pressure
    tee_list = [
        tee_final(psu_list[0], tsu, ken, ate, ipr_su, prop_su)[0],
        tee_final(psu_list[1], tsu, ken, ate, ipr_su, prop_su)[0],
    ]
    psu_diff = 5  # criteria for when you've converged to an answer
    n = 0  # loop counter
    while abs(psu_list[-2] - psu_list[-1]) > psu_diff:
        # use secant method to calculate next guess value for psu to use
        psu_nxt = psu_list[-1] - tee_list[-1] * (psu_list[-2] - psu_list[-1]) / (tee_list[-2] - tee_list[-1])
        tee_nxt, qoil_std, pte_ray, rho_ray, vte_ray, tee_ray = tee_final(psu_nxt, tsu, ken, ate, ipr_su, prop_su)
        psu_list.append(psu_nxt)
        tee_list.append(tee_nxt)
        n = n + 1
        if n == 10:
            print("TEE Minimization did not converge")
            break
    pte, rho_te, vte = tee_zero(pte_ray, rho_ray, vte_ray, tee_ray)  # type: ignore
    return psu_list[-1], qoil_std, pte, rho_te, vte  # type: ignore


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
    ptm = pte - dp_th. The biggest issue with this equation is it assumes the discharge conditions
    are at same conditions as the inlet. This is false. There is an increase in pressure across the
    diffuser. Which using this equation equates potentially 600 psig.

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
    """Throat Differential Pressure

    Solves the throat mixture equation of the jet pump. Calculates throat differntial pressure.
    Use the throat entry pressure and differential pressure to calculate throat mix pressure.
    Account for the discharge pressure is greater than the inlet pressure.

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
    mnz = vnz * anz * rho_nz  # mass flow of the nozzle
    mte = vte * ate * rho_te  # mass flow of the throat entry
    ath = anz + ate  # area of the throat
    mtm = mnz + mte  # mass flow of total mixture

    rho_tm = prop_tm.condition(pte, tte).pmix()  # density of total mixture
    vtm = mtm / (ath * rho_tm)  # velocity of total mixture
    # units of lbm/(s2*ft)
    dp_tm = 0.5 * kth * rho_tm * vtm**2 + mtm * vtm / ath - mnz * vnz / ath - mte * vte / ath
    # convert to lbf/in2
    dp_tm = dp_tm / (32.174 * 144)
    ptm = pte - dp_tm
    ptm_list = [pte, ptm]
    ptm_diff = 5
    n = 0
    while abs(ptm_list[-2] - ptm_list[-1]) > ptm_diff:
        ptm = ptm_list[-1]

        rho_tm = prop_tm.condition(ptm, tte).pmix()  # density of total mixture
        vtm = mtm / (ath * rho_tm)  # velocity of total mixture
        # units of lbm/(s2*ft)
        dp_tm = 0.5 * kth * rho_tm * vtm**2 + mtm * vtm / ath - mnz * vnz / ath - mte * vte / ath
        # convert to lbf/in2
        dp_tm = dp_tm / (32.174 * 144)
        ptm = pte - dp_tm
        ptm_list.append(ptm)
        n = n + 1
        if n == 10:
            print("Throat Mixture did not converge")
            break
    print(f"Throat took {n} loops to find solution")
    return ptm_list[-1]


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


# def diffuser_kinetic ?
# def diffuser_expanse ?


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
    # rho_oil_std = prop_tm.oil.condition(0, 60).density  # oil standard density # legacy

    prop_tm = prop_tm.condition(ptm, ttm)
    qtot = sum(prop_tm.insitu_volm_flow(qoil_std))
    # qtot = total_actual_flow(qoil_std, rho_oil_std, prop_tm) # legacy

    vtm = qtot / ath
    vdi = qtot / adi

    pdi_ray = np.array([ptm])
    vdi_ray = np.array([vdi])
    rho_ray = np.array([prop_tm.pmix()])
    # snd_list = [prop_tm.cmix()]

    kse_ray = np.array([(vdi**2 - (1 - kdi) * vtm**2) / 2])
    ese_ray = np.array([0])
    dte_ray = np.array([kse_ray + ese_ray])

    n = 0
    pinc = 100  # pressure increase

    while dte_ray[-1] < 0:
        pdi = pdi_ray[-1] + pinc

        prop_tm = prop_tm.condition(pdi, ttm)
        qtot = sum(prop_tm.insitu_volm_flow(qoil_std))
        # qtot = total_actual_flow(qoil_std, rho_oil_std, prop_tm) # legacy
        vdi = qtot / adi

        vdi_ray = np.append(vdi_ray, vdi)
        kse_ray = np.append(kse_ray, (vdi**2 - (1 - kdi) * vtm**2) / 2)

        pdi_ray = np.append(pdi_ray, pdi)
        rho_ray = np.append(rho_ray, prop_tm.pmix())
        # specific volume integration
        sv_int = trapezoid(1 / rho_ray[-2:], 144 * 32.174 * pdi_ray[-2:])
        ese_ray = np.append(ese_ray, ese_ray[-1] + sv_int)
        dte_ray = np.append(dte_ray, kse_ray[-1] + ese_ray[-1])

        n = n + 1
        if n == 15:
            print("Diffuser did not find discharge pressure")
            break

    # print(f"Diffuser took {n} loops to find solution")
    pdi = np.interp(0, dte_ray, pdi_ray)

    return vtm, pdi
