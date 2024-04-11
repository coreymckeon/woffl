"""Out Flow for an Oil Well

Performs the calculations that occur inside a vertical, inclined or horizontal wellbore.
Predominate equations are two phase flow correlations. The most common being Beggs and Brill
"""

import numpy as np

from woffl.flow import singlephase as sp
from woffl.flow import twophase as tp
from woffl.geometry import forms as fm
from woffl.geometry.pipe import Annulus, Pipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.resmix import ResMix


def homo_diff_press(
    pin: float, tin: float, inn_dia: float, abs_ruff: float, length: float, height: float, qoil_std: float, prop: ResMix
) -> tuple[float, float, float]:
    """Homogenous Differential Pressure

    Calculate Homogenous Differential Pressure Across Wellbore / Pipe.
    Note, could have positive / negative length and height, depending on which node you
    are starting with. EG, starting at bottom and going up or starting at top and going down.

    Args:
        pin (float): Inlet Pressure, psig
        tin (float): Inlet Temperature, deg F
        inn_dia (float): Inner Diameter of Pipe, inches
        abs_ruff (float): Absolute Roughness of Pipe, inches
        length (float): Length of Pipe, feet, Positive is with flow direction
        height (float): Height Difference of Pipe, feet, Positive is upwards
        qoil_std (float): Oil Rate, STBOPD
        prop (ResMix): Properties of Fluid Mixture in Wellbore, ResMix

    Returns:
        dp (float): Differential Pressure, psid
        nslh (float): No Slip Liquid Holdup, unitless
    """
    prop = prop.condition(pin, tin)
    qtot = sum(prop.insitu_volm_flow(qoil_std))
    rho_mix = prop.rho_mix()
    visc_mix = prop.visc_mix()
    nslh = prop.nslh()

    area = sp.cross_area(inn_dia)
    vmix = sp.velocity(qtot, area)
    NRe = sp.reynolds(rho_mix, vmix, inn_dia, visc_mix)
    rel_ruff = sp.relative_roughness(inn_dia, abs_ruff)
    ff = sp.ffactor_darcy(NRe, rel_ruff)

    dp_fric = sp.diff_press_friction(ff, rho_mix, vmix, inn_dia, length)
    dp_stat = sp.diff_press_static(rho_mix, height)
    return dp_stat, dp_fric, nslh


def beggs_diff_press(
    pin: float, tin: float, inn_dia: float, abs_ruff: float, length: float, height: float, qoil_std: float, prop: ResMix
) -> tuple[float, float, float]:
    """Beggs Differential Pressure

    Calculate Beggs Differential Pressure Across Wellbore / Pipe.
    Note, could have positive / negative length and height, depending on which node you
    are starting with. EG, starting at bottom and going up or starting at top and going down.

    Args:
        pin (float): Inlet Pressure, psig
        tin (float): Inlet Temperature, deg F
        inn_dia (float): Inner Diameter of Pipe, inches
        abs_ruff (float): Absolute Roughness of Pipe, inches
        length (float): Length of Pipe, feet, Positive is with flow direction
        height (float): Height Difference of Pipe, feet, Positive is upwards
        qoil_std (float): Oil Rate, STBOPD
        prop (ResMix): Properties of Fluid Mixture in Wellbore, ResMix

    Returns:
        dp_stat (float): Static Differential Pressure, psid
        dp_fric (float): Friction Differntial Pressure, psid
        slh (float): Slip Liquid Holdup, unitless
    """
    prop = prop.condition(pin, tin)
    qoil, qwat, qgas = prop.insitu_volm_flow(qoil_std)
    rho_liq, rho_gas = prop.rho_two()
    sig_liq = prop.tension()
    nslh = prop.nslh()

    visc_mix = prop.visc_mix()
    rho_mix = prop.rho_mix()

    area = sp.cross_area(inn_dia)
    vsl, vsg, vmix = tp.velocities(qoil, qwat, qgas, area)
    NFr = tp.froude(vmix, inn_dia)
    NLv = tp.ros_nlv(vsl, rho_liq, sig_liq)  # ros liquid velocity number

    hpat, tran = tp.beggs_flow_pattern(nslh, NFr)
    incline = fm.horz_angle(length, height)
    slh = tp.beggs_holdup_inc(nslh, NFr, NLv, incline, hpat, tran)
    slh = min(slh, 1)  # liquid holdup never above one (before or after payne?)
    slh = tp.payne_correction(slh, incline)  # 1979 correction
    rho_slip = tp.density_slip(rho_liq, rho_gas, slh)
    dp_stat = tp.beggs_press_static(rho_slip, height)

    NRe = sp.reynolds(rho_mix, vmix, inn_dia, visc_mix)
    rel_ruff = sp.relative_roughness(inn_dia, abs_ruff)
    ff = sp.ffactor_darcy(NRe, rel_ruff)  # make this nomenclature consistent with beggs?
    yb = tp.beggs_yf(nslh, slh)  # NSLh, Lh
    sb = tp.beggs_sf(yb)
    fb = tp.beggs_ff(ff, sb)
    dp_fric = tp.beggs_press_friction(fb, rho_mix, vmix, inn_dia, length)
    return dp_stat, dp_fric, slh


def top_down_press(
    ptop: float, ttop: float, qoil_std: float, prop: ResMix, tubing: Pipe, wellprof: WellProfile, model: str = "beggs"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top Down WellBore Pressure Calculation

    Uses the specificed model to calculate the pressure gradient in a wellbore, starting
    at the top and working down to inflow / jet pump node. This is the preferred method
    of calculation, as a negative pressure cannot be created during calculations.

    Args:
        ptop (float): Pressure at top node, psig
        ttop (float): Temperature at top node, deg F
        qoil_std (float): Oil Rate, STBOPD
        prop (ResMix): Properties of Fluid Mixture in Wellbore, ResMix
        tubing (Pipe): Piping geometry inside the wellbore, Pipe
        wellprof (WellProfile): survey dimensions and location of jet pump, WellProfile
        model (str): Specify which model to use, either homo or beggs

    Returns:
        md_seg (list): Measured depth of calculated pressure
        prs_ray (list): Calculated pressure along wellbore, psig
        slh_ray (list): Liquid Holdup along wellbore, unitless
    """
    # mp_models = {'homo': homo_diff_press(), 'beggs': beggs_diff_press()}
    prs_ray = np.array([ptop])
    slh_ray = np.array([])
    md_seg, vd_seg = wellprof.outflow_spacing(100)  # space every 100'
    md_diff = np.diff(md_seg, n=1) * -1  # against flow
    vd_diff = np.diff(vd_seg, n=1) * -1  # going down piping
    n = 0
    for length, height in zip(md_diff, vd_diff):
        dp_stat, dp_fric, slh = beggs_diff_press(
            prs_ray[-1], ttop, tubing.inn_dia, tubing.abs_ruff, length, height, qoil_std, prop
        )
        pdwn = prs_ray[-1] - dp_stat - dp_fric  # dp is subtracted
        prs_ray = np.append(prs_ray, pdwn)
        slh_ray = np.append(slh_ray, slh)
        n = n + 1
    # the no slip array is going to be one shorter than the md_seg and prs_ray...
    # i'm not sure if this is problem that I should "fix" later?
    return md_seg, prs_ray, slh_ray


def bottom_up_press(
    pbot: float, tbot: float, qoil_std: float, prop: ResMix, tubing: Pipe, wellprof: WellProfile, model: str = "beggs"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NOT TESTED: Bottom Up WellBore Pressure Calculation

    Uses the specificed model to calculate the pressure gradient in a wellbore, starting
    at the bottom inflow / jet pump node and working up to ther surface. Problem with this
    method is that a negative pressure can be calculated, causing an error.

    Args:
        pbot (float): Pressure at bottom node, psig
        tbot (float): Temperature at bottom node, deg F
        qoil_std (float): Oil Rate, STBOPD
        prop (ResMix): Properties of Fluid Mixture in Wellbore, ResMix
        tubing (Pipe): Piping geometry inside the wellbore, Pipe
        wellprof (WellProfile): survey dimensions and location of jet pump, WellProfile
        model (str): Specify which model to use, either homo or beggs

    Returns:
        md_seg (list): Measured depth of calculated pressure
        prs_ray (list): Calculated pressure along wellbore, psig
        slh_ray (list): Liquid Holdup along wellbore, unitless
    """
    # mp_models = {'homo': homo_diff_press(), 'beggs': beggs_diff_press()}
    prs_ray = np.array([pbot])
    slh_ray = np.array([])
    md_seg, vd_seg = wellprof.outflow_spacing(100)  # space every 100'
    md_diff = np.diff(md_seg, n=1)  # with the flow
    vd_diff = np.diff(vd_seg, n=1)  # going up piping
    n = 0
    for length, height in zip(np.flip(md_diff), np.flip(vd_diff)):  # start at bottom
        dp_stat, dp_fric, slh = beggs_diff_press(
            prs_ray[-1], tbot, tubing.inn_dia, tubing.abs_ruff, length, height, qoil_std, prop
        )
        pdwn = prs_ray[-1] - dp_stat - dp_fric  # dp is subtracted
        prs_ray = np.append(prs_ray, pdwn)
        slh_ray = np.append(slh_ray, slh)
        n = n + 1
    # the no slip array is going to be one shorter than the md_seg and prs_ray...
    # i'm not sure if this is problem that I should "fix" later?
    return md_seg, prs_ray, slh_ray
