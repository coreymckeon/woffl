import matplotlib.pyplot as plt
import numpy as np

import flow.outflow as of
from flow import jetcheck as jc
from flow import jetflow as jf
from flow import jetplot as jplt
from flow import singlephase as sp
from flow.inflow import InFlow
from geometry.jetpump import JetPump
from geometry.pipe import Annulus, Pipe
from geometry.wellprofile import WellProfile
from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater
from pvt.resmix import ResMix


def model_pump(
    is_sch: bool,
    pwh: float,
    rho_pf: float,
    ppf_surf: float,
    out_dia: float,
    thick: float,
    qwf: float,
    pwf: float,
    res_pres: float,
    nozzle_no: str,
    throat: str,
    form_wc: float,
    form_gor: float,
    form_temp: float,
    wellname: str,
):
    """Pump Model Wrapper

    Wrapper function to run jetpump_solver with the associated classes.
    Eliminates the need to separately create each class.

    Args:
        is_sch (boolean): Is the Well Schrader?
        pwh (float): Wellhead Pressure, psi
        rho_pf (float): Power fluid density, lbm/ft3
        ppf_surf (float): Power Fluid pressure at surface, psi
        out_dia (float): Tubing OD, inches
        thick (float): Tubing Wall thickness, inches
        qwf (float): Oil Rate at Flowing Bottom Hole Pressure, STBOPD
        pwf (float): Flowing Bottom Hole Pressure, psi
        res_pres (float): Reservoir Pressure, psig
        nozzle_no (str): Nozzle Size
        throat (str): Throat Ratio
        form_wc(float): Formation Water Cut, fraction
        form_gor (float): Formation Gas Oil Ratio, SCF/STB
        form_temp (float): Formation Temperature, deg F
        wellname (str): Name of Modeled Well

    Returns:
        psu_solv (float): Suction Pressure, psig
        sonic_status (boolean): Choked or not
        qoil_std (float): Oil Rate at Predicted Suction Pressure, STBOPD
        fwat_bwpd (float): Formation Water Rate, BWPD
        qnz_bwpd (float): Power Fluid Rate, BWPD
        mach_te(float): Mach Number Throat Entry, unitless
        total_wc (float): Total Water Cut (PF + Form), fraction
        total_water (float): Total Water Rate (PF + Form), BWPD
        wellname (str): Wellname

    """
    if is_sch:
        mpu_oil = BlackOil.schrader_oil()  # class method
        mpu_wat = FormWater.schrader_wat()  # class method
        mpu_gas = FormGas.schrader_gas()  # class method

        wellprof = WellProfile.schrader()

    else:
        mpu_oil = BlackOil.kuparuk_oil()  # class method
        mpu_wat = FormWater.kuparuk_wat()  # class method
        mpu_gas = FormGas.kuparuk_gas()  # class method

        wellprof = WellProfile.kuparuk()

    tube = Pipe(out_dia, thick)

    ipr_su = InFlow(qwf, pwf, res_pres)

    prop_su = ResMix(wc=form_wc, fgor=form_gor, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)

    jp = JetPump(nozzle_no, throat, ken=0.03, kth=0.3, kdi=0.4)

    psu_solv, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = jc.jetpump_solver(
        pwh, form_temp, rho_pf, ppf_surf, jp, tube, wellprof, ipr_su, prop_su
    )
    total_water = qnz_bwpd + fwat_bwpd
    total_wc = (qnz_bwpd + fwat_bwpd) / (qnz_bwpd + fwat_bwpd + qoil_std)

    print(f"\nResults for well: {wellname} ")
    print(f"JP Size: {nozzle_no} {throat}")
    print(f"Suction pressure: {psu_solv:.0f}")
    print(f"Sonic Status: {sonic_status}")
    print(f"Oil Rate: {qoil_std:.0f}")
    print(f"Water Rate: {fwat_bwpd:.0f}")
    print(f"PF Rate: {qnz_bwpd:.0f}")
    print(f"Mach TE: {mach_te:.2f}")
    print(f"Total WC: {total_wc:.2f}\n")

    return psu_solv, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te, total_wc, total_water, wellname
