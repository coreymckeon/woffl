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
    """
    Wrapper function to run jetpump_solver with all the associated classes without having to call every class.
    
    Args:
        is_sch (bool): TRUE/FALSE is the well schrader? to determine PVT and well geometry,
        pwh (float): wellhead pressure psi,
        rho_pf (float): power fluid density lbm/ft3,
        ppf_surf (float): PF pressure at surface, psi,
        out_dia (float): tubing OD inches,
        thick (float): tubing wall thickness inches,
        qwf (float): Oil rate BOPD,
        pwf (float): FBHP psi,
        res_pres (float): reservoir pressure,
        nozzle_no (str): nozzle size ,
        throat (str): throat ratio,
        form_wc(float): formation watercut,
        form_gor (float): formation gor,
        form_temp (float): formation temp,
        wellname (str): name of well being modeled,

    Returns:
        psu_solv (float): suction pressure
        sonic_status (bool): Choked or not
        qoil_std (float): oil rate
        fwat_bwpd (float): water rate
        qnz_bwpd (float): PF rate
        mach_te(float): Mach number
        total_wc (float): watercut
        total_water (float): Total water PF + formation
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

