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
    is_sch: TRUE/FALSE is the well schrader? to determine PVT and well geometry,
    pwh: wellhead pressure psi,
    rho_pf: power fluid density lbm/ft3,
    ppf_surf: PF pressure at surface, psi,
    out_dia: tubing OD inches,
    thick: tubing wall thickness inches,
    qwf: Oil rate BOPD,
    pwf: FBHP psi,
    res_pres: reservoir pressure,
    nozzle_no: nozzle size ,
    throat: throat ratio,
    form_wc: formation watercut,
    form_gor: formation gor,
    form_temp: formation temp,
    wellname: name of well being modeled,
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


psu_solv, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te, total_wc, total_water, wellname = model_pump(
    True,  # isSchrader?
    pwh=210,  # WHP
    rho_pf=62.4,  # PF density
    ppf_surf=3168,  # PF pres
    out_dia=4.5,  # Tubing OD
    thick=0.5,  # tubing thickness
    qwf=246,  # Oil rate
    pwf=1049,  # FBHP
    res_pres=1400,  # res pressure
    form_wc=0.894,  # watercut
    form_gor=600,  # gor
    form_temp=111,  # form temp
    nozzle_no="13",  # nozzle size
    throat="A",  # nozzel area ratio with throat
    wellname="MPB-28",
)
