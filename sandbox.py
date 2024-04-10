import matplotlib.pyplot as plt
import numpy as np

import flow.outflow as of
from assembly import sysops as so
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

"""pwh = 210  # 210 psi
rho_pf = 62.4  # lbm/ft3
ppf_surf = 3168  # 3168 psi, power fluid surf pressure

# testing the jet pump code on E-41
tube = Pipe(out_dia=4.5, thick=0.5)  # E-42 tubing
# case = Pipe(out_dia=6.875, thick=0.5)  # E-42 casing
# ann = Annulus(inn_pipe=tube, out_pipe=case)  # define the annulus

ipr_su = InFlow(qwf=246, pwf=1049, pres=1400)  # define an ipr

e41_jp = JetPump(nozzle_no="13", area_ratio="A", ken=0.03, kth=0.3, kdi=0.4)

mpu_oil = BlackOil.schrader_oil()  # class method
mpu_wat = FormWater.schrader_wat()  # class method
mpu_gas = FormGas.schrader_gas()  # class method

form_wc = 0.894
form_gor = 600  # formation gor
form_temp = 111
prop_su = ResMix(wc=form_wc, fgor=form_gor, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)

wellprof = WellProfile.schrader()

psu_solv, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = so.jetpump_solver(
    pwh, form_temp, rho_pf, ppf_surf, e41_jp, tube, wellprof, ipr_su, prop_su
)

print(psu_solv, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te)
# create a graph that shows how psu changes as the power fluid pressure is changed
"""
# analysis for deoiler approximate gas rate

o_wat = FormWater.schrader_wat()  # Omega Pad Power Fluid
o_oil = BlackOil.schrader_oil()

pf_rate = 125000  # bwpd in powerfluid to the deoiler
oil_ppm = 7000  # ppm of oil in water


def approx_offgas(
    pin: float, pout: float, temp: float, qwat: float, oil_ppm: float, wat_prop: FormWater, oil_prop: BlackOil
) -> tuple[float, float]:
    """Calculate Rate of Gas Coming off the Deoiler

    Args:
        pin (float): Pressure Upstream Control valve, psig
        pout (float): Pressure of Deoiler, psig
        temp (float): Deg F
        qwat (float): Water Flowrate into deoiler, bwpd
        oil_ppm (float): Oil PPM in Deoiler Water Stream

    Returns:
        qoil (float): Oil Rate, bopd
        qgas (float): Gas Off Deoiler, SCF/Day
    """
    qoil = qwat * oil_ppm / 1e6
    drsw = wat_prop.condition(pin, temp).gas_solubility() - wat_prop.condition(pout, temp).gas_solubility()
    drso = oil_prop.condition(pin, temp).gas_solubility() - oil_prop.condition(pout, temp).gas_solubility()

    print(f"Difference in Water Gas Solubility: {round(drsw, 2)} scf/bbl")
    print(f"Difference in Oil Gas Solubility: {round(drso, 2)} scf/bbl")
    qgas = drso * qoil + drsw * qwat
    return qoil, qgas


qoil, qgas = approx_offgas(300, 200, 150, pf_rate, oil_ppm, o_wat, o_oil)
print(f"Oil Rate: {qoil} bopd and Gas Rate: {round(qgas/1e3, 2)} mscfd")
