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

pwh = 210
rho_pf = 62.4  # lbm/ft3
ppf_surf = 3168  # 3168 psi, power fluid surf pressure

# testing the jet pump code on E-41
tube = Pipe(out_dia=4.5, thick=0.5)  # E-42 tubing
case = Pipe(out_dia=6.875, thick=0.5)  # E-42 casing
ann = Annulus(inn_pipe=tube, out_pipe=case)  # define the annulus

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

psu_solv, flow_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = jc.jetpump_solver(
    pwh, form_temp, rho_pf, ppf_surf, e41_jp, tube, wellprof, ipr_su, prop_su
)

print(psu_solv, flow_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te)
# create a graph that shows how psu changes as the power fluid pressure is changed
