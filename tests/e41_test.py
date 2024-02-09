from flow import jetcheck as jc
from flow import jetflow as jf

# from flow import jetplot as jplt
from flow.inflow import InFlow
from geometry.jetpump import JetPump
from geometry.pipe import Annulus, Pipe
from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater
from pvt.resmix import ResMix

# data from MPU E-41 Well Test on 11/27/2023
# only works if the command python -m tests.e41_test is used

jpump_tvd = 4065  # feet, interpolated off well profile
rho_pf = 62.4  # lbm/ft3
ppf_surf = 3168  # psi, power fluid surf pressure

# testing the jet pump code on E-42
tube = Pipe(out_dia=4.5, thick=0.5)  # E-42 tubing
case = Pipe(out_dia=6.875, thick=0.5)  # E-42 casing
ann = Annulus(inn_pipe=tube, out_pipe=case)  # define the annulus

e41_ipr = InFlow(qwf=246, pwf=1049, pres=1400)  # define an ipr

e41_jp = JetPump(nozzle_no="13", area_ratio="A", ken=0.03, kth=0.3, kdi=0.4)

mpu_oil = BlackOil.schrader_oil()  # class method
mpu_wat = FormWater.schrader_wat()  # class method
mpu_gas = FormGas.schrader_gas()  # class method

form_wc = 0.894
form_gor = 600  # formation gor
form_temp = 111
e41_res = ResMix(wc=form_wc, fgor=form_gor, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)

jc.jet_check(form_temp, jpump_tvd, rho_pf, ppf_surf, e41_jp, tube, e41_ipr, e41_res)

# psu = 838
# vnz = jf.nozzle_velocity(4889, psu, e41_jp.knz, rho_pf)
# qnz_ft3s, qnz_bpd = jf.nozzle_rate(vnz, e41_jp.anz)
# print(f"Using suction pressure {psu} psig, power fluid rate is {round(qnz_bpd, 1)} bwpd")
