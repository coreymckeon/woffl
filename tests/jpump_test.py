import matplotlib.pyplot as plt
import numpy as np

import woffl.flow.outflow as of
from woffl.flow import jetflow as jf
from woffl.flow import jetplot as jplt
from woffl.flow import singlephase as sp
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Annulus, Pipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix

# data from MPU E-41 Well Test on 11/27/2023
# only works if the command python -m tests.jpump_test is used

pwh = 210
rho_pf = 62.4  # lbm/ft3
ppf_surf = 3168  # psi, power fluid surf pressure

# testing the jet pump code on E-41
tube = Pipe(out_dia=4.5, thick=0.5)  # E-42 tubing
case = Pipe(out_dia=6.875, thick=0.5)  # E-42 casing
ann = Annulus(inn_pipe=tube, out_pipe=case)  # define the annulus

ipr_su = InFlow(qwf=246, pwf=1049, pres=1400)  # define an ipr

e41_jp = JetPump(nozzle_no="13", area_ratio="A", ken=0.03, kth=0.3, kdi=0.4)

mpu_oil = BlackOil.schrader()  # class method
mpu_wat = FormWater.schrader()  # class method
mpu_gas = FormGas.schrader()  # class method

form_wc = 0.894
form_gor = 600  # formation gor
form_temp = 111
prop_su = ResMix(wc=form_wc, fgor=form_gor, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)

wellprof = WellProfile.schrader()

# find the minimum psu, then find maximum, calculate pdi and compare?
# also calculate pdi_of for each case / residual?

psu_min, qoil_std, te_book = jf.psu_minimize(form_temp, e41_jp.ken, e41_jp.ate, ipr_su, prop_su)
psu_max = ipr_su.pres - 10

psu_list = np.linspace(psu_min, psu_max, 10)

pte_list = []
ptm_list = []
pdi_list = []
qoil_list = []

pni = ppf_surf + sp.diff_press_static(rho_pf, wellprof.jetpump_vd)  # static

for psu in psu_list:
    pte, ptm, pdi, qoil_std, fwat_bwpd, qnz_bwpd, mach_te, prop_tm = jf.jetpump_overall(
        psu,
        form_temp,
        pni,
        rho_pf,
        e41_jp.ken,
        e41_jp.knz,
        e41_jp.kth,
        e41_jp.kdi,
        e41_jp.ath,
        e41_jp.anz,
        tube.inn_area,
        ipr_su,
        prop_su,
    )

    pte_list.append(pte)
    ptm_list.append(ptm)
    pdi_list.append(pdi)
    qoil_list.append(qoil_std)

plt.scatter(psu_list, pdi_list, label="Discharge")
plt.scatter(psu_list, ptm_list, label="Throat Mix")
plt.scatter(psu_list, pte_list, label="Throat Entry")
plt.xlabel("Suction Pressure, psig")
plt.ylabel("Pressure, psig")
plt.title("Comparison of Jet Pump Pressures Against Suction")
plt.legend()
plt.show()
