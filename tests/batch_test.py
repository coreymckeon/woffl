import numpy as np
import pandas as pd

from woffl.assembly.batchrun import BatchPump, batch_results_mask, batch_results_plot
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Annulus, Pipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix

# data from MPU E-41 Well Test on 11/27/2023
# only works if the command python -m tests.batch_test is used

surf_pres = 210
jpump_tvd = 4065  # feet, interpolated off well profile
rho_pf = 62.4  # lbm/ft3
ppf_surf = 3168  # psi, power fluid surf pressure 3168
tsu = 80

# testing the jet pump code on E-41
tube = Pipe(out_dia=4.5, thick=0.5)  # E-42 tubing
case = Pipe(out_dia=6.875, thick=0.5)  # E-42 casing
ann = Annulus(inn_pipe=tube, out_pipe=case)  # define the annulus

e41_ipr = InFlow(qwf=246, pwf=1049, pres=1400)  # define an ipr

mpu_oil = BlackOil.schrader()  # class method
mpu_wat = FormWater.schrader()  # class method
mpu_gas = FormGas.schrader()  # class method

form_wc = 0.894
form_gor = 600  # formation gor
form_temp = 111
e41_res = ResMix(wc=form_wc, fgor=form_gor, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)
e41_profile = WellProfile.schrader()

nozs = ["9", "10", "11", "12", "13", "14", "15", "16"]
thrs = ["X", "A", "B", "C", "D", "E"]

jp_list = BatchPump.jetpump_list(nozs, thrs)
e41_batch = BatchPump(surf_pres, tsu, rho_pf, ppf_surf, tube, e41_profile, e41_ipr, e41_res)
result_dict = e41_batch.batch_run(jp_list)

df = pd.DataFrame(result_dict)

mask_pump = batch_results_mask(df["qoil_std"], df["total_water"])

batch_results_plot(df["qoil_std"], df["total_water"], df["nozzle"], df["throat"], wellname="MPE-41", mask=mask_pump)
