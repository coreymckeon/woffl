import matplotlib.pyplot as plt
import numpy as np

import flow.outflow as of
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

e42_profile = WellProfile.schrader()

# only works if the command python -m tests.outflow_test is used
# mirror the hysys stuff
md_list = np.linspace(0, 6000, 100)
vd_list = np.linspace(0, 4000, 100)

# well with 600 fgor, 90% wc, 200 bopd
mpu_oil = BlackOil.schrader_oil()
mpu_wat = FormWater.schrader_wat()
mpu_gas = FormGas.schrader_gas()
form_gor = 600  # scf/stb
form_wc = 0.9
qoil_std = 500  # stbopd

test_prop = ResMix(form_wc, form_gor, mpu_oil, mpu_wat, mpu_gas)
wellprof = WellProfile(md_list, vd_list, 6000)
tubing = Pipe(out_dia=4.5, thick=0.237)

ptop = 378  # psig
ttop = 100  # deg f

md_seg, prs_ray, slh_ray = of.top_down_press(ptop, ttop, qoil_std, test_prop, tubing, wellprof)

slh_ray = np.append(slh_ray, np.nan)  # add a nan to make same length for graphing
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(md_seg, prs_ray, linestyle="--", color="b", label="Pressure")
ax1.set_ylabel("Pressure, PSIG")
ax1.set_xlabel("Measured Depth, Feet")
ax2.plot(md_seg, slh_ray, linestyle="-", color="r", label="Holdup")
ax2.set_ylabel("Slip Liquid Holdup")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
plt.show()

print(f"Bottom Pressure: {round(prs_ray[-1], 2)} psi")

# grad school work
one_gas = FormGas(gas_sg=0.67922)
two_gas = FormGas(gas_sg=0.70606)

one_gas = one_gas.condition(3045.8 - 14.7, 161.6)
two_gas = two_gas.condition(2900.7 - 14.7, 161.6)

print(f"First Gas Z-Factor is {round(one_gas.zfactor(), 4)}")
print(f"Second Gas Z-Factor is {round(two_gas.zfactor(), 4)}")

three_gas = FormGas(gas_sg=0.759)
three_gas = three_gas.condition(5000 - 14.7, 180)

print(f"Three Gas Z-Factor is {round(three_gas.zfactor(), 4)}")
print(f"Three Gas Density is {round(three_gas.density, 2)} lbm/ft3")
