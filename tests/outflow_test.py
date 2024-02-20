import numpy as np

import flow.outflow as of
from geometry.pipe import Pipe
from geometry.wellprofile import WellProfile
from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater
from pvt.resmix import ResMix

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
qoil_std = 200  # stbopd

test_prop = ResMix(form_wc, form_gor, mpu_oil, mpu_wat, mpu_gas)
wellprof = WellProfile(md_list, vd_list, 6000)
tubing = Pipe(out_dia=4.5, thick=0.5)

ptop = 350  # psig
ttop = 100  # deg f

md_seg, prs_ray, slh_ray = of.top_down_press(ptop, ttop, qoil_std, test_prop, tubing, wellprof)
