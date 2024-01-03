import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pvt.formgas import FormGas

# only works if the command python -m tests.fgas_test is used
dirname = os.getcwd()
filename = os.path.join(dirname, r"data\methane_hysys_peng_rob.xlsx")  # , r"data\methane_hysys_peng_rob.xlsx")

hys_df = pd.read_excel(filename, header=1)

prs_ray = hys_df["pressure"]
temp = 80
hy_rho = hys_df["density"]
hy_z_fact = hys_df["z_factor"]
hy_visc = hys_df["viscosity"]

py_rho = []
py_z_fact = []
py_visc = []

py_meth = FormGas.methane_gas()

for prs in prs_ray:
    py_meth = py_meth.condition(prs, temp)
    py_rho.append(py_meth.density)
    py_z_fact.append(py_meth.zfactor())
    py_visc.append(py_meth.viscosity())

fig, axs = plt.subplots(3, sharex=True)

axs[0].scatter(prs_ray, hy_rho, label="hysys")
axs[0].scatter(prs_ray, py_rho, label="python")
axs[0].set_ylabel("Density, lbm/ft3")
axs[0].legend()

axs[1].scatter(prs_ray, hy_visc, label="hysys")
axs[1].scatter(prs_ray, py_visc, label="python")
axs[1].set_ylabel("Viscosity, cP")
axs[1].legend()

axs[2].scatter(prs_ray, hy_z_fact, label="hysys")
axs[2].scatter(prs_ray, py_z_fact, label="python")
axs[2].set_ylabel("Z-Factor, Unitless")
axs[2].legend()
axs[2].set_xlabel("Pressure, psig")

fig.suptitle(f"Pure Methane Properties at {temp} deg F")

plt.show()
