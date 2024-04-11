import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from woffl.pvt.blackoil import BlackOil

# only works if the command python -m tests.boil_test is used
dirname = os.getcwd()
filename = os.path.join(dirname, r"data\oil_22_api_hysys_peng_rob.xlsx")  # , r"data\methane_hysys_peng_rob.xlsx")

hys_df = pd.read_excel(filename, header=1)

prs_ray = hys_df["pressure"]
temp = 80
hy_rho = hys_df["density"]
# hy_z_fact = hys_df["z_factor"]
hy_visc = hys_df["viscosity"]

py_rho = []
py_solub = []
py_visc = []
py_fvf = []
py_comp = []

py_boil = BlackOil.test_oil()

for prs in prs_ray:
    py_boil = py_boil.condition(prs, temp)
    py_rho.append(py_boil.density)
    py_solub.append(py_boil.gas_solubility())
    py_visc.append(py_boil.viscosity())
    py_fvf.append(py_boil.oil_fvf())
    py_comp.append(py_boil.compress())

fig, axs = plt.subplots(5, sharex=True)

axs[0].scatter(prs_ray, hy_rho, label="hysys")
axs[0].scatter(prs_ray, py_rho, label="python")
axs[0].set_ylabel("Density, lbm/ft3")
axs[0].legend()

axs[1].scatter(prs_ray, hy_visc, label="hysys")
axs[1].scatter(prs_ray, py_visc, label="python")
axs[1].set_ylabel("Viscosity, cP")
axs[1].legend()

# axs[2].scatter(prs_ray, hy_z_fact, label="hysys")
axs[2].scatter(prs_ray, py_solub, label="python")
axs[2].set_ylabel("Gas Solubility, SCF/STB")
axs[2].legend()
# axs[2].set_xlabel("Pressure, psig")

axs[3].scatter(prs_ray, py_fvf, label="python")
axs[3].set_ylabel("Oil Formation Vol. Factor, RB/STB")

ycoord = (max(py_comp) - min(py_comp)) / 10
axs[4].scatter(prs_ray, py_comp, label="python")
axs[4].set_ylabel("Oil Compressibility, psi^-1")
axs[4].axvline(x=py_boil.pbp, color="black", linestyle="--", linewidth=1)
axs[4].annotate(text="Bubble Point", xy=(py_boil.pbp, ycoord), rotation=90)
axs[4].set_xlabel("Pressure, psig")

fig.suptitle(f"Oil {py_boil.oil_api} API Properties at {temp} deg F")
plt.show()

print(f"Oil Surface Tension: {round(py_boil.tension() / 0.0000685, 2)} dyne/cm")
