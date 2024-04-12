import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from flow import jetflow as jf # for actual flow conditions
from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

# test python reservoir mixture equations vs hysys
# only works if the command python -m tests.rmix_test is used
# q_oil_std = 100  # bopd

temp = 80
oil_api = 22
pbub = 1750  # psig
gas_sg = 0.55  # methane
wc = 0.3  # watercut
fgor = 800  # scf/stb

# mass fraction, volm fraction, mixture density, speed of sound (no hysys...)
dirname = os.getcwd()
filename = os.path.join(dirname, r"data\resmix_hysys_peng_rob.xlsx")

hy_df = pd.read_excel(filename, header=1)

prs_ray = hy_df["press"]
hy_mfac = [hy_df["oil_mfac"], hy_df["wat_mfac"], hy_df["gas_mfac"]]
hy_vfac = [hy_df["oil_vfac"], hy_df["wat_vfac"], hy_df["gas_vfac"]]
hy_rho_mix = hy_df["rho_mix"]

py_oil = BlackOil(oil_api, pbub, gas_sg)
py_wat = FormWater(wat_sg=1)
py_gas = FormGas(gas_sg)
py_mix = ResMix(wc, fgor, py_oil, py_wat, py_gas)

plot_names = ["Oil", "Wat", "Gas"]

py_mfac = []
py_vfac = []
py_rho_mix = []
py_cmix = []

for prs in prs_ray:
    py_mix = py_mix.condition(prs, temp)
    py_mfac.append(py_mix.mass_fract())
    py_vfac.append(py_mix.volm_fract())
    py_rho_mix.append(py_mix.rho_mix())
    py_cmix.append(py_mix.cmix())

# change formating to make it more user friendly to print
py_mfac = list(zip(*py_mfac))
py_vfac = list(zip(*py_vfac))

fig, axs = plt.subplots(4, sharex=True)

for i, hy in enumerate(hy_mfac):
    axs[0].scatter(prs_ray, hy, label=f"Hysys {plot_names[i]}")
    axs[0].scatter(prs_ray, py_mfac[i], marker="*", label=f"Python {plot_names[i]}")
axs[0].set_ylabel("Mass Fraction")
axs[0].legend()

for i, hy in enumerate(hy_vfac):
    axs[1].scatter(prs_ray, hy, label=f"Hysys {plot_names[i]}")
    axs[1].scatter(prs_ray, py_vfac[i], marker="*", label=f"Python {plot_names[i]}")
axs[1].set_ylabel("Volume Fraction")
axs[1].legend()

axs[2].scatter(prs_ray, hy_rho_mix, label="hysys")
axs[2].scatter(prs_ray, py_rho_mix, label="python")
axs[2].set_ylabel("Mixture Density, lbm/ft3")
axs[2].legend()

axs[3].scatter(prs_ray, py_cmix, label="python")
axs[3].set_ylabel("Speed of Sound, ft/s")
axs[3].set_xlabel("Pressure, psig")
axs[3].legend()

fig.suptitle(f"{py_mix}")

plt.show()
