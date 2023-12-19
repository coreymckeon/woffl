# import sys

import numpy as np

from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater
from pvt.resmix import ResMix

# print(sys.path)
# sys.path.append("c:\\Users\\ka9612\\OneDrive - Hilcorp\\vs_code\\hilcorpak")

mpu_oil = BlackOil.schrader_oil()  # class method
mpu_wat = FormWater.schrader_wat()  # class method
mpu_gas = FormGas.schrader_gas()  # class method

e42 = ResMix(0.1, 300, mpu_oil, mpu_wat, mpu_gas)

temp = 80
press = 700

e42.condition(press, temp)

oil_comp = mpu_oil.condition(700, 80).compress()
wat_comp = mpu_wat.compress()
gas_comp = mpu_gas.condition(700, 80).compress()

print("Compressibility")
print(f"Oil: {oil_comp}, Water: {wat_comp}, Gas: {gas_comp} psi-1")

print("Bulk Modulus")
print(f"Oil: {1/oil_comp}, Water: {1/wat_comp}, Gas: {1/gas_comp} psi")

print("Speed of Sound in Mixture")
print(f"cmix: {e42.cmix()} ft/s")

print("Oil, Water, Gas Volm Fractions")
print(e42.volm_fract())
