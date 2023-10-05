import numpy as np
from blackoil import BlackOil
from formgas import FormGas
from formwat import FormWater
from resmix import ResMix

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

print('Compressibility')
print(f'Oil: {oil_comp}, Water: {wat_comp}, Gas: {gas_comp} psi-1')

print('Bulk Modulus')
print(f'Oil: {1/oil_comp}, Water: {1/wat_comp}, Gas: {1/gas_comp} psi')

print('Speed of Sound in Mixture')
print(f'cmix: {e42.cmix()} ft/s')

print('Oil, Water, Gas Volm Fractions')
print(e42.volm_fract())
"""

print('Oil, Water, Gas Density lbm/ft3')
print(e42.dens_comp())
print('Mixture Density lbm/ft3')
print(e42.pmix())
print('Oil, Water, Gas Viscosity cP')
print(e42.visc_comp())
print('Oil, Water, Gas Mass Fractions')
print(e42.mass_fract())
print('Oil, Water, Gas Volm Fractions')
print(e42.volm_fract())


res_temp = 80  # reservoir temperature
res_pres = 5000  # reservoir pressure

# make an array of pressures from 0 that goes 10% above the reservoir pressure
press_aray = np.arange(0, res_pres*1.1, 50)

pvt_vew = {
    "pressure": 0,
    "dens_oil": 1,
    "dens_gas": 2,
    "visc_oil": 3,
    "visc_gas": 4,
    "gas_solb": 5,
}

pvt_data = np.empty(shape=(len(press_aray), len(pvt_vew)))

for i, pres in enumerate(press_aray):
    mpu_oil.condition(pres, res_temp)
    mpu_gas.condition(pres, res_temp)

    pvt_data[i, pvt_vew['pressure']] = pres
    pvt_data[i, pvt_vew['dens_oil']] = mpu_oil.density()
    pvt_data[i, pvt_vew['dens_gas']] = mpu_gas.density()
    # the oil viscosity just seems high...
    pvt_data[i, pvt_vew['visc_oil']] = mpu_oil.viscosity()
    pvt_data[i, pvt_vew['visc_gas']] = mpu_gas.viscosity()
    pvt_data[i, pvt_vew['gas_solb']] = mpu_oil.gas_solubility()

print(pvt_data)
# print(mpu_oil.condition(500, 80))
"""
