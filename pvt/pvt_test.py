import blackoil as bo
import formgas as fg
import numpy as np

res_temp = 80  # reservoir temperature
res_pres = 5000  # reservoir pressure

# make an array of pressures from 0 that goes 10% above the reservoir pressure
press_aray = np.arange(0, res_pres*1.1, 50)

# define Schrader Bluff Formation Gas
# define Schrader Bluff Oil
gas_sg = 0.8
oil_api = 22
bub_pnt = 1750  # bubble point pressure

mpu_gas = fg.FormGas.schrader_gas() # use a class method
mpu_oil = bo.BlackOil(oil_api, bub_pnt, gas_sg)

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