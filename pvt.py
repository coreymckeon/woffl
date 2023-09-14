import math
from pprint import pprint

import numpy as np


class FormGas():

    # physical properties of hydrocarbon and selected compounds, field units (Whitson and Brule 2000)
    # Table B-1 from the text "Applied Multiphase Flow in Pipes and Flow Assurance"

    fldpco = {'n2':  [28.02, 0.4700, 29.31,  493.0, 227.3, 1.443, 0.2916, 0.0450,  13.3, 0, 0],
              'co2': [44.01, 0.5000, 31.18, 1070.6, 547.6, 1.505, 0.2742, 0.2310, 350.4, 0, 0],
              'h2s': [34.08, 0.5000, 31.18, 1306.0, 672.4, 1.564, 0.2831, 0.1000, 383.1, 0, 672],
              'c1':  [16.04, 0.3300, 20.58,  667.8, 343.0, 1.590, 0.2864, 0.0115, 201.0, 0, 1012],
              'c2':  [30.07, 0.4500, 28.06,  707.8, 548.8, 2.370, 0.2843, 0.0908, 332.2, 0, 1783],
              'c3':  [44.09, 0.5077, 31.66,  616.3, 665.7, 3.250, 0.2804, 0.1454, 416.0, 27.4, 2557],
              'ic4': [58.12, 0.5613, 35.01,  529.1, 734.7, 4.208, 0.2824, 0.1756, 470.8, 32.7, 3354],
              'nc4': [58.12, 0.5844, 36.45,  550.7, 785.3, 4.080, 0.2736, 0.1928, 490.8, 31.4, 3369],
              'ic5': [72.15, 0.6274, 39.13,  490.4, 828.8, 4.899, 0.2701, 0.2273, 541.8, 36.3, 4001],
              'nc5': [72.15, 0.6301, 39.30,  488.6, 845.4, 4.870, 0.2623, 0.2510, 556.6, 36.2, 4009],
              'nc6': [86.17, 0.6604, 41.19,  436.9, 913.4, 5.929, 0.2643, 0.2957, 615.4, 41.2, 4756],
              'nc7': [100.2, 0.6828, 42.58,  396.8, 972.5, 6.924, 0.2633, 0.3506, 668.8, 46.3, 5503],
              'nc8': [114.2, 0.7806, 44.19,  360.6, 1023.9, 7.882, 0.2587, 0.3976, 717.9, 50.9, 6250],
              'nc9': [128.3, 0.7271, 45.35,  332.0, 1070.3, 8.773, 0.2536, 0.4437, 763.1, 55.7, 6996],
              'nc10': [142.3, 0.7324, 45.68,  304.0, 1111.8, 9.661, 0.2462, 0.4902, 805.2, 81.4, 7743],
              'air': [28.97, 0.4700, 29.31,  547.0, 239.0, 1.364, 0.2910, 0.0400, 141.9, 0, 0],
              'h2o': [18.02, 1.0000, 62.37, 3206.0, 1165.0, 0.916, 0.2350, 0.3440, 671.6, 0, 0],
              'o2': [32.00, 0.5000, 31.18,  732.0, 278.0, 1.174, 0.2880, 0.0250, 182.2, 0, 0]}

    # will leave the dictionary lookups alone for now, come back later for molecular composition

    # ideal gas constant
    _R = 10.73  # psia*ft^3/(lbmol*Rankine)

    def __init__(self, gas_sg):

        if (0.5 < gas_sg < 1.2) == False:
            print(f'Gas SG {gas_sg} Outside Range')  # do I need more here?

        # define a lot of properties just on the gas' specific gravity
        self.gas_sg = gas_sg

        # the following is the code for Pseudo Critical Pressure from correlations
        # can be adjusted later for H2S or CO2 if necessary
        self._pcrit = 756.8 - 131.07*self.gas_sg - 3.6*self.gas_sg**2  # psia

        # the following is the code for Pseudo Critical Temp from correlations
        # can be adjusted later for H2S or CO2 if necessary
        self._tcrit = 169.2 + 349.5*self.gas_sg - 74.0*self.gas_sg**2  # rankine

        # calculate molecular weight
        # reservoir engineer book, red
        self.mw = round(28.96443*self.gas_sg, 4)

    # does something when you print your class
    def __repr__(self):
        return f'Gas: {self.gas_sg} SG and {self.mw} Mol Weight'

    # almost need seperate function to change pressure / temperature
    def condition(self, press, temp):
        # define the condition, where are you at?
        # what is the pressure and what is the temperature?
        self.press = press
        self.temp = temp

        # input press in psig
        # input temp in deg F
        self._pressa = self.press + 14.7  # convert psig to psia
        self._tempr = self.temp + 459.67  # convert fahr to rankine

        # calculate pseudo reduced temperature and pressure
        # the following have not been adjusted for non-hydrocarbon gas such as H2S or CO2
        # code can be modified later if this is needed
        self._ppr = self._pressa/self._pcrit  # unitless, pressure pseudo reduced
        self._tpr = self._tempr/self._tcrit  # unitless, temperature pseudo reduced

    # should I just make something called properties
    # then you pull out what you want?
    # would be nice to add optional arguements here
    def zfactor(self):
        """ Name:   Natural Gas Compressibility Factor (Z-Factor)
            Inputs: press - Pressure (psig)
                    temp - Temperature (deg F)
                    gasSG - Gas Specific Gravity (Float)
                    molcomp - Molar Fractions of Composition (Dictionary)
                Note:   Function will accept gasSG OR molcomp, not BOTH
                        Do not use gasSG method for sour gases with significant H2S or CO2
            Output: Z-factor - Real to Ideal Compressibility Ratio
            Ref:    Fundamental Principles of Reservoir Engineering, Brian F. Towler (2002) 15
                    Applied Multiphase Flow in Pipes and Flow Assurance, Al-Safran and Brill (2017) 300-302
                    Numerical Methods For Engineers and Scientists, Amos Gilat and Vish (2013) 71-74 
            Rev:    01/13/19 - K.Ellis wrote into python
                    02/04/19 - K.Ellis modified to accept gasSG an an input
                    09/01/23 - K.Ellis wrote into OOP"""
        # need error handling if a pressure / temperature aren't defined arlready
        # can I define a pressure / temp here and it will calculate?
        # but if I already have it, basically making it optional?
        try:
            # calculate m and n, method given to me in multiphase clase
            # not sure where the equations come from exactly
            self._m = 0.51*(self._tpr)**(-4.133)
            self._n = 0.038 - 0.026*(self._tpr)**(1/2)

        except AttributeError:
            print('Need to define Pressure and Temperature')

        # calculate z-factor
        # does everything have to use self?
        zfactor = 1 - self._m*self._ppr + self._n*self._ppr**2 + 0.0003*self._ppr**3
        zfactor = round(zfactor, 4)
        self._zfactor = zfactor
        return zfactor

    def density(self):
        """ Name:   Gas Density
            Inputs: press - Pressure (psig)
                    temp - Temperature (deg F)
                    gasSG - Gas Specific Gravity (Float)
                    molcomp - Molar Fractions of Composition (Dictionary)
                Note:   Function will accept gasSG OR molcomp, not BOTH
                        Do not use gasSG method for sour gases with significant H2S or CO2
            Output: Dgas - Gas Density (lbm/ft^3)
            Ref:    Fundamental Principles of Reservoir Engineering, Brian F. Towler (2002) 15
                    Applied Multiphase Flow in Pipes and Flow Assurance, Al-Safran and Brill (2017) 302-303
            Rev:    01/13/19 - K.Ellis wrote into python
                    02/04/19 - K.Ellis modified to accept gasSG an an input
                    09/01/23 - K.Ellis wrote into OOP"""
        # calculate gas density
        # should I calculate Bg in RCF/SCF as well???
        # density units, lbm/ft^3

        zval = self.zfactor()  # call method if it hasn't been already?

        dgas = self._pressa*self.mw/(zval*FormGas._R*self._tempr)
        dgas = round(dgas, 4)
        self.dgas = dgas
        return dgas

    def viscosity(self):
        """ Name:   Gas Viscosity (Lee et al. 1966) Semi-Empirical
            Inputs: press - Pressure (psig)
                    Temp - Temperature (deg F)
                    gasSG - Gas Specific Gravity (Float)
                    molcomp - Molar Fractions of Composition (Dictionary)
                Note:   Function will accept gasSG OR molcomp, not BOTH
                        Do not use gasSG method for sour gases with significant H2S or CO2
            Output: Ug - Gas Viscosity (centipoise)
            Ref:    Fundamental Principles of Reservoir Engineering, Brian F. Towler (2002) 15
                    Applied Multiphase Flow in Pipes and Flow Assurance, Al-Safran and Brill (2017) 304
                    Lee et al. (1966)
            Rev:    01/13/19 - K.Ellis wrote into python
                    02/04/19 - K.Ellis modified to accept gasSG an an input
                    09/01/23 - K.Ellis wrote into OOP"""
        # calculate gas viscosity
        # viscosity units, cP
        # should I change this for later consistency

        dens = self.density()  # call method if you haven't done it already?
        mw = self.mw
        tempr = self._tempr

        K = ((9.4+0.02*mw)*tempr**1.5)/(209+19*mw+tempr)
        X = 3.5+(986/tempr)+0.01*mw
        Y = 2.4-0.2*X

        # [eqn B-72] viscosity of the gas
        visc = (10**-4)*K*np.exp(X*(dens/62.4)**Y)
        visc = round(visc, 5)
        self.ugas = visc
        return visc

    # def gas_fvf(self):
        #


class DeadOil():

    def __init__(self, oil_api, bubblepoint):
        # define an oil stream that has no live gas in it
        # what it wad before it died

        self.oil_api = oil_api
        self.pbp = bubblepoint

    # does something when you print your class
    def __repr__(self):
        return f'Dead Oil: {self.oil_api} API and {self.pbp} PSIG BubblePoint'

# note, the problem with inheritance is that it inherits your method names
# so if BlackOil inherits properties from FormGas, when I call density, it
# will call the FormGas Density Properties instead of...what I want?


class BlackOil():

    def __init__(self, oil_api, bubblepoint, gas_sg):
        """ Name:   Define Oil Stream         
            Inputs: oilAPI - oil API gravity of the dead oil
                    Pbp - bubble point pressure (psia)
                    gasSG - gas specific gravity, relative to air
            Output: None
            Rev:    01/16/19 - K. Ellis wrote into Python
                    09/01/23 - K. Ellis made into OOP"""

        if (10 < oil_api < 40) == False:
            print(f'Oil API {oil_api} Outside Range')  # do I need more here?

        if (1000 < bubblepoint < 3000) == False:
            # do I need more here?
            print(f'Bubblepoint {bubblepoint} Outside Range')

        if (0.5 < gas_sg < 1.2) == False:
            print(f'Gas SG {gas_sg} Outside Range')  # do I need more here?

        # pass the defined gas sg into FormGas Method
        # the BlackOil inherited many of the FormGas Methods?
        # I kind of liked having to call the other method
        # since now how to I define between the gas mw and oil mw??

        self.oil_api = oil_api
        self.pbp = bubblepoint
        self.gas_sg = gas_sg

    # does something when you print your class
    def __repr__(self):
        return f'Oil: {self.oil_api} API and a {round(self.gas_sg,2)} SG Gas'

    def condition(self, press, temp):
        # define the condition, where are you at?
        # what is the pressure and what is the temperature?
        self.press = press
        self.temp = temp

        # input press in psig
        # input temp in deg F
        self._pressa = self.press + 14.7  # convert psig to psia
        self._tempr = self.temp + 459.67  # convert fahr to rankine

    def gas_solubility(self):
        """ Name:   Gas Solubility in Oil (Solution Gas-Oil Ratio)         
            Inputs: press - system pressure (psig)
                    temp - system temperature (deg F)
                    oilAPI - oil API gravity of the dead oil
                    gasSG - gas specific gravity, relative to air
                    Pbp - bubble point pressure (psia)
            Output: Rs - Gas Solubility in Oil (SCF/STB)
            Ref:    Vasquez and Beggs (1980)
                    Fundamental Principles of Reservoir Engineering (2002) 20-21
            Rev:    01/16/19 - K. Ellis wrote into Python
                    09/01/23 - K. Ellis wrote into OOP"""

        if self.oil_api <= 30:
            C1, C2, C3 = [0.0362, 1.0937, 25.7240]
        else:  # if oilAPI is greater than 30API
            C1, C2, C3 = [0.0178, 1.187, 23.9310]

        # if you are above your bubblepoint pressure
        if self._pressa > self.pbp:
            pabs = self.pbp + 14.7  # calculate gas solubility using bubblepoint pressure
        else:
            pabs = self._pressa

        gassol = (pabs**C2)*C1*self.gas_sg*np.exp(C3*self.oil_api/self._tempr)
        gassol = round(gassol, 2)
        self.gassol = gassol
        return gassol

    def oil_comp(self):
        """ Name:   Oil Compressibility
            Inputs: press - system pressure (psig)
                    temp - system temperature (deg F)
                    oilAPI - oil API gravity of the dead oil
                    gasSG - gas specific gravity, relative to air
                    Pbp - bubble point pressure (psia)
            Output: Co - Oil Compressibility (psi**-1)
            Ref:    McCain et al. (1988)
                    Vasquez and Beggs (1980)
                    Fundamental Principles of Reservoir Engineering (2002) 22
            Rev:    01/27/19 - K. Ellis wrote into Python
                    09/01/23 - K. Ellis wrote into OOP"""

        pabs = self._pressa  # convert pressure from psig to psia
        rs = self.gassol  # solubility of gas in the oil

        oil_api = self.oil_api
        temp = self.temp  # deg F
        gas_sg = self.gas_solubility()

        if self._pressa > self.pbp:  # above bubblepoint
            co = (5*rs+17.2*temp-1180*gas_sg+12.61*oil_api-1433)/(pabs*10**5)

        else:  # below the bubblepoint
            co_list = [
                -7.573,
                -1.450*np.log(pabs),
                -0.383*np.log(self.pbp),
                1.402*np.log(temp),
                0.256*np.log(oil_api),
                0.449*np.log(rs)
            ]

            co = np.exp(sum(co_list))

        co = round(co, 7)
        self.co = co
        return co

    def oil_fvf(self):
        """ Name:   Oil FVF Vasquez and Beggs
            Inputs: press - system pressure (psig)
                    temp - system temperature (deg F)
                    oilAPI - oil API gravity of the dead oil
                    gasSG = gas specific gravity, relative to air
                    Pbp - pressure bubble point (psia)
            Output: Bo - Oil Formation Volume Factor (RB/STB)
            Ref:    Vasquez and Beggs (1980)
                    Fundamental Principles of Reservoir Engineering (2002) 20-21
            Rev:    01/27/19 - K. Ellis wrote into Python
                    09/01/23 - K. Ellis wrote into OOP"""

        if self.oil_api <= 30:
            C4, C5, C6 = [4.677*10**-4, 1.751*10**-5, -1.811*10**-8]
        else:  # if oilAPI is greater than 30API
            C4, C5, C6 = [4.670*10**-4, 1.100*10**-5, 1.337*10**-9]

        oil_api = self.oil_api
        temp = self.temp  # deg F, not deg R
        gas_sg = self.gas_sg

        rs = self.gas_solubility()  # solubility of gas in the oil
        bo = 1 + C4*rs + C5*(temp-60)*(oil_api/gas_sg) + \
            C6*rs*(temp-60)*(oil_api/gas_sg)

        if self._pressa > self.pbp:  # above bubblepoint
            # need to factor in the isothermal compressibility of the oil
            # above bubblepoint, oil starts shrinking with pressure
            pbp = self.pbp
            pabs = self._pressa
            # call function if it hasn't been already
            co = self.oil_comp()  # isothermal compressibility
            bo = bo*np.exp(co*(pbp-pabs))

        bo = round(bo, 2)
        self.bo = bo
        return bo

    def density(self):
        """ Name:   Oil Density with Entrained Gas
            Inputs: press - system pressure (psig)
                    temp - system temperature (deg F)
                    oilAPI - oil API gravity of the dead oil
                    gasSG - gas specific gravity, relative to air
                    Pbp - pressure bubble point (psia)
            Output: Doil - density of oil with entrained gas (lbm/ft^3)
            Ref:    Multiphase Class Excel Spreadsheet
            Rev:    01/16/19 - K. Ellis wrote into Python
                    09/01/23 - K. Ellis wrote into OOP"""

        oil_api = self.oil_api
        oil_sg = 141.5/(oil_api+131.5)  # oil specific gravity
        bo = self.oil_fvf()

        gas_sg = self.gas_sg
        rs = self.gas_solubility()

        doil = (62.42796*oil_sg+(0.0136*gas_sg*rs))/bo
        doil = round(doil, 2)
        self.doil = doil
        return doil

    def viscosity(self):
        """ Name:   Oil Viscosity with Entrained Gas    
            Inputs: press - system pressure (psig)
                    temp - system temperature (deg F)
                    oilAPI - oil API gravity of the dead oil
                    gasSG - gas specific gravity, relative to air
                    Pbp - bubble point pressure (psia)
            Output: Uo - Viscosity of live oil (cP)
            Ref:    Vasquez and Beggs (1980)
                    Beggs and Robinson (1975)
                    Chew and Connally (1959)
                    Fundamental Principles of Reservoir Engineering (2002) 23
            Rev:    01/28/19 - K. Ellis wrote into Python
                    09/01/23 - K. Ellis wrote into OOP"""

        pabs = self._pressa  # absolute pressure
        temp = self.temp  # deg F
        oil_api = self.oil_api

        rs = self.gas_solubility()

        # find dead-oil viscosity first
        x = (temp**-1.163)*np.exp(6.9824-0.04658*oil_api)
        uod = (10**x)-1  # beggs and robinson correlation for dead oil viscosity

        a = 10.715*(rs+100)**-0.515  # same method as multiphase excel sheet
        b = 5.44*(rs+150)**-0.338
        uoil = a*(uod**b)

        if self._pressa > self.pbp:  # above bubblepoint
            pbp = self.pbp
            pabs = self._pressa  # absolute pressure
            # increase in viscosity due to compression
            m = 2.6*(pabs**1.187)*np.exp(-11.513-pabs*8.98*10**-5)
            uoil = uoil*(pabs/pbp)**m

        uoil = round(uoil, 2)
        self.uoil = uoil
        return uoil


res_temp = 80  # reservoir temperature
res_pres = 5000  # reservoir pressure

# make an array of pressures from 0 that goes 10% above the reservoir pressure
press_aray = np.arange(0, res_pres*1.1, 50)

# define Schrader Bluff Formation Gas
# define Schrader Bluff Oil
gas_sg = 0.8
oil_api = 22
bub_pnt = 1700  # bubble point pressure

mpu_gas = FormGas(gas_sg)
mpu_oil = BlackOil(oil_api, bub_pnt, gas_sg)

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
