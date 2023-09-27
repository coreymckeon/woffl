import math


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

        gassol = (pabs**C2)*C1*self.gas_sg * \
            math.exp(C3*self.oil_api/self._tempr)
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
                -1.450*math.log(pabs),
                -0.383*math.log(self.pbp),
                1.402*math.log(temp),
                0.256*math.log(oil_api),
                0.449*math.log(rs)
            ]

            co = math.exp(sum(co_list))

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
            bo = bo*math.exp(co*(pbp-pabs))

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
        x = (temp**-1.163)*math.exp(6.9824-0.04658*oil_api)
        uod = (10**x)-1  # beggs and robinson correlation for dead oil viscosity

        a = 10.715*(rs+100)**-0.515  # same method as multiphase excel sheet
        b = 5.44*(rs+150)**-0.338
        uoil = a*(uod**b)

        if self._pressa > self.pbp:  # above bubblepoint
            pbp = self.pbp
            pabs = self._pressa  # absolute pressure
            # increase in viscosity due to compression
            m = 2.6*(pabs**1.187)*math.exp(-11.513-pabs*8.98*10**-5)
            uoil = uoil*(pabs/pbp)**m

        uoil = round(uoil, 2)
        self.uoil = uoil
        return uoil
