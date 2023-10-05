import math


class BlackOil:

    def __init__(
            self, oil_api: float, bubblepoint: float, gas_sg: float) -> None:
        '''Initialize a Black Oil Stream

        Args:
                oil_api (float): Oil API, 10 to 40
                bubblepoint (float): Bubble Point Pressure, PSIG
                gas_sg (float): In-Situ Gas Specific Gravity

        Returns:
                Self
        '''

        if (10 < oil_api < 40) == False:
            raise ValueError(f'Oil API {oil_api} Outside Range')

        if (1000 < bubblepoint < 3000) == False:
            raise ValueError(f'Bubblepoint {bubblepoint} Outside Range')

        if (0.5 < gas_sg < 1.2) == False:
            raise ValueError(f'Gas SG {gas_sg} Outside Range')

        self.oil_api = oil_api
        self.pbp = bubblepoint
        self.gas_sg = gas_sg

    def __repr__(self):
        return f'Oil: {self.oil_api} API and a {round(self.gas_sg,2)} SG Gas'

    @classmethod
    def schrader_oil(cls):
        return cls(oil_api=22, bubblepoint=1750, gas_sg=0.8)

    def condition(self, press: float, temp: float):
        '''Set condition of evaluation

        Args:
                press (float): Pressure of the oil, psig
                temp (float): Temperature of the oil, deg F

        Returns:
                Self
        '''
        # define the condition, where are you at?
        # what is the pressure and what is the temperature?
        self.press = press
        self.temp = temp

        # input press in psig
        # input temp in deg F
        self._pressa = self.press + 14.7  # convert psig to psia
        self._tempr = self.temp + 459.67  # convert fahr to rankine
        return self

    def gas_solubility(self) -> float:
        """Gas Solubility in Oil (Solution GOR)

        Return the gas solubility in the oil, scf/stb.
        Follows the Vasquez and Beggs Methodology

        Args:
            None

        Returns:
            gassol (float): gas solubility scf/stb

        References:
            Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 22
            Correlations for Fluid Physical Property... Vasquez and Beggs (1980)
        """

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

    def compress(self) -> float:
        """Oil Compressibility Isothermal

        Calculate isothermal oil compressibility.
        Inverse of the bulk modulus of elasticity.
        Correlations are named in references.

        Args:
            None

        Returns:
            co (float): oil compressibility, psi**-1

        References:
            Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 22
            Correlations for Fluid Physical Property... Vasquez and Beggs (1980)
            Coefficent of Isothermal Compressibility... McCain et al. (1988)
        """

        pabs = self._pressa  # convert pressure from psig to psia
        rs = self.gas_solubility()  # solubility of gas in the oil

        oil_api = self.oil_api
        temp = self.temp  # deg F
        gas_sg = self.gas_sg

        # vazquez and beggs correlation
        if self._pressa > self.pbp:  # above bubblepoint
            co = (5*rs+17.2*temp-1180*gas_sg+12.61*oil_api-1433)/(pabs*10**5)

        # McCain et al.
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

    def oil_fvf(self) -> float:
        """Oil Formation Volume Factor, Bo

        Calculate formation volume factor for the oil phase.
        Correlations are named in references.

        Args:
            None

        Returns:
            bo (float): oil formation volume, rb/stb

        References:
            Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 21
            Correlations for Fluid Physical Property... Vasquez and Beggs (1980)
        """

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
            co = self.compress()  # isothermal compressibility
            bo = bo*math.exp(co*(pbp-pabs))

        bo = round(bo, 2)
        self.bo = bo
        return bo

    # property decorator makes it so you don't need brackets
    # if you are just calling it with self in the arguement
    @property
    def density(self) -> float:
        """Live oil density, lbm/ft3

        Return the density of the oil.
        Requires a pressure and temperature condition to previously be set.

        Args:
            None

        Returns:
            doil (float): density of the oil, lbm/ft3

        References:
            Multiphase class excel spreadsheet
        """

        oil_api = self.oil_api
        oil_sg = 141.5/(oil_api+131.5)  # oil specific gravity
        bo = self.oil_fvf()

        gas_sg = self.gas_sg
        rs = self.gas_solubility()

        doil = (62.42796*oil_sg+(0.0136*gas_sg*rs))/bo
        doil = round(doil, 2)
        self.doil = doil
        return doil

    def viscosity(self) -> float:
        """Live oil viscosity, cP

        Return the live oil viscosity, cP.
        Requires pressure and temperature condition to previously be set.

        Args:
            None

        Returns:
            uoil (float): live oil viscosity, cP

        References:
            Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 23
            Correlations for Fluid Physical Property... Vasquez and Beggs (1980)
        """

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
