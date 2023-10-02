from blackoil import BlackOil
from formgas import FormGas
from formwat import FormWater


class ResMix:

    def __init__(
            self,
            wc: float,
            fgor: int,
            oil: BlackOil,
            wat: FormWater,
            gas: FormGas
    ) -> None:
        """
        Name:    Define Reservoir Mixture
        Inputs:  BlackOil - Class with Oil_API, Gas_SG, and Pbubble
                 FormWater - Class with Water SG
                 FormGas - Class with Gas SG
                 wc - Watercut of the Mixture, between 0 and 1
                 fgor - Formation GOR of the Mixture
        Output:  None
        Rev:     09/22/23 - K. Ellis wrote into Python
        """

        self.wc = wc  # specify a decimal
        self.fgor = fgor
        self.oil = oil
        self.wat = wat
        self.gas = gas

    def __repr__(self) -> str:

        # return(f'Mixture with {self.oil.oil_api} API Oil')
        return f'Mixture at {100*self.wc}% Watercut and {self.fgor} SCF/STB FGOR'

    def condition(self, press: float, temp: float):
        # define the condition, where are you at?
        # what is the pressure and what is the temperature?
        self.press = press
        self.temp = temp

        self.oil = self.oil.condition(press, temp)
        self.wat = self.wat.condition(press, temp)
        self.gas = self.gas.condition(press, temp)

        # input press in psig
        # input temp in deg F
        self._pressa = self.press + 14.7  # convert psig to psia
        self._tempr = self.temp + 459.67  # convert fahr to rankine
        return self

    def dens_comp(self) -> tuple[float, float, float]:
        """
        Name:   Density Components
        Input:  Blah
        Output: poil - density of oil, lbm/ft3
                pwat - density of water, lbm/ft3
                pgas - density of gas, lbm/ft3
        About:  Outputs the 3 individual densities of the stream
        """
        poil = self.oil.density
        pwat = self.wat.density
        pgas = self.gas.density
        return poil, pwat, pgas

    def visc_comp(self) -> tuple[float, float, float]:
        """
        Name:   Viscosity Components
        Input:  Blah
        Output: uoil - viscosity of oil, cp
                uwat - viscosity of water, cp
                ugas - viscosity of gas, cp
        About:  Outputs the 3 individual viscosities of the stream
        """
        uoil = self.oil.viscosity()
        uwat = self.wat.viscosity()
        ugas = self.gas.viscosity()
        return uoil, uwat, ugas

    def mass_fract(self) -> tuple[float, float, float]:
        """ 
        Name:   Mass Fractions
        Input:  watercut - watercut at standard conditions
                fgor - formation gas oil ratio, scf/stb
                rs - gas solubility,  scf/stb                
                pwat - density of water, lbm/ft3 (standard conditions)
                poil - density of oil, lbm/ft3 (standard conditions)
                pgas - density of gas, lbm/ft3 (standard conditions)
        Output: xoil - mass fraction oil, mass oil / mass total
                xwat - mass fraction water, mass water / mass total
                xgas - mass fraction gas, mass gas / mass total
        About:  Uses standard condition densities and wc / fgor to calculate
                the mass fractions of the stream
        """

        # pull out the eval. press and temp first
        # standard condition will overwrite them otherwise
        press = self.press
        temp = self.temp

        pstd = 0  # psig standard pressure
        tstd = 60  # deg f standard temperature

        # wc and fgor are evaluated at standard conditions
        # might be good to rethink the condition stuff?
        poil, pwat, pgas = self.condition(pstd, tstd).dens_comp()

        # convert back to evaluated conditions
        self = self.condition(press, temp)

        wc = self.wc
        fgor = self.fgor
        rs = self.oil.gas_solubility()

        # mass based gas solubility
        mrs = (7.48/42)*rs*pgas/poil

        # convert from scf/bbl to scf/cf (make sure gas density is at standard conditions)
        # mass formation gas oil ratio
        mfgor = (7.48/42)*fgor*pgas/poil

        # mass watercut
        mwc = pwat*wc/(pwat*wc+poil*(1-wc))

        # mass formation gas to liquid ratio
        mfglr = (7.48/42)*fgor*pgas*(1-wc)/(wc*pwat+(1-wc)*poil)

        # mass fraction of gas
        xgas = mfglr/(1+mfglr)

        # mass fraction of oil
        xoil = (1-mwc)/(1+mfgor*(1-mwc))

        # mass fraction of water
        # won't change even with gas and oil trading some mass back and forth
        # slight differences in excel spreadsheet and python solution...? py: 0.9595, excel 0.972
        xwat = 1 - xoil - xgas

        # correct for the gas that is inside the oil
        xrs = xoil*mrs

        # calculate new mass fraction of free gas
        xgas = xgas - xrs
        # make sure mass fraction of gas is always zero or above
        xgas = max(xgas, 0)

        xoil = 1 - xwat - xgas

        # round the mass fractions
        deci = 4
        xgas = round(xgas, deci)
        xoil = round(xoil, deci)
        xwat = round(xwat, deci)

        # mass fractions
        return xoil, xwat, xgas

    def pmix(self) -> float:
        """ 
        Name:   Mixture Density
        Input:  xoil - mass fraction oil, mass oil / mass total
                xwat - mass fraction water, mass water / mass total
                xgas - mass fraction gas, mass gas / mass total                
                pwat - density of water, lbm/ft3 (actual cond.)
                poil - density of oil, lbm/ft3 (actual cond.)
                pgas - density of gas, lbm/ft3 (actual cond.)
        Output: pmix - density of mixture, lbm/ft3 (actual cond.)
        """

        press = self.press
        temp = self.temp

        xoil, xwat, xgas = self.condition(press, temp).mass_fract()
        poil, pwat, pgas = self.condition(press, temp).dens_comp()

        # mixture specific volume
        vmix = (xoil/poil) + (xwat/pwat) + (xgas/pgas)

        pmix = 1/vmix
        pmix = round(pmix, 4)

        return pmix

    def volm_fractions(self) -> tuple[float, float, float]:
        """ 
        Name:   Volume Fractions
        Input:  xwat - mass fraction water, mass water / mass total
                xoil - mass fraction oil, mass oil / mass total
                xgas - mass fraction gas, mass gas / mass total                
                pwat - density of water, lbm/ft3 (actual cond.)
                poil - density of oil, lbm/ft3 (actual cond.)
                pgas - density of gas, lbm/ft3 (actual cond.)

        Output: ywat - volume fraction water, volume water / volume total
                yoil - volume fraction oil, volume oil / volume total
                ygas - volume fraction gas, volume gas / volume total
        """

        press = self.press
        temp = self.temp

        # these condition calls are probably redundant
        xoil, xwat, xgas = self.condition(press, temp).mass_fract()
        poil, pwat, pgas = self.condition(press, temp).dens_comp()

        # mixture specific volume
        vmix = (xoil/poil) + (xwat/pwat) + (xgas/pgas)

        pmix = 1/vmix

        yoil = xoil*pmix/poil
        ywat = xwat*pmix/pwat
        ygas = xgas*pmix/pgas

        # round the mass fractions
        deci = 4
        yoil = round(yoil, deci)
        ywat = round(ywat, deci)
        ygas = round(ygas, deci)

        return yoil, ywat, ygas


def prop_table(self, press_array, temp):
    """
    Name:   Property Table
    Input:  press_array - Numpy Array, Lowest pressure to highest, psig
            temp - evaluated temperature deg F
    Output: Property Table
            Oil, Water, Gas Density
            Oil, Water, Gas Viscosity
            3 Phase Mixture Density
            Oil, Water, Gas Mass Fraction
            Oil, Water, Gas Volm Fraction
    Rev:    09/22/23 - K.Ellis wrote into Python
    """
