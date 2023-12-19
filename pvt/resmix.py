import math

from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater


class ResMix:
    def __init__(self, wc: float, fgor: int, oil: BlackOil, wat: FormWater, gas: FormGas) -> None:
        """Initialize a Reservoir Mixture

        Args:
                wc (float): Watercut of the Mixture, 0 to 1
                fgor (int): Formation GOR of the Mixture
                oil (BlackOil): Class with Oil_API, Gas_SG, Bubblepoint
                wat (FormWater): Class with Water SG
                gas (FormGas): Class with Gas SG

        Returns:
                Self
        """

        self.wc = wc  # specify a decimal
        self.fgor = fgor
        self.oil = oil
        self.wat = wat
        self.gas = gas

    def __repr__(self) -> str:
        # return(f'Mixture with {self.oil.oil_api} API Oil')
        return f"Mixture at {100*self.wc}% Watercut and {self.fgor} SCF/STB FGOR"

    def condition(self, press: float, temp: float):
        """Set condition of evaluation

        Args:
                press (float): Pressure of the mixture, psig
                temp (float): Temperature of the mixture, deg F

        Returns:
                Self
        """
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
        """Density Components

        Return the density of the oil, water and gas from the mixture.
        Requires a pressure and temperature condition to previously be set.

        Args:
                None

        Returns:
                poil (float): density of oil, lbm/ft3
                pwat (float): density of water, lbm/ft3
                pgas (float): density of gas, lbm/ft3
        """
        poil = self.oil.density
        pwat = self.wat.density
        pgas = self.gas.density
        return poil, pwat, pgas

    def visc_comp(self) -> tuple[float, float, float]:
        """Viscosity Components

        Return the viscosity of the oil, water and gas from the mixture.
        Requires a pressure and temperature condition to previously be set.

        Args:
                None

        Returns:
                uoil (float): viscosity of oil, cP
                uwat (float): viscosity of water, cP
                ugas (float): viscosity of gas, cP
        """
        uoil = self.oil.viscosity()
        uwat = self.wat.viscosity()
        ugas = self.gas.viscosity()
        return uoil, uwat, ugas

    def comp_comp(self) -> tuple[float, float, float]:
        """Compressibility Components

        Return the compressibility of the oil, water and gas from the mixture.
        Requires a pressure and temperature condition to previously be set.

        Args:
                None

        Returns:
                co (float): compressibility of oil, psi**-1
                cw (float): compressibility of oil, psi**-1
                cg (float): compressibility of oil, psi**-1
        """
        co = self.oil.compress()
        cw = self.wat.compress()
        cg = self.gas.compress()
        return co, cw, cg

    def mass_fract(self) -> tuple[float, float, float]:
        """Mass Fractions

        Return the mass fractions of the oil, water and gas from the mixture.
        Requires a pressure and temperature condition to previously be set.
        Uses algebraic relationships of wc, gor and density to solve for mass fractions.

        Args:
                None

        Returns:
                xoil (float): mass fraction of oil in the mixture
                xwat (float): mass fraction of water in the mixture
                xgas (float): mass fraction of gas in the mixture

        References:
                Derivations from Kaelin available on request
        """
        # pull out the eval. press and temp first
        # standard condition will overwrite them otherwise
        press = self.press
        temp = self.temp

        pstd = 0  # psig standard pressure
        tstd = 60  # deg f standard temperature

        # wc and fgor are evaluated at standard conditions
        poil, pwat, pgas = self.condition(pstd, tstd).dens_comp()

        # convert back to evaluated conditions
        self = self.condition(press, temp)

        wc = self.wc
        fgor = self.fgor
        rs = self.oil.gas_solubility()

        # mass based gas solubility
        mrs = (7.48 / 42) * rs * pgas / poil

        # convert from scf/bbl to scf/cf
        # densities are at standard conditions
        # mass formation gas oil ratio
        mfgor = (7.48 / 42) * fgor * pgas / poil

        # mass watercut
        mwc = pwat * wc / (pwat * wc + poil * (1 - wc))

        # mass formation gas to liquid ratio
        mfglr = (7.48 / 42) * fgor * pgas * (1 - wc) / (wc * pwat + (1 - wc) * poil)

        # mass fraction of gas
        xgas = mfglr / (1 + mfglr)

        # mass fraction of oil
        xoil = (1 - mwc) / (1 + mfgor * (1 - mwc))

        # mass fraction of water
        xwat = 1 - xoil - xgas

        # correct for the gas that is inside the oil
        xrs = xoil * mrs

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
        """Mixture Density

        Return the homogenous density of the mixture.
        Requires a pressure and temperature condition to previously be set.

        Args:
                None

        Returns:
                pmix (float): density of mixture, lbm/ft3
        """
        press = self.press
        temp = self.temp

        xoil, xwat, xgas = self.condition(press, temp).mass_fract()
        poil, pwat, pgas = self.condition(press, temp).dens_comp()

        # mixture specific volume
        vmix = (xoil / poil) + (xwat / pwat) + (xgas / pgas)

        pmix = 1 / vmix
        pmix = round(pmix, 4)

        return pmix

    def volm_fract(self) -> tuple[float, float, float]:
        """Volume Fractions

        Return the volume fractions of the oil, water and gas from the mixture.
        Requires a pressure and temperature condition to previously be set.

        Args:
                None

        Returns:
                yoil (float): volume fraction of oil in the mixture
                ywat (float): volume fraction of water in the mixture
                ygas (float): volume fraction of gas in the mixture
        """
        press = self.press
        temp = self.temp

        # these condition calls are probably redundant
        xoil, xwat, xgas = self.condition(press, temp).mass_fract()
        poil, pwat, pgas = self.condition(press, temp).dens_comp()

        # mixture specific volume
        vmix = (xoil / poil) + (xwat / pwat) + (xgas / pgas)

        pmix = 1 / vmix

        yoil = xoil * pmix / poil
        ywat = xwat * pmix / pwat
        ygas = xgas * pmix / pgas

        # round the mass fractions
        deci = 4
        yoil = round(yoil, deci)
        ywat = round(ywat, deci)
        ygas = round(ygas, deci)

        return yoil, ywat, ygas

    def cmix(self) -> float:
        """Mixture Speed of Sound

        Return the adiabatic? Speed of Sound in the mixture.
        Requires a pressure and temperature condition to previously be set.

        Args:
                None

        Returns:
                cmix (float): speed of sound in the mixture, ft/s

        References:
                Sound Speed in the Mixture Water-Air D.Himr (2009)
        """
        # paper calculated 390 ft/s there conditions
        # we calculated 460 ft/s, with our conditions, seems ballpark
        co, cw, cg = self.comp_comp()  # isothermal compressibility
        yoil, ywat, ygas = self.volm_fract()  # volume fractions
        ps = self.pmix()

        cs = co * yoil + cw * ywat + cg * ygas  # mixture compressibility
        ks = 1 / cs  # mixture bulk modulus of elasticity

        cmix = math.sqrt(32.174 * 144 * ks / ps)  # speed of sound, ft/s
        cmix = round(cmix, 2)
        return cmix
