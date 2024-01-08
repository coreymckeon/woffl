import math

from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater

# add property / method that calculates / stores standard oil density
# just call it on init for 0 psig and 60 deg F, that way it is only
# called once during the __init__ method


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
        # store it so you don't have to call it continually
        self.rho_oil_std = oil.condition(0, 60).density

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
                rho_oil (float): density of oil, lbm/ft3
                rho_wat (float): density of water, lbm/ft3
                rho_gas (float): density of gas, lbm/ft3
        """
        rho_oil = self.oil.density
        rho_wat = self.wat.density
        rho_gas = self.gas.density
        return rho_oil, rho_wat, rho_gas

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

    def insitu_volm_flow(self, qoil_std: float) -> tuple[float, float, float]:
        """Insitu Volumetric Flow of Components

        Calculate the insitu volumetric flow rates of the oil, water and gas in ft3/s

        Args:
            qoil_std (float): Oil Rate, BOPD

        Returns:
            qoil (float): Oil Volumetric Flow, Insitu ft3/s
            qwat (float): Water Volumetric Flow, Insitu ft3/s
            qgas (float): Gas Volumetric Flow, Insitu ft3/s
        """
        yoil, ywat, ygas = self.volm_fract()
        qoil, qwat, qgas = self._static_insitu_volm_flow(qoil_std, self.rho_oil_std, self.oil.density, yoil, ywat, ygas)
        return qoil, qwat, qgas

    @staticmethod
    def _static_insitu_volm_flow(
        qoil_std: float, rho_oil_std: float, rho_oil: float, yoil: float, ywat: float, ygas: float
    ) -> tuple[float, float, float]:
        """Insitu Volumetric Flow of Components

        Calculate the insitu volumetric flow rates of the oil, water and gas in ft3/s

        Args:
            qoil_std (float): Oil Rate, BOPD
            rho_oil_std (float): Density Oil at Std Cond, lbm/ft3
            rho_oil (float): Density Oil Insitu Cond, lbm/ft3
            yoil (float): Volm Fraction Oil Insitu Cond, ft3/ft3
            ywat (float): Volm Fraction Water Insitu Cond, ft3/ft3
            ygas (float): Volm Fraction Gas Insitu Cond, ft3/ft3

        Returns:
            qoil (float): Oil Volumetric Flow, Insitu ft3/s
            qwat (float): Water Volumetric Flow, Insitu ft3/s
            qgas (float): Gas Volumetric Flow, Insitu ft3/s
        """
        # 42 gal/bbl, 7.48052 gal/ft3, 24 hr/day, 60min/hour, 60sec/min
        qoil_cfs = qoil_std * 42 / (24 * 60 * 60 * 7.48052)  # ft3/s at standard conditions
        moil = qoil_cfs * rho_oil_std  # mass flow of oil
        qoil = moil / rho_oil  # actual flow, ft3/s

        qtot = qoil / yoil  # oil flow divided by oil total fraction
        qwat = ywat * qtot
        qgas = ygas * qtot

        return qoil, qwat, qgas

    def insitu_mass_flow(self, qoil_std: float) -> tuple[float, float, float]:
        """Insitu Mass Flow of Components

        Calculate the insitu mass flow rates of the oil, water and gas in lbm/s
        The fractions change with pressure and temperature from the gas solubility in the oil.
        As a result the mass fractions cannot be assumed to be constant in the process.

        Args:
            qoil_std (float): Oil Rate, BOPD

        Returns:
            moil (float): Oil Mass Flow, lbm/s
            mwat (float): Water Mass Flow, lbm/s
            mgas (float): Gas Mass Flow, lbm/s
        """
        xoil, xwat, xgas = self.mass_fract()
        moil, mwat, mgas = self._static_insitu_mass_flow(qoil_std, self.rho_oil_std, xoil, xwat, xgas)
        return moil, mwat, mgas

    @staticmethod
    def _static_insitu_mass_flow(
        qoil_std: float, rho_oil_std: float, xoil: float, xwat: float, xgas: float
    ) -> tuple[float, float, float]:
        """Insitu Mass Flow of Components

        Calculate the insitu mass flow rates of the oil, water and gas in lbm/s
        The fractions change with pressure and temperature from the gas solubility in the oil.
        As a result the mass fractions cannot be assumed to be constant in the process.

        Args:
            qoil_std (float): Oil Rate, BOPD
            rho_oil_std (float): Density Oil at Std Cond, lbm/ft3
            xoil (float): Mass Fraction Oil Insitu Cond, lbm/lbm
            xwat (float): Mass Fraction Water Insitu Cond, lbm/lbm
            xgas (float): Mass Fraction Gas Insitu Cond, lbm/lbm

        Returns:
            moil (float): Oil Mass Flow, lbm/s
            mwat (float): Water Mass Flow, lbm/s
            mgas (float): Gas Mass Flow, lbm/s
        """
        # 42 gal/bbl, 7.48052 gal/ft3, 24 hr/day, 60min/hour, 60sec/min
        qoil_cfs = qoil_std * 42 / (24 * 60 * 60 * 7.48052)  # ft3/s at standard conditions
        moil = qoil_cfs * rho_oil_std  # mass flow of oil
        mtot = moil / xoil
        mwat = xwat * mtot
        mgas = xgas * mtot
        return moil, mwat, mgas
