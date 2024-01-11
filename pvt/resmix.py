import math

from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater


class ResMix:
    def __init__(self, wc: float, fgor: int, oil: BlackOil, wat: FormWater, gas: FormGas) -> None:
        """Initialize a Reservoir Mixture

        Args:
                wc (float): Watercut of the Mixture, 0 to 1
                fgor (int): Formation GOR of the Mixture, scf/stb
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

        # store standard densities so you don't have to call continually
        pstd = 0  # psig standard pressure
        tstd = 60  # deg f standard temperature
        self.rho_oil_std = oil.condition(pstd, tstd).density
        self.rho_wat_std = wat.condition(pstd, tstd).density
        self.rho_gas_std = gas.condition(pstd, tstd).density

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
        # use static method from below to run the calcs
        xoil, xwat, xgas = self._owg_mass_fraction(
            self.wc, self.fgor, self.oil.gas_solubility(), self.rho_oil_std, self.rho_wat_std, self.rho_gas_std
        )
        # return the mass fractions
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
        xoil, xwat, xgas = self.mass_fract()
        rho_oil, rho_wat, rho_gas = self.dens_comp()

        # mixture specific volume
        vmix = (xoil / rho_oil) + (xwat / rho_wat) + (xgas / rho_gas)
        rho_mix = 1 / vmix
        return rho_mix

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
        xoil, xwat, xgas = self.mass_fract()
        rho_oil, rho_wat, rho_gas = self.dens_comp()

        # mixture specific volume
        vmix = (xoil / rho_oil) + (xwat / rho_wat) + (xgas / rho_gas)
        rho_mix = 1 / vmix

        yoil = xoil * rho_mix / rho_oil
        ywat = xwat * rho_mix / rho_wat
        ygas = xgas * rho_mix / rho_gas

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
        co, cw, cg = self.comp_comp()  # isothermal compressibility
        yoil, ywat, ygas = self.volm_fract()  # volume fractions
        ps = self.pmix()

        cs = co * yoil + cw * ywat + cg * ygas  # mixture compressibility
        ks = 1 / cs  # mixture bulk modulus of elasticity

        cmix = math.sqrt(32.174 * 144 * ks / ps)  # speed of sound, ft/s
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

    @staticmethod
    def _owg_mass_fraction(
        wc: float, fgor: float, rs: float, rho_oil_std: float, rho_wat_std: float, rho_gas_std: float
    ) -> tuple[float, float, float]:
        """Oil Water and Gas Mass Fractions

        Return the mass fractions of the oil, water and gas from the mixture.
        Uses algebraic relationships of wc, gor and density to solve for mass fractions.
        Densities are evaluated at standard conditions

        Args:
            wc (float): Watercut of the Mixture, 0 to 1
            fgor (float): Formation GOR of the Mixture, scf/stb
            rs (float): Gas Solubility of the Oil, scf/stb
            rho_oil_std (float): Oil Density Standard Conditions, lbm/ft3
            rho_wat_std (float): Water Density Standard Conditions, lbm/ft3
            rho_gas_std (float): Gas Density Standard Conditions, lbm/ft3

        Returns:
            xoil (float): mass fraction of oil in the mixture
            xwat (float): mass fraction of water in the mixture
            xgas (float): mass fraction of gas in the mixture

        References:
            Derivations from Kaelin available on request
        """
        # mass based gas solubility
        mrs = (7.48 / 42) * rs * rho_gas_std / rho_oil_std
        # mass formation gas oil ratio, convert from scf/bbl to scf/cf
        mfgor = (7.48 / 42) * fgor * rho_gas_std / rho_oil_std
        # mass watercut
        mwc = rho_wat_std * wc / (rho_wat_std * wc + rho_oil_std * (1 - wc))

        # mass formation gas to liquid ratio
        mfglr = (7.48 / 42) * fgor * rho_gas_std * (1 - wc) / (wc * rho_wat_std + (1 - wc) * rho_oil_std)

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

        # mass fractions
        return xoil, xwat, xgas
