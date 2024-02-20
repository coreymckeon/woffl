import math


class FormGas:
    # physical properties of hydrocarbon and selected compounds, field units (Whitson and Brule 2000)
    # Table B-1 from the text "Applied Multiphase Flow in Pipes and Flow Assurance"

    # turns the black formatting on or off
    # fmt: off
    # flake8: noqa
    fyprop = {'n2':  [28.02, 0.4700, 29.31,  493.0, 227.3, 1.443, 0.2916, 0.0450,  13.3, 0, 0],
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
    # fmt: on
    # flake8: qa

    # will leave the dictionary lookups alone for now, come back later for molecular composition

    # ideal gas constant
    _R = 10.73  # psia*ft^3/(lbmol*Rankine)

    def __init__(self, gas_sg) -> None:
        """Initialize a Formation Gas Stream

        Args:
            gas_sg (float): Free Gas Specific Gravity

        Returns:
            Self
        """

        if (0.5 < gas_sg < 1.2) is False:
            # do I need more here?
            raise ValueError(f"Gas SG {gas_sg} Outside Range")

        # define a lot of properties just on the gas' specific gravity
        self.gas_sg = gas_sg

        # the following is the code for Pseudo Critical Pressure from correlations
        # can be adjusted later for H2S or CO2 if necessary
        self._pcrit = 756.8 - 131.07 * gas_sg - 3.6 * gas_sg**2  # psia

        # the following is the code for Pseudo Critical Temp from correlations
        # can be adjusted later for H2S or CO2 if necessary
        self._tcrit = 169.2 + 349.5 * gas_sg - 74.0 * gas_sg**2  # rankine

        self.mw = 28.96443 * gas_sg

    def __repr__(self):
        return f"Gas: {self.gas_sg} SG and {self.mw} Mol Weight"

    @classmethod
    def schrader_gas(cls):
        return cls(gas_sg=0.65)

    @classmethod
    def kuparuk_gas(cls):
        return cls(gas_sg=0.65)

    @classmethod
    def methane_gas(cls):
        return cls(gas_sg=0.55)

    # almost need seperate function to change pressure / temperature
    def condition(self, press, temp):
        """Set condition of evaluation

        Args:
            press (float): Pressure of the gas, psig
            temp (float): Temperature of the gas, deg F

        Returns:
            Self
        """
        self.press = press
        self.temp = temp

        self._pressa = press + 14.7  # convert psig to psia
        self._tempr = temp + 459.67  # convert fahr to rankine

        # not adjusted for non-hydrocarbon gas such as H2S or CO2
        self._ppr = self._pressa / self._pcrit  # unitless, pressure pseudo reduced
        self._tpr = self._tempr / self._tcrit  # unitless, temperature pseudo reduced
        return self

    def zfactor(self) -> float:
        """Gas Z-Factor Compressibility

        Return the gas z-factor

        Args:
            None

        Returns:
            zfactor(float): gas zfactor, no units

        References:
            Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 16
            Applied Multiphase Flow in Pipes..., Al-Safran and Brill (2017) Page 305
        """
        m = 0.51 * (self._tpr) ** (-4.133)
        n = 0.038 - 0.026 * (self._tpr) ** (1 / 2)
        zfactor = 1 - m * self._ppr + n * self._ppr**2 + 0.0003 * self._ppr**3
        return zfactor

    @property
    def density(self) -> float:
        """Gas Density, lbm/ft3

        Return the density of gas.
        Requires pressure and temperature to previously be set.

        Args:
            None

        Returns:
            dgas (float): density of the gas, lbm/ft3

        References:
            Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 16
            Applied Multiphase Flow in Pipes..., Al-Safran and Brill (2017) Page 305
        """
        zval = self.zfactor()  # call method if it hasn't been already?
        dgas = self._pressa * self.mw / (zval * FormGas._R * self._tempr)
        return dgas

    def viscosity(self) -> float:
        """Gas Viscosity, cP

        Return the gas viscosity, cP

        Args:
            None

        Returns:
            ugas (float): gas viscosity, cP
        """
        ug = self._viscosity_lee(self._tempr, self.mw, self.density)
        return ug

    def compress(self) -> float:
        """Gas Compressibility Isothermal

        Calculate isothermal gas compressibility.

        Args:
            None

        Returns:
            cg (float): gas compressibility, psi**-1

        References:
            Fundamental Principles of Reservoir Engineering, B.Towler (2002) Page 16
            Applied Multiphase Flow in Pipes..., Al-Safran and Brill (2017) Page 305
        """
        temp = self.temp
        p1 = self.press
        p2 = p1 + 10  # add 100 psi to evaluate a different condition for compressibility

        z1 = self.zfactor()
        z2 = self.condition(p2, temp).zfactor()

        cg = 1 / p1 - (1 / z1) * ((z2 - z1) / (p2 - p1))
        return cg

    def hydrate_check(self) -> tuple[bool, float]:
        """DO NOT USE

        Hydrate Formation Check
        Checks to see if the specified pressure and temperature is at risk of hydrates.
        Calculate hydrate formation temperature at specified pressure.
        Requires a pressure and temperature to be specified.

        Args:
            None

        Returns:
            hydrate risk (bool): True or False
            hydrate temp (float): Return Temperature of hydrate formation at pressure

        References:
            - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill Page 176
        """
        # placeholder for future work
        return True, 55

    @staticmethod
    def _viscosity_lee(tempr: float, mw: float, rho_gas: float) -> float:
        """Lee et. al Gas Viscosity

        Gas viscosity following Lee 1966 et al methodology.

        Args:
            tempr (float): Absolute Temperature of Gas, deg R
            mw (float): Molecular Weight of Gas Mixture, lb/lb-mol
            rho_gas (float): In-Situ Density of Gas, lbm/ft3

        Returns:
            ug (float): Gas Viscosity, cP

        References:
            - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 304
        """
        # tempr = temp + 459.67
        K = ((9.4 + 0.02 * mw) * tempr**1.5) / (209 + 19 * mw + tempr)
        X = 3.5 + (986 / tempr) + 0.01 * mw
        Y = 2.4 - 0.2 * X
        ug = (10**-4) * K * math.exp(X * (rho_gas / 62.4) ** Y)
        return ug
