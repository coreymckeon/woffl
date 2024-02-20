class FormWater:
    def __init__(self, wat_sg: float) -> None:
        """Initialize a water stream

        Args:
            wat_sg (float): Water Specific Gravity, 0.5 to 1.5

        Returns:
            self
        """

        if (0.5 < wat_sg < 1.5) is False:
            raise ValueError(f"Water SG {wat_sg} Outside Range")

        self.wat_sg = wat_sg

    def __repr__(self) -> str:
        return f"Water {self.wat_sg} Specific Gravity"

    @classmethod
    def schrader_wat(cls):
        return cls(wat_sg=1.02)

    @classmethod
    def kuparuk_wat(cls):
        return cls(wat_sg=1.02)

    def condition(self, press: float, temp: float):
        """Set condition of evaluation

        Args:
            press (float): Pressure of the water, psig
            temp (float): Temperature of the water, deg F

        Returns:
            self
        """
        # define the condition, where are you at?
        # what is the pressure and what is the temperature?
        self.press = press
        self.temp = temp

        # input press in psig
        # input temp in deg F
        self._pressa = self.press + 14.7  # convert psig to psia
        self._tempr = self.temp + 459.67  # convert fahr to rankine
        return self

    @property
    def density(self) -> float:
        """Water Density

        Calculate water density

        Args:
            None

        Returns:
            dwat (float): water density, lbm/ft3
        """

        # leave it simple now, asume no compressibility
        dwat = self.wat_sg * 62.4
        dwat = round(dwat, 3)

        self.dwat = dwat
        return dwat

    def viscosity(self) -> float:
        """Water Viscosity

        Calculate water viscosity

        Args:
            None

        Returns:
            uwat (float): water viscosity, cP
        """

        # come back later and rewrite
        uwat = 0.75  # leave as 0.75 cP for now

        self.uwat = uwat
        return uwat

    def compress(self) -> float:
        """Water Isothermal Compressibility

        Calculate isothermal water compressibility.
        Inverse of the bulk modulus of elasticity.
        Used water compressibility at 62.4 lbm/ft3 since we don't change density.

        Args:
            None

        Returns:
            wo (float): water compressibility, psi**-1

        References:
            https://roymech.org/Related/Fluids/Fluids_Water_Props.html
        """
        cw_si = 0.0004543  # 1/MPa
        cw = cw_si / 145.038  # 1/psi
        return cw

    def tension(self) -> float:
        """Water Surface Tension

        Calculate the water surface tension

        Args:
            None

        Returns:
            sigw (float): Water Surface Tension, lbf/ft
        """
        sigw_cgs = 72.8  # dyne /cm
        sigw = sigw_cgs * 0.0000685  # lbf/ft
        return sigw
