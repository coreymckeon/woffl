class FormWater():

    def __init__(self, wat_sg) -> None:
        """
        Name:   Define Water Stream
        Inputs: wat_sg - water specific gravity
        Output: None
        Rev:    09/22/23 - K. Ellis wrote into Python
        """

        if (0.5 < wat_sg < 1.5) == False:
            print(f'Water SG {wat_sg} Outside Range')

        self.wat_sg = wat_sg

    def __repr__(self) -> str:
        return(f'Water {self.wat_sg} Specific Gravity')

    def condition(self, press, temp) -> None:
        """
        Name:   Condition
        Inputs: press - pressure, psig
                temp - temperature, deg F
        Self:   pressa - absolute pressure, psia
                tempr - absolute temperature Rankine
        Output: None
        Rev:    09/22/23 - K. Ellis wrote into Python
        """
        # define the condition, where are you at?
        # what is the pressure and what is the temperature?
        self.press = press
        self.temp = temp

        # input press in psig
        # input temp in deg F
        self._pressa = self.press + 14.7  # convert psig to psia
        self._tempr = self.temp + 459.67  # convert fahr to rankine

    def density(self):
        """ Name:   Water Density
            Inputs: None
            Output: dwat - density of water (lbm/ft^3)
            Ref:    None
            Rev:    09/22/23 - K. Ellis wrote into Python
        """

        # leave it simple now, asume no compressibility
        dwat = self.wat_sg*62.4
        dwat = round(dwat, 3)

        self.dwat = dwat
        return dwat

    def viscosity(self):
        """ Name:   Water Viscosity   
            Inputs: None
            Output: uwat - Viscosity of water (cP)
            Ref:    None
            Rev:    09/22/23 - K. Ellis wrote into Python
        """

        # come back later and rewrite
        uwat = 0.75  # leave as 0.75 cP for now

        self.uwat = uwat
        return uwat
