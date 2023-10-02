import math


class FormGas:

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

    def __init__(self, gas_sg) -> None:

        if (0.5 < gas_sg < 1.2) == False:
            # do I need more here?
            raise ValueError(f'Gas SG {gas_sg} Outside Range')

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

    # class method can be used to pre define clases, could be useful later in code
    # eg making a ivashak_gas, schrader_gas, etc...
    # especially as various inputs continue to grow
    @classmethod
    def schrader_gas(cls):
        return cls(gas_sg=0.8)

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
        return self

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

    @property
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

        dens = self.density  # call method if you haven't done it already?
        mw = self.mw
        tempr = self._tempr

        K = ((9.4+0.02*mw)*tempr**1.5)/(209+19*mw+tempr)
        X = 3.5+(986/tempr)+0.01*mw
        Y = 2.4-0.2*X

        # [eqn B-72] viscosity of the gas
        visc = (10**-4)*K*math.exp(X*(dens/62.4)**Y)
        visc = round(visc, 5)
        self.ugas = visc
        return visc
