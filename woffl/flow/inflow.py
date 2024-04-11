import math


class InFlow:
    def __init__(self, qwf: float, pwf: float, pres: float) -> None:
        """Initialize Well InFlow Parameters

        The well inflow class is the sandface ability to produce oil.
        Either a vogel curve or productivity index can be defined from
        the specified parameters. The oil rate is used in conjuction with
        a reservoir mixtures wc and gor to calculate other components.

        Args:
            qwf (float): Oil Rate at pwf, STB/D
            pwf (float): Flowing bottomhole pressure, psig
            pres (float): Reservoir Pressure, psig

        Returns:
            Self
        """
        if pwf > pres is True:
            raise ValueError("Flowing pressure is greater than reservoir pressure")

        self.qwf = qwf
        self.pwf = pwf
        self.pres = pres

    def __repr__(self) -> str:
        return f"Tested inflow of {self.qwf} bopd at {self.pwf} psig"

    @staticmethod
    def prod_index(qwf: float, pwf: float, pres: float) -> float:
        """Productivity Index of the Wellbore

        Calculate the straight line productivity of the wellbore.

        Args:
            qwf (float): Oil Rate at pwf, STB/D
            pwf (float): Flowing bottomhole pressure, psig
            pres (float): Reservoir Pressure, psig

        Returns:
            pidx (float): Productivity Index (STB/(day*psi))
        """
        pidx = qwf / (pres - pwf)  # stb/(day*psi)
        return pidx

    @staticmethod
    def vogel_qmax(qwf: float, pwf: float, pres: float) -> float:
        """Vogel max flow of the well

        Calculate the max flow of the well with the test data. Qmax is used
        later to calculate how much the well can flow at a new pwf.

        Args:
            qwf (float): Oil Rate at pwf, stb/d
            pwf (float): Flowing bottomhole pressure, psig
            pres (float): Reservoir Pressure, psig

        Returns:
            qmax (float): Vogel Max Theoretical Flowrate

        """
        qmax = qwf / (1 - 0.2 * (pwf / pres) - 0.8 * (pwf / pres) ** 2)
        return qmax

    def oil_flow(self, pnew: float, method: str = "vogel") -> float:
        """Oil flowrate at a new pwf, pnew

        Calculate an oil production rate at a specific bottomhole pressure.
        Could always just call it pnew, for a new bottomhold pressure?

        Args:
            pnew (float): Flowing bottomhole pressure, psig, different than self
            method (str): "vogel" or "pidx" (prod index)

        Returns:
            qnew (float): Oil rate at pnew, stb/d
        """
        if pnew < 0:
            ValueError("Flowing pressure must be greater than zero")

        if pnew > self.pres:
            ValueError("Flowing pressure must be less than reservoir pressure")

        if method == "vogel":
            qmax = self.vogel_qmax(self.qwf, self.pwf, self.pres)
            qnew = qmax * (1 - 0.2 * (pnew / self.pres) - 0.8 * (pnew / self.pres) ** 2)

        else:
            pidx = self.prod_index(self.qwf, self.pwf, self.pres)
            qnew = pidx * (self.pres - pnew)

        return qnew
