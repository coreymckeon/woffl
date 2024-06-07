import math


class Pipe:
    def __init__(self, out_dia: float, thick: float, abs_ruff: float = 0.004) -> None:
        """Initialize a Piece of Piping

        The pipe doesn't have a length. The object will work with the profile
        class to create an appropriate length and angle of piping.

        Args:
            out_dia (float): Outside diameter of the pipe, inches
            thick (float): Thickness of the piping, inches
            abs_ruff (float): Abs Roughness, inches, Default: 0.004 inches

        Returns:
            Self
        """

        if thick > out_dia:
            raise ValueError("Pipe thickness is greater than outer diameter")

        self.out_dia = out_dia
        self.thick = thick
        self.abs_ruff = abs_ruff

        self.inn_dia = out_dia - 2 * thick  # inner diamter

    def __repr__(self):
        return f"Pipe: Out Dia {self.out_dia} inches and Inn Dia {self.inn_dia} inches"

    @property
    def inn_area(self) -> float:
        """Inner area of piping, ft2"""
        return self.area_circle(self.inn_dia) / 144

    @property
    def out_area(self) -> float:
        """Outer area of piping, ft2"""
        return self.area_circle(self.out_dia) / 144

    @staticmethod
    def area_circle(diameter):
        """Area of a Circle

        Args:
            diameter (float): diameter of a circle

        Returns:
            area (float): area of a circle
        """
        area = math.pi * (diameter**2) / 4
        return area


class Annulus:
    def __init__(self, inn_pipe: Pipe, out_pipe: Pipe) -> None:
        """Initialize a pipe annulus

        The annulus doesn't have a length. The object will work with the
        profile class to create an appropriate length and angle of annulus.

        Args:
            inn_pipe (Pipe): Inside Pipe in an annulus set up
            out_pipe (Pipe): Outside Pipe in an annulus set up

        Returns:
            Self
        """
        if inn_pipe.out_dia > out_pipe.inn_dia:
            raise ValueError("Inner pipe will not fit into the outer pipe")

        self.inn_pipe = inn_pipe
        self.out_pipe = out_pipe

    def __repr__(self):
        return f"Inner Pipe OD: {self.inn_pipe.out_dia} inches, Outer Pipe OD: {self.out_pipe.out_dia} inches"

    @property
    def ann_area(self) -> float:
        """Annulus Cross Sectional Area, ft2"""
        return self.out_pipe.inn_area - self.inn_pipe.out_area

    @property
    def hyd_dia(self) -> float:
        """Annulus Hydraulic Diameter, inches"""
        return self.out_pipe.inn_dia - self.inn_pipe.out_dia
