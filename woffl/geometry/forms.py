"""Forms of Mechanical Components

The following file is a place to house basic equations for calculating geometric properties.
Basic examples of the properties are the area of circle, perimeter, or angle of a right triangle.
The important configuration of these equations are they do not contain units. Units need to be
handled in other places...(is this going to get confusing?)
"""

import math

import numpy as np


def _area_circle(diameter: float) -> float:
    """Area of a Circle

    Args:
        diameter (float): diameter of a circle

    Returns:
        area (float): area of a circle
    """
    area = math.pi * (diameter**2) / 4
    return area


def vert_angle(hlen: float, ylen: float) -> float:
    """Vertical Angle, degrees

    Imagine a triangle whose hypotenuse starts at (0, 0) and ends at (x1, y1).
    The angle being calculated is between the hypotenuse and y-axis.

    Args:
        hlen (float): Length of the hypotenuse
        ylen (float): Length along the y-axis

    Returns:
        theta (float): Vertical angle, degrees
    """
    theta = np.arccos(ylen / hlen)  # angle in radians
    theta = np.degrees(theta)  # convert to degrees
    return theta


def horz_angle(hlen: float, ylen: float) -> float:
    """Horizontal Angle, degrees

    Imagine a triangle whose hypotenuse starts at (0, 0) and ends at (x1, y1).
    The angle being calculated is between the hypotenuse and x-axis.

    Args:
        hlen (float): Length of the hypotenuse
        ylen (float): Length along the y-axis

    Returns:
        theta (float): Horizontal angle, degrees
    """
    theta = np.arcsin(ylen / hlen)  # angle in radians
    theta = np.degrees(theta)  # convert to degrees
    return theta
