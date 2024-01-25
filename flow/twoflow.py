import math

import numpy as np


# NFr, NLv, NRe
def sigmoid(x: float, L: float, x0: float, k: float, b: float) -> float:
    """Sigmoid Function for Curve Fitting

    Args:
        x (float): Input data
        L (float): Scales Output Range from [0, 1] to [0, L]
        x0 (float): Middle point of Sigmoid on x-axis
        k (float): Scales the input, remains in (-inf, inf)
        b (float): Output bias, changing range from [0, L] to [b, L + b]

    Returns:
        y (float): Sigmoid Function Output
    """
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def froude(vmix: float, dhyd: float) -> float:
    """Froude Number

    Args:
        vmix (float): Homogenous Mixture Velocity, ft/s
        dhyd (float): Hydraulic Diameter, inches

    Returns:
        froude (float): Froude Number, unitless
    """
    g = 32.174  # ft/s2
    d = dhyd / 12  # feet
    froude = vmix**2 / (g * d)
    return froude


def ros_nlv(vsl: float, rho_liq: float, sig_liq: float) -> float:
    """Ros Dimensionless Liquid Velocity

    Args:
        vsl (float): Superficial Liquid Velocity, ft/s
        rho_liq (float): Liquid Density, lbm/ft3
        sig_liq (float): Liquid Surface Tension, lbf/ft

    Return:
        NLv (float): Ros Dimensionless Liquid Velocity
    """
    g = 32.174  # ft/s2
    sig_liq = sig_liq * g  # lbm/s2, unit conversion
    ros_nlv = vsl * (rho_liq / (g * sig_liq)) ** (1 / 4)
    return ros_nlv


def ros_ngv(vsg: float, rho_liq: float, sig_liq: float) -> float:
    """Ros Dimensionless Gas Velocity

    Args:
        vsg (float): Superficial Gas Velocity, ft/s
        rho_liq (float): Liquid Density, lbm/ft3
        sig_liq (float): Liquid Surface Tension, lbf/ft

    Return:
        NGv (float): Ros Dimensionless Liquid Velocity
    """
    g = 32.174  # ft/s2
    sig_liq = sig_liq * g  # lbm/s2, unit conversion
    ros_ngv = vsg * (rho_liq / (g * sig_liq)) ** (1 / 4)
    return ros_ngv


def ros_nd(dhyd: float, rho_liq: float, sig_liq: float) -> float:
    """Ros Dimensionless Pipe Diameter

    Args:
        dhyd (float): Hydraulic Diameter, inches
        rho_liq (float): Liquid Density, lbm/ft3
        sig_liq (float): Liquid Surface Tension, lbf/ft

    Return:
        ros_nd (float): Ros Dimensionless Pipe Diameter
    """
    g = 32.174  # ft/s2
    dhyd = dhyd / 12  # ft
    sig_liq = sig_liq * g  # lbm/s2, unit conversion
    ros_nd = dhyd * (rho_liq * g / sig_liq) ** (1 / 2)
    return ros_nd


def ros_lp(ros_nd: float) -> tuple[float, float]:
    """Ros L1 and L2 Bubble to Slug Transistion Parameter

    Ros did not provide an equation. Only a figure was provided to look at.
    The sigmoid function was used for a curve fit of the Ros figure.

    Args:
        ros_nd (float): Ros Pipe Diameter Number

    Return:
        ros_l1 (float): Ros l1 number
        ros_l2 (float): Ros l2 number
    """
    ros_l1 = sigmoid(ros_nd, 1.02, 43.39, -0.139, 0.99)
    ros_l2 = sigmoid(ros_nd, 0.698, 39.287, 0.093, 0.46)
    return ros_l1, ros_l2


def beggs_flow_pattern(nslh: float, froude: float) -> tuple[str, float]:
    """Beggs and Brill Horizontal Flow Pattern

    Args:
        nslh (float): No Slip Liquid Holdup, unitless
        froude (float): Froude Number, unitless

    Return:
        hpat (str): Horizontal Flow Pattern
        tran (float): Transitional Flow Interpolating Parameter

    References:
        - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 52
    """
    l1 = 316 * nslh**0.302
    l2 = 0.000925 * nslh**-2.468
    l3 = 0.1 * nslh**-1.468
    l4 = 0.5 * nslh**-6.738

    tran = 1  # zero for all flow patterns except for transitional

    if (nslh < 0.01) and (froude < l1):
        hpat = "segregated"
    elif (nslh >= 0.01) and (froude < l2):
        hpat = "segregated"
    elif (nslh >= 0.01) and (l2 <= froude) and (froude <= l3):
        hpat = "transition"  # can I pack the A value in here?
        tran = (l3 - froude) / (l3 - l2)
    elif (0.01 <= nslh) and (nslh < 0.4) and (l3 <= froude) and (froude <= l1):
        hpat = "intermittent"
    elif (nslh >= 0.4) and (l3 <= froude) and (froude <= l4):
        hpat = "intermittent"
    elif (nslh < 0.4) and (froude >= l1):
        hpat = "distributed"
    elif (nslh >= 0.4) and (froude > l4):
        hpat = "distributed"
    else:
        hpat = "unknown"

    return hpat, tran


def beggs_holdup_base(nslh: float, froude: float, a: float, b: float, c: float) -> float:
    """Beggs and Brill Liquid Holdup Horizontal Piping

    Args:
        nslh (float): No Slip Liquid Holdup, unitless
        froude (float): Froude Number, unitless
        a (float): Empirical Horizontal Flow Coefficient
        b (float): Empirical Horizontal Flow Coefficient
        c (float): Empirical Horizontal Flow Coefficient

    Return:
        hlh (float): Slip Horizontal Liquid Holdup

    References:
        - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 58
    """
    hlh = a * nslh**b / ((froude**2) ** c)
    return hlh


def beggs_holdup_horz(nslh: float, froude: float, hpat: str, tparm: float) -> float:
    """Beggs and Brill Liquid Holdup Horizontal Piping

    Args:
        nslh (float): No Slip Liquid Holdup, unitless
        froude (float): Froude Number, unitless
        hpat (str): Horizontal Flow Pattern
        tparm (float): Transitional Flow Interpolating Parameter

    Return:
        hlh (float): Slip Horizontal Liquid Holdup
    """
    hlh_seg = beggs_holdup_base(nslh, froude, 0.98, 0.4846, 0.0868)
    hlh_int = beggs_holdup_base(nslh, froude, 0.845, 0.5351, 0.0173)
    hlh_tran = tparm * hlh_seg + (1 - tparm) * hlh_int
    hlh_dis = beggs_holdup_base(nslh, froude, 1.065, 0.5824, 0.0609)

    hlh_map = {
        "segregated": hlh_seg,
        "intermittent": hlh_int,
        "distributed": hlh_dis,
        "transition": hlh_tran,
    }
    hlh = hlh_map[hpat]
    return hlh


def beggs_cfactor(nslh: float, ros_nlv: float, froude: float, e: float, f: float, g: float, h: float) -> float:
    """Beggs and Brill C Factor Equation

    Args:
        nslh (float): No Slip Liquid Holdup, unitless
        ros_nlv (float): Ros Dimensionless Liquid Velocity
        froude (float): Froude Number, unitless
        e (float): Beggs Empirical Inclined Coefficient
        f (float): Beggs Empirical Inclined Coefficient
        g (float): Beggs Empirical Inclined Coefficient
        h (float): Beggs Empirical Inclined Coefficient

    Return:
        hlh (float): Slip Horizontal Liquid Holdup
    """
    c = (1 - nslh) * math.log(e * (nslh**f) * (ros_nlv**g) * (froude**h))
    return c


# NFr, NLv, NGv, NRe
def beggs_psi(nslh: float, ros_nlv: float, froude: float, incline: float, hpat: str, tparm: float) -> float:
    """Beggs and Brill Psi

    Args:
        nslh (float): No Slip Liquid Holdup, unitless
        ros_nlv (float): Ros Dimensionless Liquid Velocity
        froude (float): Froude Number, unitless
        incline (float): Pipe Angle from Horizontal, degrees (?)
        hpat (str): Horizontal Flow Pattern
        tparm (float): Transistional Flow Interpolating Parameter (do we need this?)

    Return:
        psi (float): Liquid Holdup Correction for Inclined Piping
    """
    inc_rad = math.radians(incline)

    # does interpolating on psi do the same thing as interpolating on the angled liquid holdup?
    # psi is multipled by HL...?
    # interpolate on psi and interpolate on HL horizontal, does that still work? or am I double interp?

    c_seg = beggs_cfactor(nslh, ros_nlv, froude, 0.0110, -3.7680, 3.5390, -1.6140)
    c_int = beggs_cfactor(nslh, ros_nlv, froude, 2.9600, 0.3050, -0.4473, 0.0978)
    c_tran = tparm * c_seg + (1 - tparm) * c_int  # this doesn't work...
    c_dis = 0
    c_down = beggs_cfactor(nslh, ros_nlv, froude, 4.7, -0.3692, 0.1244, -0.5056)

    if incline < 0:
        hpat = "downhill"

    c_map = {"segregated": c_seg, "intermittent": c_int, "distributed": c_dis, "transition": c_tran, "downhill": c_down}
    c = c_map[hpat]

    psi = 1 + c * (math.sin(1.8 * inc_rad) - 0.333 * (math.sin(1.8 * inc_rad)) ** 3)
    return psi


def payne_correction(ilh: float, incline: float) -> float:
    """Payne Inclined Correction Factor

    Applies the Payne et. al. (1979) correction factor.

    Args:
        ilh (float): Inclined Liquid Holdup
        incline(float): Pipe Angle from Horizontal, degrees

    Returns:
        clh (float): Corrected Liquid Holdup
    """
    if incline > 0:
        clh = 0.924 * ilh
    elif incline < 0:
        clh = 0.685 * ilh
    else:
        clh = ilh
    return clh


print(beggs_flow_pattern(0.6, 1.04))
print(beggs_holdup_horz(0.6, 1.04, "intermittent", 1))
