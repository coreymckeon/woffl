# area for writing two phase flow equations
# mainly will be limited to using beggs / brill
# empirical equation for starting

import math

# beggs and brill horizontal flow pattern prediction
# NFr, NLv, NRe


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
    hlh_dis = beggs_holdup_base(nslh, froude, 1.065, 0.5824, 0.0609)
    hlh_tran = tparm * hlh_seg + (1 - tparm) * hlh_int

    hlh_map = {
        "segregated": hlh_seg,
        "intermittent": hlh_int,
        "distributed": hlh_dis,
        "transition": hlh_tran,
    }
    hlh = hlh_map[hpat]
    return hlh


print(beggs_flow_pattern(0.6, 1.04))
print(beggs_holdup_horz(0.6, 1.04, "intermittent", 1))
