import math

import numpy as np


# NFr, NLv, NGv, Nd, NRe
# HLn, HLs_0, HLs_a
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
        ros_l1 (float): Ros L1 number
        ros_l2 (float): Ros L2 number

    References:
        - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 93
    """
    ros_l1 = sigmoid(ros_nd, 1.02, 43.39, -0.139, 0.99)
    ros_l2 = sigmoid(ros_nd, 0.698, 39.287, 0.093, 0.46)
    return ros_l1, ros_l2


def ros_flow_pattern(ros_ngv: float, ros_nlv: float, ros_nd: float) -> str:
    """Ros Vertical Flow Pattern

    Args:
        ros_ngv (float): Ros Gas Velocity Number, unitless
        ros_nlv (float): Ros Liquid Velocity Number, unitless
        ros_nd (float): Ros Pipe Diameter Number, unitless

    Returns:
        vpat (str): Vertical Flow Pattern

    References:
        - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 92
        - Two Phase Flow in Pipes (1988) Beggs and Brill, Page 3-31"""
    l1, l2 = ros_lp(ros_nd)
    bound_bs = l1 + l2 * ros_nlv  # bubble slug boundary
    bound_st = 50 + 36 * ros_nlv  # slug transistion boundary
    bound_tm = 75 + 84 * ros_nlv**0.75  # transition mist boundary
    # print(f"L1: {l1}, L2 {l2}, Lb: {bound_bs}, Ls: {bound_st}, Lm: {bound_tm}")

    if ros_ngv <= bound_bs:
        vpat = "bubble"
    elif (bound_bs < ros_ngv) and (ros_ngv <= bound_st):
        vpat = "slug"
    elif (bound_st < ros_ngv) and (ros_ngv <= bound_tm):
        vpat = "transistion"
    elif bound_tm < ros_ngv:
        vpat = "mist"
    else:
        # I should throw an error here
        vpat = "unknown"

    return vpat


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
        # need to thrown an error here
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
        - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 58 (error in eqn)
    """
    hlh = a * nslh**b / froude**c
    return hlh


def beggs_holdup_horz(nslh: float, froude: float) -> tuple[float, float, float]:
    """Beggs and Brill Liquid Holdup Horizontal Piping

    Horizontal Holdup with no incline. Calculates all three to be used in interpolating
    if the flow pattern is transistional.

    Args:
        nslh (float): No Slip Liquid Holdup, unitless
        froude (float): Froude Number, unitless

    Return:
        hlh_seg (float): Segregated Slip Horizontal Liquid Holdup
        hlh_int (float): Intermittent Slip Horizontal Liquid Holdup
        hlh_dis (float): Distributed Slip Horizontal Liquid Holdup
    """
    hlh_seg = beggs_holdup_base(nslh, froude, 0.98, 0.4846, 0.0868)
    hlh_int = beggs_holdup_base(nslh, froude, 0.845, 0.5351, 0.0173)
    hlh_dis = beggs_holdup_base(nslh, froude, 1.065, 0.5824, 0.0609)
    return hlh_seg, hlh_int, hlh_dis


def beggs_cf_base(nslh: float, froude: float, ros_nlv: float, e: float, f: float, g: float, h: float) -> float:
    """Beggs and Brill C Factor Base

    Args:
        nslh (float): No Slip Liquid Holdup, unitless
        froude (float): Froude Number, unitless
        ros_nlv (float): Ros Dimensionless Liquid Velocity
        e (float): Beggs Empirical Inclined Coefficient
        f (float): Beggs Empirical Inclined Coefficient
        g (float): Beggs Empirical Inclined Coefficient
        h (float): Beggs Empirical Inclined Coefficient

    Return:
        c (float): Beggs C Factor
    """
    c = (1 - nslh) * math.log(e * (nslh**f) * (ros_nlv**g) * (froude**h))
    return c


def beggs_cf(nslh: float, froude: float, ros_nlv: float) -> tuple[float, float, float, float]:
    """Beggs and Brill C Factor Equation

    Args:
        nslh (float): No Slip Liquid Holdup, unitless
        froude (float): Froude Number, unitless
        ros_nlv (float): Ros Dimensionless Liquid Velocity

    Return:
        c_seg (float): Segregated Beggs C Factor
        c_int (float): Intermittent
        c_dis (float): Distributed
        c_down (float): Dowhill
    """
    c_seg = beggs_cf_base(nslh, froude, ros_nlv, 0.0110, -3.7680, 3.5390, -1.6140)
    c_int = beggs_cf_base(nslh, froude, ros_nlv, 2.9600, 0.3050, -0.4473, 0.0978)
    c_dis = 0
    c_down = beggs_cf_base(nslh, froude, ros_nlv, 4.7, -0.3692, 0.1244, -0.5056)
    return c_seg, c_int, c_dis, c_down


def beggs_phi(c: float, incline: float) -> float:
    """Beggs and Brill Phi

    The phi term is actually psi, but the nomenclature was changed since psi looks
    too much like pounds per square inch when reading it quickly.

    Args:
        c (float): Beggs C Factor
        incline (float): Angle from Horizontal, degrees

    Return:
        phi (float): Liquid Holdup Correction for Inclined Piping
    """
    inc_rad = math.radians(incline)
    phi = 1 + c * (math.sin(1.8 * inc_rad) - 0.333 * (math.sin(1.8 * inc_rad)) ** 3)
    return phi


# inc looks like int...
def beggs_holdup_inc(nslh: float, froude: float, ros_nlv: float, incline: float, hpat: str, tparm: float) -> float:
    """Beggs and Brill Incline Holdup

    Args:
        nslh (float): No Slip Liquid Holdup, unitless
        froude (float): Froude Number, unitless
        ros_nlv (float): Ros Dimensionless Liquid Velocity
        incline (float): Pipe Angle from Horizontal, degrees
        hpat (str): Horizontal Flow Pattern
        tparm (float): Transistional Flow Interpolating Parameter

    Return:
        ilh (float): Inclined Liquid Holdup
    """
    hlh_seg, hlh_int, hlh_dis = beggs_holdup_horz(nslh, froude)
    c_list = list(beggs_cf(nslh, froude, ros_nlv))
    phi_seg, phi_int, phi_dis, phi_down = [beggs_phi(c, incline) for c in c_list]

    # inclined liquid holdup, calculated for each flow regime
    ilh_up = {
        "segregated": hlh_seg * phi_seg,
        "intermittent": hlh_int * phi_int,
        "distributed": hlh_dis * phi_dis,
        "transition": tparm * hlh_seg * phi_seg + (1 - tparm) * hlh_int * phi_int,
    }

    # declined liquid holdup
    ilh_down = {
        "segregated": hlh_seg * phi_down,
        "intermittent": hlh_int * phi_down,
        "distributed": hlh_dis * phi_down,
        "transition": tparm * hlh_seg * phi_down + (1 - tparm) * hlh_int * phi_down,
    }

    if incline < 0:
        ilh = ilh_down[hpat]
    else:
        ilh = ilh_up[hpat]

    return ilh


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


def beggs_yf(nslh: float, ilh: float) -> float:
    """Beggs and Brill Y Factor

    Ratio of no slip liquid holdup to slip liquid holdup squared.
    Page 69 from Al-Safran Book.

    Args:
        nslh (float): No Slip Liquid Holdup
        ilh (float): Slip Liquid Holdup

    Returns:
        y (float): Beggs and Brill y factor
    """
    y = nslh / ilh**2
    return y


def beggs_sf(y: float) -> float:
    """Beggs and Brill S Factor

    Used for properly scalling friction factor for Beggs and Brill piping.
    Page 69 from Al-Safran Book.

    Args:
        y (float): Beggs and Brill y factor

    Returns:
        s (float): Beggs and Brill s factor
    """
    if (1 < y) and (y < 1.2):
        s = math.log(2.2 * y - 1.2)
    else:
        ln_y = math.log(y)
        s = ln_y / (-0.0523 + 3.182 * ln_y - 0.8725 * ln_y**2 + 0.01853 * ln_y**4)
    return s


def beggs_ff(fns: float, s: float) -> float:
    """Beggs and Brill Friction Factor

    Slip friction factor for Beggs and Brill piping.
    Page 69 from Al-Safran Book.

    Args:
        fns: No Slip Friction Factor, Darcy
        s (float): Beggs and Brill s factor

    Returns:
        fs (float): Slip Friction Factor, Beggs and Brill
    """
    fs = fns * math.exp(s)
    return fs


def beggs_ek(p: float, rho_ns: float, vmix: float, vsg: float) -> float:
    """Beggs Dimensionless Kinetic Energy

    Page 69 from Al-Safran Book, Equation 4.42

    Args:
        p (float): Pressure, psig
        rho_ns (float): No Slip Mixture Density, lbm/ft3
        vmix (float): No Slip Mixture Velocity, ft/s
        vsg (float): Superficial Gas Velocity, ft/s

    Returns:
        ek (float): Beggs Dimensionless Kinetic Energy
    """
    p_abs = p + 14.7  # psia
    p_base = p_abs * 144 * 32.174  # lbm/(s2*ft)
    ek = vmix * vsg * rho_ns / p_base
    ek = min(ek, 0.9)  # ensure ek doesn't get bigger than 0.9
    return ek


def beggs_press_static(rho_slip: float, height: float) -> float:
    """Beggs Static Differential Pressure

    Similiar to single phase but the density is the slip mixture.
    Positive height is up, negative height is down.

    Args:
        rho_slip (float): Slip Mixture Density, lbm/ft3
        height (float): Fluid Vertical Height, feet

    Returns:
        dp_stat (float): Static Differential Pressure, psi
    """
    dp_stat = rho_slip * height / 144  # psi, gravity cancels each other out with US Units
    return dp_stat


def beggs_press_friction(fb: float, rho_ns: float, vmix: float, dhyd: float, length: float) -> float:
    """Beggs Frictional Differential Pressure

    Similiar to single phase but uses beggs friction factor, and no-slip mixture
    as well as no slip density. (Which is kind of weird if we have Holdup...)

    Args:
        fb (float): Beggs friction factor of the pipe, unitless
        rho_ns (float): No Slip Mixture Density, lbm/ft3 (why use no slip...?)
        vmix (float): No Slip Mixture Velocity, ft/s
        dhyd (float): Hydraulic Diameter, inches
        length (float): Length of Pipe Segment, feet

    Returns:
        dp_fric (float): Frictional Differential Pressure, psi
    """
    g = 32.174  # 1 lbf equals 32.174 lbm*ft/s2
    dhyd = dhyd / 12  # feet
    dp_fric = fb * rho_ns * vmix**2 * length / (2 * dhyd * g)  # lbf/ft2
    dp_fric = dp_fric / 144  # lbf/in2
    return dp_fric
