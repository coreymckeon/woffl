# type: ignore
# flake8: noqa

# a place to store legacy code
# as stuff is cleaned up


def tee_near_pmo(psu: float, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix) -> float:
    """Throat Entry Equation near pmo, mach 1 pressure

    Find the value of the throat entry equation near the mach 1 pressure. The following
    function will be iterated across to minimize the throat entry equation.

    Args:
        psu (float): Suction Press, psig
        tsu (float): Suction Temp, deg F
        ken (float): Throat Entry Friction, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        tee_pmo (float): Throat Entry Value near pmo, psig"""

    qoil_std, pte_ray, rho_ray, vel_ray, snd_ray = throat_entry_arrays(psu, tsu, ate, ipr_su, prop_su)
    pmo = throat_entry_mach_one(pte_ray, vel_ray, snd_ray)
    mask = pte_ray >= pmo  # only use values where pte_ray is greater than pmo, haven't hit mach 1
    # note: do we even have to calc  and filter pmo? TEE vs pte is a parabola which anyway...?
    # discontinuities with mask function might screw all this up...
    kse_ray, ese_ray = throat_entry_energy(ken, pte_ray[mask], rho_ray[mask], vel_ray[mask])
    tee_ray = kse_ray + ese_ray
    tee_pmo = min(tee_ray)  # find the smallest value of tee where mach <=1
    return tee_pmo


def cross_zero_tee(
    ken: float, pte_ray: np.ndarray, rho_ray: np.ndarray, vel_ray: np.ndarray, snd_ray: np.ndarray
) -> tuple[float, float, float]:
    """Throat Entry Parameters with a zero TEE

    Calculate the throat entry pressure, density, and velocity where TEE crosses zero.
    Valid for one suction pressure  of the pump / reservoir.

    Args:
        ken (float): Throat Entry Friction, unitless
        pte_ray (np array): Press Throat Entry Array, psig
        rho_ray (np array): Density Throat Entry Array, lbm/ft3
        vel_ray (np array): Velocity Throat Entry Array, ft/s
        snd_ray (np array): Speed of Sound Array, ft/s

    Return:
        pte (float): Throat Entry Pressure, psig
        rho_te (float): Throat Entry Density, lbm/ft3
        vte (float): Throat Entry Velocity, ft/s
    """
    pmo = throat_entry_mach_one(pte_ray, vel_ray, snd_ray)
    mask = pte_ray >= pmo  # only use values where pte_ray is greater than pmo, haven't hit mach 1
    kse_ray, ese_ray = throat_entry_energy(ken, pte_ray[mask], rho_ray[mask], vel_ray[mask])
    tee_ray = kse_ray + ese_ray
    # is there a way to speed up all these interpolations?
    pte = np.interp(0, np.flip(tee_ray), np.flip(pte_ray[mask]))
    rho_te = np.interp(0, np.flip(tee_ray), np.flip(rho_ray[mask]))
    vte = np.interp(0, np.flip(tee_ray), np.flip(vel_ray[mask]))
    return pte, rho_te, vte


# find where mach = 1 (pmo), insert pmo into pte, calculate rho, vel and snd at arrays
pmo = throat_entry_mach_one(pte_ray, vel_ray, snd_ray)
# flip array for ascedning order instead of descending
pmo_idx = np.searchsorted(np.flip(pte_ray), pmo) - pte_ray.size  # find position of pmo

# repeat finding properties where Mach Number equals one
prop_su = prop_su.condition(pmo, tsu)
rho_oil = prop_su.oil.density  # oil density
yoil, ywat, ygas = prop_su.volm_fract()
qoil, qwat, qgas = actual_flow(qoil_std, rho_oil_std, rho_oil, yoil, ywat, ygas)
qtot = qoil + qwat + qgas

# insert values where Mach Number equals one
pte_ray = np.insert(arr=pte_ray, obj=pmo_idx, values=pmo)
vel_ray = np.insert(arr=vel_ray, obj=pmo_idx, values=qtot / ate)
rho_ray = np.insert(arr=rho_ray, obj=pmo_idx, values=prop_su.pmix())
snd_ray = np.insert(arr=snd_ray, obj=pmo_idx, values=prop_su.cmix())


@staticmethod
def almarhoun_fvf_above(press, temp, oil_api, pbp, gas_sg, rs, bob) -> float:
    """Al-Marhoun FVF Above Bubblepoint

    Calculate the formation volume factor. Using Al-Marhoun
    In the 2015 Paper Al-Marhoun wrote, he doesn't discuss this method
    As a result, I am guessing that he doesn't want to use it.

    Args:
        press (float): Pressure of the oil, psig
        temp (float): Temperature of the oil, deg F
        oil_api (float): Oil API Degrees
        pbp (float): Bubblepoint pressure, psig
        gas_sg (float): Gas Specific Gravity, air
        rs (float): Solubility of Gas in Oil, SCF/STB
        bob (float): Oil FVF at Bubblepoint, RB/STB

    Returns:
        bo (float): Oil FVF, rb/stb

    References:
        - New Correlations for FVF of Oil and Gas, M. Al-Marhoun (1992) PETSOC-92-03-02
    """

    # oil_sg = 141.5 / (oil_api + 131.5)  # oil specific gravity

    # oil fvf above bubblepoint pressure
    a5 = -0.0136680 * 10**-3
    a6 = -0.0195682 * 10**-6
    a7 = 0.02409026
    a8 = -0.0926019 * 10**-6

    c = a5 * rs + a6 * rs**2 + a7 * gas_sg + a8 * (temp + 460) ** 2

    bo = bob * (press / pbp) ** c
    return bo


# rename insitu_volm_flow or actual_volm_flow
# created a method inside ResMix, that does just this
def actual_flow(
    qoil_std: float, rho_oil_std: float, rho_oil: float, yoil: float, ywat: float, ygas: float
) -> tuple[float, float, float]:
    """Actual Flow of Mixture

    Calculate the actual flow rates of the oil, water and gas in ft3/s

    Args:
        qoil_std (float): Oil Rate, BOPD
        rho_oil_std (float): Density Oil at Std Cond, lbm/ft3
        rho_oil (float): Density Oil Act. Cond, lbm/ft3
        yoil (float): Volm Fraction Oil Act. Cond, ft3/ft3
        ywat (float): Volm Fraction Water Act. Cond, ft3/ft3
        ygas (float): Volm Fraction Gas Act. Cond, ft3/ft3

    Returns:
        qoil (float): Oil rate, actual ft3/s
        qwat (float): Water rate, actual ft3/s
        qgas (float): Gas Rate, actual ft3/s
    """
    # 42 gal/bbl, 7.48052 gal/ft3, 24 hr/day, 60min/hour, 60sec/min
    qoil_cfs = qoil_std * 42 / (24 * 60 * 60 * 7.48052)  # ft3/s at standard conditions
    moil = qoil_cfs * rho_oil_std  # mass flow of oil
    qoil = moil / rho_oil  # actual flow, ft3/s

    qtot = qoil / yoil  # oil flow divided by oil total fraction
    qwat = qtot * ywat
    qgas = qtot * ygas
    return qoil, qwat, qgas


def total_actual_flow(qoil_std: float, rho_oil_std: float, prop: ResMix) -> float:
    """Total Actual Flow of Mixture

    Calculate the total actual flow of the three phase mixture. Requires a
    condition (pressure, temp) to have already been set in the ResMix. The
    standard density of oil could have been looked up in the ResMix, but multiple
    condition swapping was desired to be avoided. (Note: Update ResMix later to
    store standard oil density?)

    Args:
        qoil_std (float): Oil Rate, BOPD
        rho_oil_std (float): Density Oil at Std Cond, lbm/ft3
        prop (ResMix): Properties of 3-Phase Mixutre

    Returns:
        qtot (float): Total Mixture Rate, actual ft3/s
    """
    rho_oil = prop.oil.density  # oil density
    yoil, ywat, ygas = prop.volm_fract()
    qoil, qwat, qgas = actual_flow(qoil_std, rho_oil_std, rho_oil, yoil, ywat, ygas)
    qtot = qoil + qwat + qgas
    return qtot


def throat_dp(kth: float, vnz: float, anz: float, rho_nz: float, vte: float, ate: float, rho_te: float):
    """Throat Differential Pressure

    Solves the throat mixture equation of the jet pump. Calculates throat differntial pressure.
    Use the throat entry pressure and differential pressure to calculate throat mix pressure.
    ptm = pte - dp_th. The biggest issue with this equation is it assumes the discharge conditions
    are at same conditions as the inlet. This is false. There is an increase in pressure across the
    diffuser. Which using this equation equates potentially 600 psig.

    Args:
        kth (float): Friction of Throat Mix, Unitless
        vnz (float): Velocity of Nozzle, ft/s
        anz (float): Area of Nozzle, ft2
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3
        vte (float): Velocity of Throat Entry Mixture, ft/s
        ate (float): Area of Throat Entry, ft2
        rho_te (float): Density of Throat Entry Mixture, lbm/ft3

    Returns:
        dp_th (float): Throat Differential Pressure, psid
    """

    mnz = vnz * anz * rho_nz  # mass flow of the mozzle
    qnz = vnz * anz  # volume flow of the nozzle

    mte = vte * ate * rho_te  # mass flow of the throat entry
    qte = vte * ate  # volume flow of the throat entry

    ath = anz + ate  # area of the throat

    mtm = mnz + mte  # mass flow of total mixture
    vtm = (vnz * anz + vte * ate) / ath  # velocity of total mixture
    rho_tm = (mnz + mte) / (qnz + qte)  # density of total mixture

    # units of lbm/(s2*ft)
    dp_tm = 0.5 * kth * rho_tm * vtm**2 + mtm * vtm / ath - mnz * vnz / ath - mte * vte / ath
    # convert to lbf/in2
    dp_tm = dp_tm / (32.174 * 144)
    return dp_tm, vtm
