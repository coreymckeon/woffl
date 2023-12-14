import math
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz, simps, trapz

from flow.inflow import InFlow
from geometry.pipe import Annulus, Pipe
from jetpump import JetPump
from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater
from pvt.resmix import ResMix

"""This file is nothing more than a collection of Kaelin trying to make
some jet pump theory work in a practical method. At this point I don't
really like Bobs write up on pto vs pte and I will attempt my own method.
I'm going to just make static functions in an attempt to make this easier
"""

jpump_md = 6693  # feet
jpump_tvd = 4096.8  # feet, interpolated off well profile

bpd_pf = 3500  # bwpd, power fluid rate
d_pf = 62.4  # lbm/ft3
u_pf = 1  # cp, viscosity of power fluid
ppf_surf = 3046  # psi, power fluid surf pressure

# testing the jet pump code on E-42
tube = Pipe(out_dia=4.5, thick=0.5)  # E-42 tubing
case = Pipe(out_dia=6.875, thick=0.5)  # E-42 casing
ann = Annulus(tube, case)  # define the annulus

ipr = InFlow(qwf=350, pwf=800, pres=1400)  # define an ipr

e42jetpump = JetPump("13", "B")

mpu_oil = BlackOil.schrader_oil()  # class method
mpu_wat = FormWater.schrader_wat()  # class method
mpu_gas = FormGas.schrader_gas()  # class method

e42 = ResMix(0.75, 1500, mpu_oil, mpu_wat, mpu_gas)

temp = 80
press = 700


# should this sit in InFlow? Or a seperate class?
def actual_flow(
    oil_rate: float, poil_std: float, poil: float, yoil: float, ywat: float, ygas: float
) -> tuple[float, float, float]:
    """Actual Flow of Mixture

    Calculate the actual flow rates of the oil, water and gas in ft3/s

    Args:
        oil_rate (float): Oil Rate, BOPD
        poil_std (float): Density Oil at Std Cond, lbm/ft3
        poil (float): Density Oil Act. Cond, lbm/ft3
        yoil (float): Volm Fraction Oil Act. Cond, ft3/ft3
        ywat (float): Volm Fraction Water Act. Cond, ft3/ft3
        ygas (float): Volm Fraction Gas Act. Cond, ft3/ft3

    Returns:
        qoil (float): Oil rate, actual ft3/s
        qwat (float): Water rate, actual ft3/s
        qgas (float): Gas Rate, actual ft3/s
    """
    # 42 gal/bbl, 7.48052 gal/ft3, 24 hr/day, 60min/hour, 60sec/min
    qoil_std = oil_rate * 42 / (24 * 60 * 60 * 7.48052)  # ft3/s at standard conditions
    moil = qoil_std * poil_std
    qoil = moil / poil

    qtot = qoil / yoil  # oil flow divided by oil total fraction
    qwat = ywat * qtot
    qgas = ygas * qtot

    return qoil, qwat, qgas


def throat_entry_mach_one(pte_ray: np.ndarray, vel_ray: np.ndarray, snd_ray: np.ndarray) -> float:
    """Throat Entry Mach One

    Calculates the pressure where the throat entry flow hits sonic velocity, mach = 1

    Args:
        pte_ray (np array): Press Throat Entry Array, psig
        vel_ray (np array): Velocity Throat Entry Array, ft/s
        snd_ray (np array): Speed of Sound Array, ft/s

    Returns:
        pmo (float): Pressure Mach One, psig
    """
    mach_ray = vel_ray / snd_ray
    # check that the mach array has values that span one for proper interpolation
    if np.max(mach_ray) <= 1:
        raise ValueError("Max value in Mach array is less than one, increase pte")
    if np.min(mach_ray) >= 1:
        raise ValueError("Min value in Mach array is greater than one, decrease pte")
    pmo = np.interp(1, mach_ray, pte_ray)
    return pmo


def throat_entry_arrays(psu: float, tsu: float, ate: float, ipr_su: InFlow, prop_su: ResMix):
    """Throat Entry Raw Arrays

    Create a series of throat entry arrays. The arrays can be graphed to visualize.
    What is occuring inside the throat entry while pressure is dropped. Keeps all the
    values, even where pte velocity is greater than mach 1.

    Args:
        psu (float): Suction Pressure, psig
        tsu (float): Suction Temp, deg F
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        qoil_std (float): Oil Rate, STD BOPD
        pte_ray (np array): Press Throat Entry Array, psig
        rho_ray (np array): Density Throat Entry Array, lbm/ft3
        vel_ray (np array): Velocity Throat Entry Array, ft/s
        snd_ray (np array): Speed of Sound Array, ft/s
        mach_ray (np array): Mach Number Array, unitless
    """
    rho_oil_std = prop_su.oil.condition(0, 60).density  # oil standard density
    qoil_std = ipr_su.oil_flow(psu, method="pidx")  # oil standard flow, bopd

    ray_len = 30  # number of elements in the array

    # create empty arrays to fill later
    vel_ray = np.empty(ray_len)
    rho_ray = np.empty(ray_len)
    snd_ray = np.empty(ray_len)

    pte_ray = np.linspace(200, psu, ray_len)  # throat entry pressures
    pte_ray = np.flip(pte_ray, axis=0)  # start with high pressure and go low

    for i, pte in enumerate(pte_ray):
        prop_su = prop_su.condition(pte, tsu)

        rho_oil = prop_su.oil.density  # oil density
        yoil, ywat, ygas = prop_su.volm_fract()
        qoil, qwat, qgas = actual_flow(qoil_std, rho_oil_std, rho_oil, yoil, ywat, ygas)
        qtot = qoil + qwat + qgas

        vel_ray[i] = qtot / ate
        rho_ray[i] = prop_su.pmix()
        snd_ray[i] = prop_su.cmix()

    return qoil_std, pte_ray, rho_ray, vel_ray, snd_ray


def energy_arrays(ken, pte_ray, rho_ray, vel_ray):
    """Specific Energy Arrays for Throat Entry

    Calculate the reservoir fluid kinetic energy and expansion energy. Return
    arrays that can be graphed for visualization.

    Args:
        ken (float): Nozzle Enterance Friction Loss, unitless
        pte_ray (np array): Press Throat Entry Array, psig
        rho_ray (np array): Density Throat Entry Array, lbm/ft3
        vel_ray (np array): Velocity Throat Entry Array, ft/s

    Returns:
        ke_ray (np array): Kinetic Energy, ft2/s2
        ee_ray (np array): Expansion Energy, ft2/s2
    """

    # convert from psi to lbm/(ft*s2)
    plbm = pte_ray * 144 * 32.174
    ee_ray = cumtrapz(1 / rho_ray, plbm, initial=0)  # ft2/s2 expansion energy
    ke_ray = (1 + ken) * (vel_ray**2) / 2  # ft2/s2 kinetic energy
    return ke_ray, ee_ray


def throat_entry_graphs(pte_ray, rho_ray, vel_ray, snd_ray) -> None:
    """Throat Entry Graphs

    Create a graph to visualize what is occuring inside the throat entry section

    Args:
        pte_ray (np array): Press Throat Entry Array, psig
        rho_ray (np array): Density Throat Entry Array, lbm/ft3
        vel_ray (np array): Velocity Throat Entry Array, ft/s
        snd_ray (np array): Speed of Sound Array, ft/s

    Returns:
        Graphs of Specific Volume, Velocity, Specific Energy, and TEE
    """
    mach_ray = vel_ray / snd_ray
    pmo = np.interp(1, mach_ray, pte_ray)  # interpolate for pressure at mach 1, pmo
    # fix this later for entrance friction loss
    kse_ray, ese_ray = energy_arrays(0.01, pte_ray, rho_ray, vel_ray)
    tee_ray = kse_ray + ese_ray
    psuc = pte_ray[0]

    fig, axs = plt.subplots(4, sharex=True)
    fig.suptitle(f"Suction at {round(psuc,0)} psi, Mach 1 at {round(pmo,0)} psi")

    axs[0].scatter(pte_ray, 1 / rho_ray)
    axs[0].set_ylabel("Specific Volume, ft3/lbm")

    axs[1].scatter(pte_ray, vel_ray, label="Mixture Velocity")
    axs[1].scatter(pte_ray, snd_ray, label="Speed of Sound")
    axs[1].set_ylabel("Velocity, ft/s")
    axs[1].legend()

    axs[2].scatter(pte_ray, ese_ray, label="Expansion")
    axs[2].scatter(pte_ray, kse_ray, label="Kinetic")
    axs[2].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axs[2].set_ylabel("Specific Energy, ft2/s2")
    axs[2].legend()

    ycoord = (max(tee_ray) + min(tee_ray)) / 2
    axs[3].scatter(pte_ray, tee_ray)
    axs[3].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axs[3].axvline(x=pmo, color="black", linestyle="--", linewidth=1)
    axs[3].annotate(text="Mach 1", xy=(pmo, ycoord), rotation=90)
    axs[3].set_ylabel("TEE, ft2/s2")
    axs[3].set_xlabel("Throat Entry Pressure, psig")
    plt.show()
    return None


@dataclass
class throat_entry_result:
    """Class for storing throat entry equation results, to be graphed later.

    Args:
        psu (float): Suction Pressure, psig
        qsu (float): Suction Oil Flow, bopd
        pmo (float): Mach One Pressure, psig
        pte_ray (np.ndarray): Throat Entry Array, psig
        tee_ray (np.ndarray): Throat Entry Eqn Array, ft2/s2
    """

    psu: float
    qsu: float
    pmo: float
    pte_ray: np.ndarray
    tee_ray: np.ndarray


def multi_throat_entry_arrays(
    psu_min: float, psu_max: float, tsu: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> list:
    """Multiple Throat Entry Arrays

    Calculate throat entry arrays at different suction pressures. Used to
    graph later. Similiar to Figure 5 from Robert Merrill Paper.

    Args:
        psu_min (float): Suction Pressure Min, psig
        psu_max (float): Suction Pressure Max, psig, less than reservoir pressure
        tsu (float): Suction Temp, deg F
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        res_list (list): List of throat_entry_result class at various suction pressures
    """
    if psu_max >= ipr_su.pres:
        raise ValueError("Max suction pressure must be less than reservoir pressure")
    # 200 is arbitary and has been hard coded into the throat entry array calcs
    if psu_min <= 200:
        raise ValueError("Min suction pressure must be greater than 200 psig")

    ray_len = 5  # number of elements in the array

    # create empty list to fill up with results
    res_lis = list()

    psu_ray = np.linspace(psu_min, psu_max, ray_len)

    for psu in psu_ray:
        qoil_std, pte_ray, rho_ray, vel_ray, snd_ray = throat_entry_arrays(psu, tsu, ate, ipr_su, prop_su)
        pmo = throat_entry_mach_one(pte_ray, vel_ray, snd_ray)
        kse_ray, ese_ray = energy_arrays(0.01, pte_ray, rho_ray, vel_ray)
        tee_ray = kse_ray + ese_ray

        # story the results as a class
        res_lis.append(
            throat_entry_result(
                psu=psu,
                qsu=qoil_std,
                pmo=pmo,
                pte_ray=pte_ray,  # drop suction pressure
                tee_ray=tee_ray,
            )
        )
    return res_lis


def multi_suction_graphs(res_lis: list) -> None:
    """Throat Entry Graphs for Multiple Suction Pressures

    Create a graph that shows throat entry equation solutions for multiple suction pressures

    Args:
        res_lis (list): List of throat_entry_result class at various suction pressures

    Returns:
        Graphs
    """
    ax = plt.gca()
    for res in res_lis:
        tee_pmo = np.interp(res.pmo, np.flip(res.pte_ray), np.flip(res.tee_ray))
        color = next(ax._get_lines.prop_cycler)["color"]
        # only graph values where the mach number is under one
        pte_ray = res.pte_ray[res.pte_ray >= res.pmo]
        tee_ray = res.tee_ray[res.pte_ray >= res.pmo]
        plt.scatter(pte_ray, tee_ray, color=color, label=f"{int(res.qsu)} bopd, {int(res.psu)} psi")
        plt.scatter(res.pmo, tee_pmo, marker="v", color=color)  # mach equals one

    plt.xlabel("Throat Entry Pressure, psig")
    plt.ylabel("Throat Entry Equation, ft2/s2")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.title("Figure 5 of SPE-202928-MS, Mach 1 at \u25BC")
    plt.legend()
    plt.show()
    return None


def tee_near_pmo(psu: float, tsu: float, ate: float, ipr_su: InFlow, prop_su: ResMix) -> float:
    """Throat Entry Equation near pmo, mach 1 pressure

    Find the value of the throat entry equation near the mach 1 pressure. The following
    function will be iterated across to minimize the throat entry equation.

    Args:
        psu (float): Suction Press, psig
        tsu (float): Suction Temp, deg F
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
    kse_ray, ese_ray = energy_arrays(0.01, pte_ray[mask], rho_ray[mask], vel_ray[mask])
    tee_ray = kse_ray + ese_ray
    tee_pmo = min(tee_ray)  # find the smallest value of tee where mach <=1
    return tee_pmo


def minimize_tee(tsu: float, ate: float, ipr_su: InFlow, prop_su: ResMix) -> float:
    """Minimize Throat Entry Equation at pmo

    Find that psu that minimizes the throat entry equation for where Mach = 1 (pmo).
    Secant method for iteration, starting point is Res Pres minus 200 and 300 psig.
    Boundary equation is the starting point that Bob Merrill uses in his paper.

    Args:
        tsu (float): Suction Temp, deg F
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        psu (float): Suction Pressure, psig"""
    # how can we guarentee mach values are reached???
    psu_list = [ipr_su.pres - 300, ipr_su.pres - 400]
    # store values of tee near mach=1 pressure
    tee_list = [
        tee_near_pmo(psu_list[0], tsu, ate, ipr_su, prop_su),
        tee_near_pmo(psu_list[1], tsu, ate, ipr_su, prop_su),
    ]
    # criteria for when you've converged to an answer
    psu_diff = 5
    n = 0  # loop counter
    while abs(psu_list[-2] - psu_list[-1]) > psu_diff:
        # use secant method to calculate next guess value for psu to use
        psu_nxt = psu_list[-1] - tee_list[-1] * (psu_list[-2] - psu_list[-1]) / (tee_list[-2] - tee_list[-1])
        tee_nxt = tee_near_pmo(psu_nxt, tsu, ate, ipr_su, prop_su)
        psu_list.append(psu_nxt)
        tee_list.append(tee_nxt)
        n = n + 1
        if n == 10:
            print("TEE Minimization did not converge")
            break
    # print(psu_list)
    # print(tee_list)
    return psu_list[-1]


def zero_tee(
    pte_ray: np.ndarray, rho_ray: np.ndarray, vel_ray: np.ndarray, snd_ray: np.ndarray
) -> tuple[float, float, float]:
    """Throat Entry Parameters with a zero TEE

    Calculate the throat entry pressure, density, and velocity where TEE crosses zero.
    Valid for one suction pressure  of the pump / reservoir.

    Args:
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
    kse_ray, ese_ray = energy_arrays(0.01, pte_ray[mask], rho_ray[mask], vel_ray[mask])
    tee_ray = kse_ray + ese_ray
    # is there a way to speed up all these interpolations?
    pte = np.interp(0, np.flip(tee_ray), np.flip(pte_ray[mask]))
    rho_te = np.interp(0, np.flip(tee_ray), np.flip(rho_ray[mask]))
    vte = np.interp(0, np.flip(tee_ray), np.flip(vel_ray[mask]))
    return pte, rho_te, vte


def pf_press_depth(fld_dens: float, prs_surf: float, pump_tvd: float) -> float:
    """Power Fluid Pressure at Depth

    Calculate the Power Fluid Pressure at Depth.

    Args:
        fld_dens (float): Density of Fluid, lbm/ft3
        prs_surf (float): Power Fluid Surface Pressure, psig
        pump_tvd (float): Pump True Vertical Depth, feet

    Returns:
        prs_dpth (float): Power Fluid Depth Pressure, psig
    """
    prs_dpth = prs_surf + fld_dens * pump_tvd / 144
    return prs_dpth


def nozzle_velocity(pni: float, pte: float, knz: float, rho_nz: float) -> float:
    """Nozzle Velocity

    Solve Bernoulli's Equation to calculate the nozzle velocity in ft/s.

    Args:
        pni (float): Nozzle Inlet Pressure, psig
        pte (float): Throat Entry Pressure, psig
        knz (float): Friction of Nozzle, unitless
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3

    Returns:
        vnz (float): Nozzle Velocity, ft/s
    """
    vnz = math.sqrt(2 * 32.2 * 144 * (pni - pte) / (rho_nz * (1 + knz)))
    return vnz


def nozzle_rate(vnz: float, anz: float) -> tuple[float, float]:
    """Nozzle Flow Rate

    Find Nozzle / Power Fluid Flowrate in ft3/s and BPD

    Args:
        vnz (float): Nozzle Velocity, ft/s
        anz (float): Area of Nozzle, ft2

    Returns:
        qnz_ft3s (float): Nozzle Flowrate ft3/s
        qnz_bpd (float): Nozzle Flowrate bpd
    """
    qnz_ft3s = anz * vnz
    qnz_bpd = qnz_ft3s * (7.4801 * 60 * 60 * 24 / 42)
    return qnz_ft3s, qnz_bpd


def throat_diff(kth: float, vnz: float, anz: float, rho_nz: float, vte: float, ate: float, rho_te: float):
    """Throat Differential Pressure

    Solves the throat mixture equation of the jet pump. Calculates throat differntial pressure.
    Use the throat entry pressure and differential pressure to calculate throat mix pressure.
    ptm = pte - dp_th

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
    return dp_tm


def throat_wc(qoil_std: float, wc_su: float, qwat_nz: float) -> float:
    """Throat Watercut

    Calculate watercut inside jet pump throat. This is the new watercut
    after the power fluid and reservoir fluid have mixed together

    Args:
        qoil_std (float): Oil Rate, STD BOPD
        wc_su (float): Watercut at pump suction, decimal
        qwat_nz (float): Powerfluid Flowrate, BWPD

    Returns:
        wc_tm (float): Watercut at throat, decimal"""

    qwat_su = qoil_std * wc_su / (1 - wc_su)
    qwat_tot = qwat_nz + qwat_su
    wc_tm = qwat_tot / (qwat_tot + qoil_std)
    return wc_tm


area_te = (e42jetpump.athr - e42jetpump.anoz) / 144
# psuc = 880

# res_lis = multi_throat_entry_arrays(psu_min=876, psu_max=1100, tsu=82, ate=area_te, ipr_su=ipr, prop_su=e42)
# multi_suction_graphs(res_lis)

psu_min = minimize_tee(80, area_te, ipr, e42)
qsu_std, pte_ray, rho_ray, vel_ray, snd_ray = throat_entry_arrays(psu_min, 80, area_te, ipr, e42)
# need to return pte, rho_te, vte
pte, rho_te, vte = zero_tee(pte_ray, rho_ray, vel_ray, snd_ray)
pni = pf_press_depth(62.4, 3000, 4000)
vnz = nozzle_velocity(pni, pte, 0.01, 62.4)
anz = e42jetpump.anoz / 144
qnz_ft3s, qnz_bpd = nozzle_rate(vnz, anz)
dp_tm = throat_diff(0.1, vnz, anz, 62.4, vte, area_te, rho_te)
ptm = pte - dp_tm
wc_tm = throat_wc(qsu_std, e42.wc, qnz_bpd)
print(e42.wc, wc_tm)


# note, suction oil flow rate is pulled out and can be used later
# if wanted for the visualization
# throat_entry_graphs(press, dens, velo, sound)


"""
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
    """
