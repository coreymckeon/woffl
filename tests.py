import math

import numpy as np
from flow.inflow import InFlow
from geometry.pipe import Annulus, Pipe
from jetpump import JetPump
from matplotlib import pyplot as plt
from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater
from pvt.resmix import ResMix
from scipy.integrate import cumtrapz, trapz

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
        poil_std (float): Density of Oil at Std Cond, lbm/ft3
        yoil (float): Volm Fraction of oil, ft3/ft3
        ywat (float): Volm Fraction of water, ft3/ft3
        ygas (float): Volm Fraction of gas, ft3/ft3

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


def throat_entry_arrays(psu: float, tsu: float, ate: float, ipr_su: InFlow, prop_su: ResMix):
    """Throat Entry Arrays

    Create a series of throat entry arrays. The arrays can be graphed to visualize.
    What is occuring inside the throat entry while pressure is dropped. Doesn't
    actually find the solutions, just for visualization.

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

    ray_len = 20  # number of elements in the array

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
    arrays that can be graphed for visualization. Returned energy arrays will
    be one shorter than the provided pressure, density and velocity arrays.

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
    ee_ray = cumtrapz(1 / rho_ray, plbm)  # ft2/s2 expansion energy
    ke_ray = (1 + ken) * (vel_ray**2) / 2  # ft2/s2 kinetic energy
    # integration starts at psu, remove that value as it has little meaning
    ke_ray = ke_ray[1:]  # chop off kinetic energy at psu

    return ke_ray, ee_ray


def throat_entry_graphs(pte_ray, rho_ray, vel_ray, snd_ray):
    """Throat Entry Graphs

    Create a graph to visualize what is occuring inside the throat entry section

    Args:
        pte_ray (np array): Press Throat Entry Array, psig
        rho_ray (np array): Density Throat Entry Array, lbm/ft3
        vel_ray (np array): Velocity Throat Entry Array, ft/s
        snd_ray (np array): Speed of Sound Array, ft/s

    Returns:
        Graphs
    """
    mach_ray = vel_ray / snd_ray
    crit_prs = np.interp(1, mach_ray, pte_ray)  # interpolate for pressure at mach 1
    # fix this later for entrance friction loss
    kse_ray, ese_ray = energy_arrays(0.01, pte_ray, rho_ray, vel_ray)
    tee_ray = kse_ray + ese_ray
    psuc = pte_ray[0]

    fig, axs = plt.subplots(4, sharex=True)
    fig.suptitle(f"Suction at {psuc} psi, Mach 1 at {round(crit_prs,0)} psi")

    axs[0].scatter(pte_ray, 1 / rho_ray)
    axs[0].set_ylabel("Specific Volume, ft3/lbm")

    axs[1].scatter(pte_ray, vel_ray, label="Mixture Velocity")
    axs[1].scatter(pte_ray, snd_ray, label="Speed of Sound")
    axs[1].set_ylabel("Velocity, ft/s")
    axs[1].legend()

    axs[2].scatter(pte_ray[1:], ese_ray, label="Expansion")
    axs[2].scatter(pte_ray[1:], kse_ray, label="Kinetic")
    axs[2].set_ylabel("Specific Energy, ft2/s2")
    axs[2].legend()

    ycoord = max(tee_ray) * 0.25
    axs[3].scatter(pte_ray[1:], tee_ray)
    axs[3].hlines(y=0, xmin=min(pte_ray[1:]), xmax=max(pte_ray[1:]), colors="red", linestyles="dashed")
    axs[3].axvline(x=crit_prs, ymin=0, ymax=1, color="red", linestyle="dashed")
    axs[3].annotate(text="Mach 1", xy=(crit_prs, ycoord), rotation=90)
    axs[3].set_ylabel("TEE, ft2/s2")
    axs[3].set_xlabel("Throat Entry Pressure, psig")
    plt.show()
    return None


area_te = (e42jetpump.athr - e42jetpump.anoz) / 144
psuc = 900

qsu_std, press, dens, velo, sound = throat_entry_arrays(psuc, 80, area_te, ipr, e42)
throat_entry_graphs(press, dens, velo, sound)


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


def throat_mix(pte: float, kth: float, vnz: float, anz: float, rho_nz: float, vte: float, ate: float, rho_te: float):
    """Throat Mixture Equation

    Solve the throat mixture equation of the jet pump. Calculates throat discharge pressure.
    Mixture Velocity, Mixture Density and other parameters.

    Args:
        pte (float): Pressure Throat Entry, psig
        kth (float): Friction of Throat Mix, Unitless
        vnz (float): Velocity of Nozzle, ft/s
        anz (float): Area of Nozzle, ft2
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3
        vte (float): Velocity of Throat Entry Mixture, ft/s
        ate (float): Area of Throat Entry, ft2
        rho_te (float): Density of Throat Entry Mixture, lbm/ft3

    Returns:
        ptm (float): Pressure Throat Mixture. psig
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
    dp = 0.5 * kth * rho_tm * vtm**2 + mtm * vtm / ath - mnz * vnz / ath - mte * vte / ath
    # convert to lbf/in2
    dp = dp / (32.174 * 144)

    ptm = pte - dp

    return ptm
