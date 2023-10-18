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

"""This file is nothing more than a collection of Kaelin trying to make
some jet pump theory work in a practical method. At this point I don't
really like Bobs write up on pto vs pte and I will attempt my own method.
I'm going to just make static functions in an attempt to make this easier
"""

"""
Needs the following:
IPR, Reservoir Fluid with WC / GOR, Jet Pump with dimensions of nozzle/throat.
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


def throat_entry(jpump: JetPump, fres: ResMix, qres: InFlow, tres: float):
    """Throat Entry Equation

    Solve the throat entry equation of the jet pump. Solves for the throat entry pressure.
    Uses the wells IPR to bound the suction pressure and rate.

    Args:
        jpump (JetPump): JetPump inside the well
        fres (ResMix): Flow of the Reservoir
        qres (InFlow): IPR of the Reservoir
        tres (float): Temperature of the Reservoir, deg F

    Returns:
        pte (float): ....this really should be a table/tablezzz?
    """
    psuct = np.linspace(200, qres.pres, 3)  # suction pressure, from 200 psi to res pressure
    psuct = np.flip(psuct, axis=0)  # flip to go from high to low pressure

    poil_std = fres.oil.condition(0, 60).density  # oil density
    ate = (jpump.athr - jpump.anoz) / 144  # area of the throat entry

    for ps in psuct:
        ps = 900
        qs = qres.oil_flow(ps, method="pidx")  # use prod index to get oil flow
        # need a method where you input oil flow in bopd, returns
        pmix = fres.condition(ps, tres).pmix()
        cmix = fres.cmix()

        # find the total flowrate
        poil = fres.oil.condition(ps, tres).density  # oil density
        yoil, ywat, ygas = fres.volm_fract()
        # really I just need the mixture flowrate / mixture density
        qoil, qwat, qgas = actual_flow(qs, poil_std, poil, yoil, ywat, ygas)
        qtot = qoil + qwat + qgas
        vte = qtot / ate  # velocity of the throat entry
        # need to convert the pressure over to lbf/ft2 during integral eval

        # first value will be at psuction
        prs_ray = np.array(ps)
        rho_ray = np.array(pmix)
        vel_ray = np.array(vte)
        snd_ray = np.array(cmix)

        # create an array of throat entry pressures
        pentry = np.linspace(200, ps, 20)
        pentry = np.flip(pentry, axis=0)

        for i, pen in enumerate(pentry):
            pmix = fres.condition(pen, tres).pmix()
            cmix = fres.cmix()

            poil = fres.oil.condition(ps, tres).density  # oil density
            yoil, ywat, ygas = fres.volm_fract()
            # really I just need the mixture flowrate / mixture density
            qoil, qwat, qgas = actual_flow(qs, poil_std, poil, yoil, ywat, ygas)
            qtot = qoil + qwat + qgas
            vte = qtot / ate  # velocity of the throat entry
            # need to convert the pressure over to lbf/ft2 during integral eval

            prs_ray = np.append(prs_ray, pen)
            rho_ray = np.append(rho_ray, pmix)
            vel_ray = np.append(vel_ray, vte)
            snd_ray = np.append(snd_ray, cmix)

    return prs_ray, rho_ray, vel_ray, snd_ray, ps, qs


press, dens, velo, sound, ps, qs = throat_entry(e42jetpump, e42, ipr, 80)

# print(press, dens, velo)
# find the area under the density curve from ps to 400 psig
# need to convert pressure to pounds/square foot!

plbm = press * 144 * 32.174  # convert from psi to lbm/(ft*s2)

# boom baby, definitely a cross over, at least in one place...
# if no cross over, your suction pressure is probably too small? not enough flow?

from scipy.integrate import cumtrapz, trapz

# I_trapz = np.absolute(trapz(1 / dens, plbm))  # ft2/s2
I_cumtrapz = np.abs(cumtrapz(1 / dens, plbm))  # ft2/s2
# print(I_cumtrapz)

ken = 0.01  # enterance friction factor

kinen = (1 + 0.01) * (velo**2) / 2  # solve for kinetic energy


fig, axs = plt.subplots(3, sharex=True)
fig.suptitle(f"Suction at {ps} psi and {round(qs,0)} bopd")
axs[0].scatter(press, 1 / dens)
axs[0].set_ylabel("Specific Volume, ft3/lbm")
axs[1].scatter(press, velo, label="Mixture Velocity")
axs[1].scatter(press, sound, label="Speed of Sound")
axs[1].set_ylabel("Velocity, ft/s")
axs[1].legend()
axs[2].scatter(press[1:], I_cumtrapz, label="Pressure")
axs[2].set_ylabel("Specific Energy, ft2/s2")
axs[2].scatter(press, kinen, label="Kinetic")
axs[2].set_xlabel("Throat Entry Pressure, PSI")
axs[2].legend()
plt.show()
