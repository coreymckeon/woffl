import numpy as np

from woffl.flow import jetflow as jf
from woffl.flow import jetplot as jplt
from woffl.flow import outflow as of
from woffl.flow import singlephase as sp
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Annulus, Pipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.resmix import ResMix


def suction_check(
    form_temp: float,
    jpump_tvd: float,
    rho_pf: float,
    ppf_surf: float,
    jpump_well: JetPump,
    tube: Pipe,
    ipr_well: InFlow,
    prop_well: ResMix,
) -> None:
    """Suction Check Choked Conditions

    The following function looks at the operating conditions of a jet pump with a choked throat
    entry. The function does no comparison of how the pump will behave based on discharge conditions
    or any other constraints. Prints results as text, creates plots of the jet pump throat entry
    and diffuser.

    Args:
        form_temp (float): Formation Temperature, deg F
        jpump_tvd (float): Jet Pump True Vertical Depth, feet
        rho_pf (float): Power Fluid Density, lbm/ft3
        ppf_surf (float): Pressure of Power Fluid at surface, psig
        jpump_well (JetPump): Jet Pump Class
        tube (Pipe): Pipe Class, used for diffuser diameter
        ipr_well (InFlow): IPR Class
        prop_well (ResMix): Reservoir conditions of the well

    Returns:
        None
    """
    psu_min, qoil_std, te_book = jf.psu_minimize(
        tsu=form_temp, ken=jpump_well.ken, ate=jpump_well.ate, ipr_su=ipr_well, prop_su=prop_well
    )
    pte, vte, rho_te, mach_te = te_book.dete_zero()

    pni = ppf_surf + sp.diff_press_static(rho_pf, jpump_tvd)
    vnz = jf.nozzle_velocity(pni, pte, jpump_well.knz, rho_pf)

    qnz_ft3s, qnz_bpd = jf.nozzle_rate(vnz, jpump_well.anz)
    wc_tm, qwat_su = jf.throat_wc(qoil_std, prop_well.wc, qnz_bpd)

    prop_tm = ResMix(wc_tm, prop_well.fgor, prop_well.oil, prop_well.wat, prop_well.gas)
    ptm = jf.throat_discharge(
        pte, form_temp, jpump_well.kth, vnz, jpump_well.anz, rho_pf, vte, jpump_well.ate, rho_te, prop_tm
    )
    vtm, pdi = jf.diffuser_discharge(ptm, form_temp, jpump_well.kdi, jpump_well.ath, tube.inn_area, qoil_std, prop_tm)

    print(f"Suction Pressure: {round(psu_min, 1)} psig")
    print(f"Oil Flow: {round(qoil_std, 1)} bopd")
    print(f"Nozzle Inlet Pressure: {round(pni, 1)} psig")
    print(f"Throat Entry Pressure: {round(pte, 1)} psig")
    print(f"Throat Discharge Pressure: {round(ptm, 1)} psig")
    print(f"Diffuser Discharge Pressure: {round(pdi, 1)} psig")
    print(f"Power Fluid Rate: {round(qnz_bpd, 1)} bwpd")
    print(f"Nozzle Velocity: {round(vnz, 1)} ft/s")
    print(f"Throat Entry Velocity: {round(vte, 1)} ft/s")

    qoil_std, te_book = jplt.throat_entry_book(psu_min, form_temp, jpump_well.ken, jpump_well.ate, ipr_well, prop_well)
    te_book.plot_te()

    vtm, di_book = jplt.diffuser_book(ptm, form_temp, jpump_well.ath, jpump_well.kdi, tube.inn_area, qoil_std, prop_tm)
    di_book.plot_di()


def discharge_check(
    surf_pres: float,
    form_temp: float,
    rho_pf: float,
    ppf_surf: float,
    jpump_well: JetPump,
    tube: Pipe,
    wellprof: WellProfile,
    ipr_well: InFlow,
    prop_well: ResMix,
) -> None:
    """Discharge Check Choked Conditions

    The following function compares what the jet pump can discharge compared to what the
    discharge pressure needs to be to lift the well. It only looks at choked conditions of
    the jet pump instead of iterating to find a zero residual. Prints results as text, creates
    plots of the jet pump throat entry and diffuser.

    Args:
        surf_pres (float): Well Head Surface Pressure, psig
        form_temp (float): Formation Temperature, deg F
        rho_pf (float): Power Fluid Density, lbm/ft3
        ppf_surf (float): Pressure of Power Fluid at surface, psig
        jpump_well (JetPump): Jet Pump Class
        tube (Pipe): Pipe Class
        wellprof (WellProfile): Well Profile Class
        ipr_well (InFlow): IPR Class
        prop_well (ResMix): Reservoir conditions of the well

    Returns:
        None
    """
    psu_min, qoil_std, te_book = jf.psu_minimize(
        tsu=form_temp, ken=jpump_well.ken, ate=jpump_well.ate, ipr_su=ipr_well, prop_su=prop_well
    )
    pte, vte, rho_te, mach_te = te_book.dete_zero()
    pni = ppf_surf + sp.diff_press_static(rho_pf, wellprof.jetpump_vd)
    vnz = jf.nozzle_velocity(pni, pte, jpump_well.knz, rho_pf)

    qnz_ft3s, qnz_bpd = jf.nozzle_rate(vnz, jpump_well.anz)
    wc_tm, qwat_su = jf.throat_wc(qoil_std, prop_well.wc, qnz_bpd)

    prop_tm = ResMix(wc_tm, prop_well.fgor, prop_well.oil, prop_well.wat, prop_well.gas)
    ptm = jf.throat_discharge(
        pte, form_temp, jpump_well.kth, vnz, jpump_well.anz, rho_pf, vte, jpump_well.ate, rho_te, prop_tm
    )
    vtm, pdi = jf.diffuser_discharge(ptm, form_temp, jpump_well.kdi, jpump_well.ath, tube.inn_area, qoil_std, prop_tm)

    md_seg, prs_ray, slh_ray = of.top_down_press(surf_pres, form_temp, qoil_std, prop_tm, tube, wellprof)

    outflow_pdi = prs_ray[-1]
    diff_pdi = pdi - outflow_pdi

    if diff_pdi >= 0:
        pdi_str = f"Well will flow choked, discharge pressure is {round(diff_pdi, 0)} psig greater than required"
    else:
        pdi_str = f"Well will NOT flow choked, discharge pressure is {round(diff_pdi, 0)} psig below the required"

    print(f"Suction Pressure: {round(psu_min, 1)} psig")
    print(f"Oil Flow: {round(qoil_std, 1)} bopd")
    print(f"Nozzle Inlet Pressure: {round(pni, 1)} psig")
    print(f"Throat Entry Pressure: {round(pte, 1)} psig")
    print(f"Throat Discharge Pressure: {round(ptm, 1)} psig")
    print(f"Required Diffuser Discharge Pressure: {round(prs_ray[-1], 1)} psig")
    print(f"Supplied Diffuser Discharge Pressure: {round(pdi, 1)} psig")
    print(pdi_str)
    print(f"Power Fluid Rate: {round(qnz_bpd, 1)} bwpd")
    print(f"Nozzle Velocity: {round(vnz, 1)} ft/s")
    print(f"Throat Entry Velocity: {round(vte, 1)} ft/s")

    # add the outflow, with the liquid holdup and pressure

    # graphing some outputs for visualization
    qsu_std, te_book = jplt.throat_entry_book(psu_min, form_temp, jpump_well.ken, jpump_well.ate, ipr_well, prop_well)
    te_book.plot_te()
    # print(te_book)
    vtm, di_book = jplt.diffuser_book(ptm, form_temp, jpump_well.ath, jpump_well.kdi, tube.inn_area, qsu_std, prop_tm)
    di_book.plot_di()
    # print(di_book)
    # te_book.plot()
