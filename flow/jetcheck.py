from flow import jetflow as jf
from flow import jetplot as jplt
from flow import outflow as of
from flow import singlephase as sp
from flow.inflow import InFlow
from geometry.jetpump import JetPump
from geometry.pipe import Annulus, Pipe
from geometry.wellprofile import WellProfile
from pvt.resmix import ResMix


# writing a function that can be ran to easily compare current jet pump
# performance to expected performance
def jet_check(
    form_temp: float,
    jpump_tvd: float,
    rho_pf: float,
    ppf_surf: float,
    jpump_well: JetPump,
    tube: Pipe,
    ipr_well: InFlow,
    prop_well: ResMix,
) -> None:
    psu_min, qsu_std, pte, rho_te, vte = jf.tee_minimize(
        tsu=form_temp, ken=jpump_well.ken, ate=jpump_well.ate, ipr_su=ipr_well, prop_su=prop_well
    )
    # pni = jf.pf_press_depth(rho_pf, ppf_surf, jpump_tvd)
    pni = ppf_surf + sp.diff_press_static(rho_pf, jpump_tvd)
    vnz = jf.nozzle_velocity(pni, pte, jpump_well.knz, rho_pf)

    qnz_ft3s, qnz_bpd = jf.nozzle_rate(vnz, jpump_well.anz)
    wc_tm = jf.throat_wc(qsu_std, prop_well.wc, qnz_bpd)

    prop_tm = ResMix(wc_tm, prop_well.fgor, prop_well.oil, prop_well.wat, prop_well.gas)
    ptm = jf.throat_discharge(
        pte, form_temp, jpump_well.kth, vnz, jpump_well.anz, rho_pf, vte, jpump_well.ate, rho_te, prop_tm
    )
    vtm, pdi = jf.diffuser_discharge(ptm, form_temp, jpump_well.kdi, jpump_well.ath, tube.inn_area, qsu_std, prop_tm)

    print(f"Suction Pressure: {round(psu_min, 1)} psig")
    print(f"Oil Flow: {round(qsu_std, 1)} bopd")
    print(f"Nozzle Inlet Pressure: {round(pni, 1)} psig")
    print(f"Throat Entry Pressure: {round(pte, 1)} psig")
    print(f"Throat Discharge Pressure: {round(ptm, 1)} psig")
    print(f"Diffuser Discharge Pressure: {round(pdi, 1)} psig")
    print(f"Power Fluid Rate: {round(qnz_bpd, 1)} bwpd")
    print(f"Nozzle Velocity: {round(vnz, 1)} ft/s")
    print(f"Throat Entry Velocity: {round(vte, 1)} ft/s")

    # graphing some outputs for visualization
    qsu_std, pte_ray, rho_ray, vel_ray, snd_ray = jplt.throat_entry_arrays(
        psu_min, form_temp, jpump_well.ate, ipr_well, prop_well
    )
    jplt.throat_entry_graphs(jpump_well.ken, pte_ray, rho_ray, vel_ray, snd_ray)

    vtm, pdi_ray, rho_ray, vdi_ray, snd_ray = jplt.diffuser_arrays(
        ptm, form_temp, jpump_well.ath, tube.inn_area, qsu_std, prop_tm
    )
    jplt.diffuser_graphs(vtm, jpump_well.kdi, pdi_ray, rho_ray, vdi_ray, snd_ray)


# writing a function that can be ran to easily compare current jet pump
# performance to expected performance
def jet_check_two(
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
    """Jet Check Two

    Brings in the outflow node of the jet pump system, looking at the required discharge pressure
    at the wells flowrate vs what the pump can actually supply.
    """
    psu_min, qsu_std, pte, rho_te, vte = jf.tee_minimize(
        tsu=form_temp, ken=jpump_well.ken, ate=jpump_well.ate, ipr_su=ipr_well, prop_su=prop_well
    )
    pni = ppf_surf + sp.diff_press_static(rho_pf, wellprof.jetpump_vd)
    vnz = jf.nozzle_velocity(pni, pte, jpump_well.knz, rho_pf)

    qnz_ft3s, qnz_bpd = jf.nozzle_rate(vnz, jpump_well.anz)
    wc_tm = jf.throat_wc(qsu_std, prop_well.wc, qnz_bpd)

    prop_tm = ResMix(wc_tm, prop_well.fgor, prop_well.oil, prop_well.wat, prop_well.gas)
    ptm = jf.throat_discharge(
        pte, form_temp, jpump_well.kth, vnz, jpump_well.anz, rho_pf, vte, jpump_well.ate, rho_te, prop_tm
    )
    vtm, pdi = jf.diffuser_discharge(ptm, form_temp, jpump_well.kdi, jpump_well.ath, tube.inn_area, qsu_std, prop_tm)

    md_seg, prs_ray, slh_ray = of.top_down_press(surf_pres, form_temp, qsu_std, prop_tm, tube, wellprof)

    print(f"Suction Pressure: {round(psu_min, 1)} psig")
    print(f"Oil Flow: {round(qsu_std, 1)} bopd")
    print(f"Nozzle Inlet Pressure: {round(pni, 1)} psig")
    print(f"Throat Entry Pressure: {round(pte, 1)} psig")
    print(f"Throat Discharge Pressure: {round(ptm, 1)} psig")
    print(f"Required Diffuser Discharge Pressure: {round(prs_ray[-1], 1)} psig")
    print(f"Supplied Diffuser Discharge Pressure: {round(pdi, 1)} psig")
    print(f"Power Fluid Rate: {round(qnz_bpd, 1)} bwpd")
    print(f"Nozzle Velocity: {round(vnz, 1)} ft/s")
    print(f"Throat Entry Velocity: {round(vte, 1)} ft/s")

    # graphing some outputs for visualization
    qsu_std, pte_ray, rho_ray, vel_ray, snd_ray = jplt.throat_entry_arrays(
        psu_min, form_temp, jpump_well.ate, ipr_well, prop_well
    )
    jplt.throat_entry_graphs(jpump_well.ken, pte_ray, rho_ray, vel_ray, snd_ray)

    vtm, pdi_ray, rho_ray, vdi_ray, snd_ray = jplt.diffuser_arrays(
        ptm, form_temp, jpump_well.ath, tube.inn_area, qsu_std, prop_tm
    )
    jplt.diffuser_graphs(vtm, jpump_well.kdi, pdi_ray, rho_ray, vdi_ray, snd_ray)
