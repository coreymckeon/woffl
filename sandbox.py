from flow import jetflow as jf
from flow import jetplot as jplt
from flow import singlephase as sp
from flow.inflow import InFlow
from geometry.jetpump import JetPump
from geometry.pipe import Annulus, Pipe
from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater
from pvt.resmix import ResMix

jpump_tvd = 4050  # feet, interpolated off well profile
rho_pf = 62.4  # lbm/ft3
ppf_surf = 3175  # psi, power fluid surf pressure

# testing the jet pump code on E-42
tube = Pipe(out_dia=4.5, thick=0.5)  # E-42 tubing
case = Pipe(out_dia=6.875, thick=0.5)  # E-42 casing
ann = Annulus(inn_pipe=tube, out_pipe=case)  # define the annulus

e42_ipr = InFlow(qwf=188, pwf=800, pres=1400)  # define an ipr

e42_jp = JetPump(nozzle_no="11", area_ratio="C")

mpu_oil = BlackOil.schrader_oil()  # class method
mpu_wat = FormWater.schrader_wat()  # class method
mpu_gas = FormGas.schrader_gas()  # class method

form_wc = 0.86
form_gor = 2300  # formation gor
form_temp = 80
e42_res = ResMix(wc=form_wc, fgor=form_gor, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)


psu_min, qsu_std, pte, rho_te, vte = jf.tee_minimize(
    tsu=form_temp, ken=e42_jp.ken, ate=e42_jp.ate, ipr_su=e42_ipr, prop_su=e42_res
)
qsu_std, pte_ray, rho_ray, vel_ray, snd_ray = jplt.throat_entry_arrays(psu_min, form_temp, e42_jp.ate, e42_ipr, e42_res)
jplt.throat_entry_graphs(e42_jp.ken, pte_ray, rho_ray, vel_ray, snd_ray)
# pte, rho_te, vte = jf.cross_zero_tee(e42_jp.ken, pte_ray, rho_ray, vel_ray, snd_ray)
# pni = jf.pf_press_depth(rho_pf, ppf_surf, jpump_tvd)
pni = ppf_surf + sp.diff_press_static(rho_pf, jpump_tvd)
vnz = jf.nozzle_velocity(pni, pte, e42_jp.knz, rho_pf)
# note, vtm from throat equation and vtm from diffuser equation will not be equal
# this is because throat equation is run at 400 psig, diffuser is run at 1100 psig
# dp_tm, vtm = jf.throat_dp(e42_jp.kth, vnz, e42_jp.anz, rho_pf, vte, e42_jp.ate, rho_te)

qnz_ft3s, qnz_bpd = jf.nozzle_rate(vnz, e42_jp.anz)
wc_tm = jf.throat_wc(qsu_std, e42_res.wc, qnz_bpd)
# redefine the ResMixture with the additional waterlift power fluid
e42_disch = ResMix(wc_tm, form_gor, mpu_oil, mpu_wat, mpu_gas)
ptm_new = jf.throat_discharge(pte, form_temp, 0.41, vnz, e42_jp.anz, rho_pf, vte, e42_jp.ate, rho_te, e42_disch)
ath = e42_jp.ath
adi = tube.inn_area
# ptm = pte - dp_tm
# print(f"Old Throat Discharge: {round(ptm, 1)} psi, New Throat Discharge: {round(ptm_new, 1)} psi")
vtm, pdi_ray, rho_ray, vdi_ray, snd_ray = jplt.diffuser_arrays(ptm_new, form_temp, ath, adi, qsu_std, e42_disch)
jplt.diffuser_graphs(vtm, e42_jp.kdi, pdi_ray, rho_ray, vdi_ray, snd_ray)
ohh, mama = jf.diffuser_discharge(ptm_new, form_temp, 0.35, ath, adi, qsu_std, e42_disch)
# kde_ray, ede_ray = diffuser_energy(0.1, vtm, pdi_ray, rho_ray, vdi_ray)
# print(vtm, pdi_ray, rho_ray, vdi_ray, snd_ray)
# print(f"Throat increase in pressure is {round(dp_tm, 0)} psi")
# print(vtm, ohh, mama)
print(qnz_bpd)
