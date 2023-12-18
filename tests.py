from flow import jetflow as jf
from flow.inflow import InFlow
from geometry.jetpump import JetPump
from geometry.pipe import Annulus, Pipe
from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater
from pvt.resmix import ResMix

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

e42_ipr = InFlow(qwf=350, pwf=800, pres=1400)  # define an ipr

e42_jp = JetPump("13", "B")

mpu_oil = BlackOil.schrader_oil()  # class method
mpu_wat = FormWater.schrader_wat()  # class method
mpu_gas = FormGas.schrader_gas()  # class method

e42_res = ResMix(0.75, 1500, mpu_oil, mpu_wat, mpu_gas)

temp = 80
press = 700

# res_lis = multi_throat_entry_arrays(psu_min=876, psu_max=1100, tsu=82, ate=area_te, ipr_su=ipr, prop_su=e42)
# multi_suction_graphs(res_lis)

psu_min = jf.minimize_tee(80, e42_jp.ate, e42_ipr, e42_res)
qsu_std, pte_ray, rho_ray, vel_ray, snd_ray = jf.throat_entry_arrays(psu_min, 80, e42_jp.ate, e42_ipr, e42_res)
# need to return pte, rho_te, vte
pte, rho_te, vte = jf.zero_tee(pte_ray, rho_ray, vel_ray, snd_ray)
pni = jf.pf_press_depth(62.4, 3000, 4000)
vnz = jf.nozzle_velocity(pni, pte, 0.01, 62.4)
qnz_ft3s, qnz_bpd = jf.nozzle_rate(vnz, e42_jp.anz)
dp_tm, vtm = jf.throat_dp(0.1, vnz, e42_jp.anz, 62.4, vte, e42_jp.ate, rho_te)
# note, vtm from throat equation and vtm from diffuser equation will not be equal
# this is because throat equation is run at 400 psig, diffuser is run at 1100 psig
wc_tm = jf.throat_wc(qsu_std, e42_res.wc, qnz_bpd)
# redefine the ResMixture with the additional waterlift power fluid
e42_disch = ResMix(wc_tm, 1500, mpu_oil, mpu_wat, mpu_gas)
ath = e42_jp.ath
adi = tube.inn_area
ptm = pte - dp_tm
vtm, pdi_ray, rho_ray, vdi_ray, snd_ray = jf.diffuser_arrays(ptm, 80, ath, adi, qsu_std, e42_disch)
jf.diffuser_graphs(vtm, pdi_ray, rho_ray, vdi_ray, snd_ray)
# kde_ray, ede_ray = diffuser_energy(0.1, vtm, pdi_ray, rho_ray, vdi_ray)
# print(vtm, pdi_ray, rho_ray, vdi_ray, snd_ray)
