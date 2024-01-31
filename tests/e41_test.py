from flow import jetflow as jf
from flow import jetplot as jplt
from flow.inflow import InFlow
from geometry.jetpump import JetPump
from geometry.pipe import Annulus, Pipe
from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater
from pvt.resmix import ResMix

# data from MOU E-41 Well Test on 11/27/2023
# only works if the command python -m tests.e41_test is used

jpump_tvd = 4065  # feet, interpolated off well profile
rho_pf = 62.4  # lbm/ft3
ppf_surf = 3168  # psi, power fluid surf pressure

# testing the jet pump code on E-42
tube = Pipe(out_dia=4.5, thick=0.5)  # E-42 tubing
case = Pipe(out_dia=6.875, thick=0.5)  # E-42 casing
ann = Annulus(inn_pipe=tube, out_pipe=case)  # define the annulus

e41_ipr = InFlow(qwf=246, pwf=1049, pres=1400)  # define an ipr

e41_jp = JetPump(nozzle_no="13", area_ratio="A")

mpu_oil = BlackOil.schrader_oil()  # class method
mpu_wat = FormWater.schrader_wat()  # class method
mpu_gas = FormGas.schrader_gas()  # class method

form_wc = 0.894
form_gor = 600  # formation gor
form_temp = 111
e41_res = ResMix(wc=form_wc, fgor=form_gor, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)


psu_min, qsu_std, pte, rho_te, vte = jf.tee_minimize(
    tsu=form_temp, ken=e41_jp.ken, ate=e41_jp.ate, ipr_su=e41_ipr, prop_su=e41_res
)
qsu_std, pte_ray, rho_ray, vel_ray, snd_ray = jplt.throat_entry_arrays(psu_min, form_temp, e41_jp.ate, e41_ipr, e41_res)
jplt.throat_entry_graphs(e41_jp.ken, pte_ray, rho_ray, vel_ray, snd_ray)

pni = jf.pf_press_depth(rho_pf, ppf_surf, jpump_tvd)
vnz = jf.nozzle_velocity(pni, pte, e41_jp.knz, rho_pf)

qnz_ft3s, qnz_bpd = jf.nozzle_rate(vnz, e41_jp.anz)
wc_tm = jf.throat_wc(qsu_std, e41_res.wc, qnz_bpd)

e41_disch = ResMix(wc_tm, form_gor, mpu_oil, mpu_wat, mpu_gas)
ptm_new = jf.throat_discharge(pte, form_temp, 0.41, vnz, e41_jp.anz, rho_pf, vte, e41_jp.ate, rho_te, e41_disch)
ath = e41_jp.ath
adi = tube.inn_area

vtm, pdi_ray, rho_ray, vdi_ray, snd_ray = jplt.diffuser_arrays(ptm_new, form_temp, ath, adi, qsu_std, e41_disch)
jplt.diffuser_graphs(vtm, 0.3, pdi_ray, rho_ray, vdi_ray, snd_ray)
vtm, pdi = jf.diffuser_discharge(ptm_new, form_temp, 0.3, ath, adi, qsu_std, e41_disch)

print(f"Throat Entry Pressure {round(pte, 2)} psig")
print(f"Oil Flow: {round(qsu_std, 2)} bopd")
print(f"Suction Pressure: {round(psu_min, 2)} psig")
print(f"Diffuser Discharge Pressure: {round(pdi, 2)} psig")
print(f"Power Fluid Rate: {round(qnz_bpd, 2)} bwpd")
