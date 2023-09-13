from jetpump import *

# testing the jet pump code on E-42

tube_od = 4.5  # inch
case_id = 6.875  # inch

jpump_md = 6693  # feet
jpump_tvd = 4096.8  # feet, interpolated off well profile

bpd_pf = 3500  # bwpd, power fluid rate
d_pf = 62.4  # lbm/ft3
u_pf = 1  # cp, viscosity of power fluid
ppf_surf = 3046  # psi, power fluid surf pressure

e42jetpump = JetPump('13', 'B')

anlr_a, anlr_dhyd = anlr_dims(tube_od, case_id)
q_pf = bpd_to_ft3s(bpd_pf)

v_ann = round(q_pf/anlr_a, 2)  # ft/s annular velocity
print(f'Annular Velocity: {v_ann} ft/s')

rey = reynolds(d_pf, v_ann, anlr_dhyd, u_pf)
print(f'Reynolds Number {rey}')
f_ann = serghide(0.001, rey)
print(f'Friction Factor {f_ann}')
# add functionality of turbulent flow or not?
dp_ann = dp_pipe(f_ann, jpump_md, d_pf, v_ann, anlr_dhyd)
print(f'Friction Drop in Annulus {dp_ann} psi')

p_st = static_press(d_pf, jpump_tvd)
p_pf = ppf_surf + p_st - dp_ann

print(f'Surf PF Pressure {ppf_surf} psi')
print(f'Pump PF Pressure {round(p_pf,0)} psi')

e42jetpump.p_jt(q_pf, p_pf, d_pf)
