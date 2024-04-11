import woffl.flow.singlephase as sp
import woffl.flow.twophase as tp

# only works if the command python -m tests.flow_test is used
# example problem in Two Phase Flow in Pipes by Beggs / Brill (1988) pg. 3-31
book_Ngv = 9.29
book_Nlv = 6.02
book_Nd = 41.34
book_l1, book_l2 = 1.53, 0.88
book_vpat = "slug"

calc_l1, calc_l2 = tp.ros_lp(book_Nd)
calc_vpat = tp.ros_flow_pattern(9.29, 6.02, 41.34)

print(f"Ros L1 & L2 - Book: {book_l1}, {book_l2}, Calc: {round(calc_l1, 2)}, {round(calc_l2, 2)}")
print(f"Ros Regime - Book: {book_vpat}, Calc: {calc_vpat}")

# example problem in Two Phase Flow in Pipes by Beggs / Brill (1988) pg. 3-62
book_nslh = 0.393
book_NFr = 5.67
book_hpat = "intermittent"
book_tparm = 1  # this is really n/a since it is intermittent flow
book_ilh = 0.512
book_dhyd = 0.249 * 12  # inches
book_rho_liq = 56.6
book_rho_gas = 2.84
book_uliq = 18
book_ugas = 0.018
book_NRe = 8450
book_vsg = 4.09  # ft/s
book_vsl = 2.65  # ft/s
book_vmix = book_vsg + book_vsl
book_ff = 0.032  # book doesn't say what value they use for absolute roughness
book_yf = 1.5
book_sf = 0.37
book_ftp = 0.046
pipe_len = 100  # feet
book_fric = 3.12 * pipe_len / 144  # convert from psf/ft to psi
book_stat = 30.36 * pipe_len / 144  # convert from psf/ft to psi

# function is the same for slip vs no slip, just depends on what value for slip vs no slip you use
book_rho_mix = tp.density_slip(book_rho_liq, book_rho_gas, book_nslh)
book_rho_slip = tp.density_slip(book_rho_liq, book_rho_gas, book_ilh)
book_umix = tp.density_slip(book_uliq, book_ugas, book_nslh)

calc_NFr = tp.froude(book_vmix, book_dhyd)
calc_hpat, calc_tparm = tp.beggs_flow_pattern(book_nslh, book_NFr)
calc_ilh = tp.beggs_holdup_inc(book_nslh, book_NFr, book_Nlv, 90, calc_hpat, calc_tparm)
calc_NRe = sp.reynolds(book_rho_mix, book_vmix, book_dhyd, book_umix)
calc_rr = sp.relative_roughness(book_dhyd, 0.004)  # book doesn't say what abs ruff they use...
calc_ff = sp.ffactor_darcy(calc_NRe, calc_rr)
calc_yf = tp.beggs_yf(book_nslh, book_ilh)
calc_sf = tp.beggs_sf(book_yf)
calc_ftp = tp.beggs_ff(book_ff, book_sf)
calc_fric = tp.beggs_press_friction(book_ftp, book_rho_mix, book_vmix, book_dhyd, pipe_len)
calc_stat = tp.beggs_press_static(book_rho_slip, pipe_len)

print(f"Froude Number - Book: {book_NFr}, Calc: {round(calc_NFr, 2)}")
print(f"Beggs Regime - Book: {book_hpat}, Calc: {calc_hpat}")
print(f"Beggs Incline Liquid Holdup - Book: {book_ilh}, Calc: {round(calc_ilh, 3)}")
print(f"Reynolds Number - Book: {book_NRe}, Calc: {round(calc_NRe, 1)}")
print(f"Friction Factor - Book: {book_ff}, Calc: {round(calc_ff, 3)}")
print(f"Beggs Y Factor - Book: {book_yf}, Calc: {round(calc_yf, 3)}")
print(f"Beggs S Factor - Book: {book_sf}, Calc: {round(calc_sf, 4)}")
print(f"Beggs Friction Factor - Book: {book_ftp}, Calc: {round(calc_ftp, 3)}")
print(f"Beggs Friction Drop - Book: {round(book_fric, 3)} psi, Calc: {round(calc_fric, 3)} psi")
print(f"Beggs Static Drop - Book: {round(book_stat, 3)} psi, Calc: {round(calc_stat, 3)} psi")
