import pandas as pd
from assembly.run_pump import model_pump

"""
This is a file to call the run_pump file from Assembly folder

Loops through a list of jet pump nozzle and throat sizes and returns a csv file with the results
Update well characterisitics in the call to the model pump function

"""


nozzles = ["9", "10", "11", "12", "13", "14"]
throats = ["X", "A", "B", "C", "D", "E"]

results = []

for nozzle in nozzles:
    for throat in throats:
        try:
            psu_solv, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te, total_wc, total_water, wellname = (
                model_pump(
                    True,  # isSchrader?
                    pwh=201,  # WHP
                    rho_pf=62.4,  # PF density
                    ppf_surf=2419,  # PF pres
                    out_dia=4.5,  # Tubing OD
                    thick=0.5,  # tubing thickness
                    qwf=320,  # Oil rate
                    pwf=547,  # FBHP
                    res_pres=800,  # res pressure
                    form_wc=0.01,  # watercut
                    form_gor=507,  # gor
                    form_temp=72,  # form temp
                    nozzle_no=nozzle,  # nozzle size
                    throat=throat,  # nozzel area ratio with throat
                    wellname="MPB-35",
                )
            )
            result = {
                "nozzle": nozzle,
                "throat": throat,
                "psu_solv": psu_solv,
                "sonic_status": sonic_status,
                "qoil_std": qoil_std,
                "fwat_bwpd": fwat_bwpd,
                "qnz_bwpd": qnz_bwpd,
                "mach_te": mach_te,
                "total water": total_water,
                "total_wc": total_wc,
                "wellname": wellname,
            }
            results.append(result)
        except Exception as e:
            print(f"An error occurred for nozzle {nozzle} and throat {throat}: {e}")

df_results = pd.DataFrame(results)
df_sorted = df_results.sort_values(by="psu_solv", ascending=True)
df_sorted.to_csv("modelrun_output B-35.csv")
print(df_sorted)
