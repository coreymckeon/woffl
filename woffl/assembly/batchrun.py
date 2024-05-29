"""Batch Jet Pump Runs

Contains code that is used to run multiple pumps at once to understand the
current conditions. Currently Scott's code is here with a simple nested for
loop. Will eventually update this to more of a class style system. Runs the
analysis and sends the results to a .csv file.
"""

import matplotlib.pyplot as plt
import pandas as pd

from woffl.assembly.easypump import jetpump_wrapper

# run from the command with the following:
# python -m woffl.assembly.batchrun

nozzles = ["9", "10", "11", "12", "13", "14"]
throats = ["X", "A", "B", "C", "D", "E"]

results = []

for nozzle in nozzles:
    for throat in throats:
        try:
            psu_solv, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te, total_wc, total_water, wellname = (
                jetpump_wrapper(
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
                "total_water": total_water,
                "total_wc": total_wc,
                "wellname": wellname,
            }
            results.append(result)
        except Exception as e:
            print(f"An error occurred for nozzle {nozzle} and throat {throat}: {e}")

dfjet = pd.DataFrame(results)
dfjet = dfjet.sort_values(by="psu_solv", ascending=True)
# df_sorted.to_csv("modelrun_output B-35.csv")
print(dfjet)

plt.plot(dfjet["total_water"], dfjet["qoil_std"], marker="o", linestyle="")
plt.xlabel("Total Water, BWPD")
plt.ylabel("Oil Rate, BOPD")
plt.show()
