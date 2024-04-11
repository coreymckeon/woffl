import matplotlib.pyplot as plt
import numpy as np

from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formwat import FormWater

o_wat = FormWater.schrader_wat()  # Omega Pad Power Fluid
o_oil = BlackOil.schrader_oil()

pf_rate = 125000  # bwpd in powerfluid to the deoiler
oil_ppm = 7000  # ppm of oil in water


def approx_offgas(
    pin: float, pout: float, temp: float, qwat: float, oil_ppm: float, wat_prop: FormWater, oil_prop: BlackOil
) -> tuple[float, float]:
    """Calculate Rate of Gas Coming off the Deoiler

    Args:
        pin (float): Pressure Upstream Control valve, psig
        pout (float): Pressure of Deoiler, psig
        temp (float): Deg F
        qwat (float): Water Flowrate into deoiler, bwpd
        oil_ppm (float): Oil PPM in Deoiler Water Stream

    Returns:
        qoil (float): Oil Rate, bopd
        qgas (float): Gas Off Deoiler, SCF/Day
    """
    qoil = qwat * oil_ppm / 1e6
    drsw = wat_prop.condition(pin, temp).gas_solubility() - wat_prop.condition(pout, temp).gas_solubility()
    drso = oil_prop.condition(pin, temp).gas_solubility() - oil_prop.condition(pout, temp).gas_solubility()

    print(f"Difference in Water Gas Solubility: {round(drsw, 2)} scf/bbl")
    print(f"Difference in Oil Gas Solubility: {round(drso, 2)} scf/bbl")
    qgas = drso * qoil + drsw * qwat
    return qoil, qgas


qoil, qgas = approx_offgas(300, 200, 150, pf_rate, oil_ppm, o_wat, o_oil)
print(f"Oil Rate: {qoil} bopd and Gas Rate: {round(qgas/1e3, 2)} mscfd")
