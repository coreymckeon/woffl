# type: ignore
# flake8: noqa

# a place to store legacy code
# as stuff is cleaned up


def tee_near_pmo(psu: float, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix) -> float:
    """Throat Entry Equation near pmo, mach 1 pressure

    Find the value of the throat entry equation near the mach 1 pressure. The following
    function will be iterated across to minimize the throat entry equation.

    Args:
        psu (float): Suction Press, psig
        tsu (float): Suction Temp, deg F
        ken (float): Throat Entry Friction, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        tee_pmo (float): Throat Entry Value near pmo, psig"""

    qoil_std, pte_ray, rho_ray, vel_ray, snd_ray = throat_entry_arrays(psu, tsu, ate, ipr_su, prop_su)
    pmo = throat_entry_mach_one(pte_ray, vel_ray, snd_ray)
    mask = pte_ray >= pmo  # only use values where pte_ray is greater than pmo, haven't hit mach 1
    # note: do we even have to calc  and filter pmo? TEE vs pte is a parabola which anyway...?
    # discontinuities with mask function might screw all this up...
    kse_ray, ese_ray = throat_entry_energy(ken, pte_ray[mask], rho_ray[mask], vel_ray[mask])
    tee_ray = kse_ray + ese_ray
    tee_pmo = min(tee_ray)  # find the smallest value of tee where mach <=1
    return tee_pmo


"""
# find where mach = 1 (pmo), insert pmo into pte, calculate rho, vel and snd at arrays
    pmo = throat_entry_mach_one(pte_ray, vel_ray, snd_ray)
    # flip array for ascedning order instead of descending
    pmo_idx = np.searchsorted(np.flip(pte_ray), pmo) - pte_ray.size  # find position of pmo

    # repeat finding properties where Mach Number equals one
    prop_su = prop_su.condition(pmo, tsu)
    rho_oil = prop_su.oil.density  # oil density
    yoil, ywat, ygas = prop_su.volm_fract()
    qoil, qwat, qgas = actual_flow(qoil_std, rho_oil_std, rho_oil, yoil, ywat, ygas)
    qtot = qoil + qwat + qgas

    # insert values where Mach Number equals one
    pte_ray = np.insert(arr=pte_ray, obj=pmo_idx, values=pmo)
    vel_ray = np.insert(arr=vel_ray, obj=pmo_idx, values=qtot / ate)
    rho_ray = np.insert(arr=rho_ray, obj=pmo_idx, values=prop_su.pmix())
    snd_ray = np.insert(arr=snd_ray, obj=pmo_idx, values=prop_su.cmix())
    """
