def mass_fractions(watercut, fgor, rs, poil, pwat, pgas):
    """ Name:   Mass Fractions
        Input:  watercut - watercut at standard conditions
                fgor - formation gas oil ratio, scf/stb
                rs - gas solubility,  scf/stb                
                pwat - density of water, lbm/ft3 (standard conditions)
                poil - density of oil, lbm/ft3 (standard conditions)
                pgas - density of gas, lbm/ft3 (standard conditions)

        Output: xoil - mass fraction oil, mass oil / mass total
                xwat - mass fraction water, mass water / mass total
                xgas - mass fraction gas, mass gas / mass total
        """
    # watercut
    wc = watercut

    # mass based gas solubility
    mrs = (7.48/42)*rs*pgas/poil

    # convert from scf/bbl to scf/cf (make sure gas density is at standard conditions)
    # mass formation gas oil ratio
    mfgor = (7.48/42)*fgor*pgas/poil

    # mass watercut
    mwc = pwat*wc/(pwat*wc+poil*(1-wc))

    # mass formation gas to liquid ratio
    mfglr = (7.48/42)*fgor*pgas*(1-wc)/(wc*pwat+(1-wc)*poil)

    # mass fraction of gas
    xgas = mfglr/(1+mfglr)
    print(xgas)

    # mass fraction of oil
    xoil = (1-mwc)/(1+mfgor*(1-mwc))

    # mass fraction of water
    # won't change even with gas and oil trading some mass back and forth
    # slight differences in excel spreadsheet and python solution...? py: 0.9595, excel 0.972
    xwat = 1 - xoil - xgas

    # correct for the gas that is inside the oil
    xrs = xoil*mrs

    # calculate new mass fraction of free gas
    xgas = xgas - xrs
    xgas = max(xgas, 0)  # make sure mass fraction of gas is always above zero

    xoil = 1 - xwat - xgas

    # round the mass fractions
    deci = 4
    xgas = round(xgas, deci)
    xoil = round(xoil, deci)
    xwat = round(xwat, deci)

    # masss fractions
    return xoil, xwat, xgas

# could always do an array of mass fractions and an array of densities
# checks out with hysys if I use there calculated densities and mass fractions


def pmix(xoil, xwat, xgas, poil, pwat, pgas):
    """ Name:   Mixture Density
        Input:  xoil - mass fraction oil, mass oil / mass total
                xwat - mass fraction water, mass water / mass total
                xgas - mass fraction gas, mass gas / mass total                
                pwat - density of water, lbm/ft3 (actual cond.)
                poil - density of oil, lbm/ft3 (actual cond.)
                pgas - density of gas, lbm/ft3 (actual cond.)

        Output: pmix - density of mixture, lbm/ft3 (actual cond.)
        """

    # mixture specific volume
    vmix = (xwat/pwat) + (xoil/poil) + (xgas/pgas)

    pmix = 1/vmix
    pmix = round(pmix, 4)

    return pmix


def volm_fractions(xoil, xwat, xgas, poil, pwat, pgas):
    """ Name:   Volume Fractions
        Input:  xwat - mass fraction water, mass water / mass total
                xoil - mass fraction oil, mass oil / mass total
                xgas - mass fraction gas, mass gas / mass total                
                pwat - density of water, lbm/ft3 (actual cond.)
                poil - density of oil, lbm/ft3 (actual cond.)
                pgas - density of gas, lbm/ft3 (actual cond.)

        Output: ywat - volume fraction water, volume water / volume total
                yoil - volume fraction oil, volume oil / volume total
                ygas - volume fraction gas, volume gas / volume total
        """
    # would be nice to have a system of arrays...!
    # at that point it would just be matrix math

    # mixture specific volume
    vmix = (xwat/pwat) + (xoil/poil) + (xgas/pgas)

    pmix = 1/vmix

    ywat = xwat*pmix/pwat
    yoil = xoil*pmix/poil
    ygas = xgas*pmix/pgas

    # round the mass fractions
    deci = 4
    ywat = round(ywat, deci)
    yoil = round(yoil, deci)
    ygas = round(ygas, deci)

    return yoil, ywat, ygas
