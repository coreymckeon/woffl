![woffl_github7](https://github.com/kwellis/woffl/assets/62774251/8b80146f-a503-4576-8f43-f1aa45d93a05)

Woffl (pronounced waffle) is a Python library for numerical modeling of subsurface jet pump oil wells.   

#### Installation   

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install woffl.   

```bash
pip install woffl
```   

#### Usage - PVT   
Defining an oil well in woffl is broken up into different classes that are combined together in an assembly that creates the model. The first part of definition are the oil properties from the reservoir. The module is called PVT and the classes are BlackOil, FormGas, FormWater and ResMix. BlackOil, FormGas and FormWat are the individual components in a reservoir stream and are fed into a ResMix where the formation gas oil ratio (FGOR) and watercut (WC) are defined.   

```python
from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

foil = BlackOil(oil_api=22, bubblepoint=1750, gas_sg=0.55)
fwat = FormWater(wat_sg=1)
fgas = FormGas(gas_sg=0.55)
fmix = ResMix(wc=0.3, fgor=800, oil=foil, wat=fwat, gas=fgas)
```

A condition of pressure and temperature can be set on individual components or on the ResMix which cascades it to the different components. Different properties can then be calculated. For example with ResMix the streams mass fractions, volumetric fractions, mixture density, component viscosities and mixture speed of sound can be estimated.   

```python
fmix = fmix.condition(press=1500, temp=80)
xoil, xwat, xgas = fmix.mass_fract()
yoil, ywat, ygas = fmix.volm_fract()
dens_mix = fmix.rho_mix()
uoil, uwat, ugas = fmix.visc_comp()
snd_mix = fmix.cmix()
```
If the reader wants to calculate the insitu 
#### Background
Jet pump studies fron the 1970's were interested in pumping single phase incompressible flows or single phase compressible flows. The models produced relied on assumptions such as constant density or an ideal gas to analytically solve the equations. In 1995 Cunningham wrote a paper with equations that govern a water jet for pumping a two-phase mixture. The equations relied on assumptions of constant density for the liquid and ideal gas law for the solution. Those assumptions are not valid when modeling a three-phase mixture of crude oil, water and natural gas. The crude oil is gas soluble and compressible. The equations for the inverse density of crude oil cannot be analytically integrated. A numerical solution needs to be applied.   
#### Fundamental Equation
The fundamental equation in the analysis of a jet pump is the un-integrated energy equation. No work is done, heat is not transferred and a significant height difference is not present. The un-integrated energy equation takes the following form. 
$$\frac{dp}{\rho} + \nu d\nu = 0$$
The fluid density is denoted by $\rho$ and the velocity is denoted by $\nu$. 
#### Relevant Papers   
- Cunningham, R. G., 1974, “Gas Compression With the Liquid Jet Pump,” ASME J Fluids Eng, 96(3), pp. 203–215.
- Cunningham, R. G., 1995, “Liquid Jet Pumps for Two-Phase Flows,” ASME J Fluids Eng, 117(2), pp. 309–316.
- Merrill, R., Shankar, V., and Chapman, T., 2020, “Three-Phase Numerical Solution for Jet Pumps Applied to a Large Oilfield,” SPE-202928-MS, November 10, 2020.
- Himr, D., Habán, V., Pochylý, F., 2009, "Sound Speed in the Mixture Water - Air," Engineering Mechanics, Svratka, Czech Republic, May 11–14, 2009, Paper 255, pp. 393-401. 

