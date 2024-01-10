![woffl_github7](https://github.com/kwellis/woffl/assets/62774251/8b80146f-a503-4576-8f43-f1aa45d93a05)
Numerical solver for liquid jet pumps with three-phase flow.   
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

