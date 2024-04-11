# figure out the equation for figure 5.4 on the Dun / Ros
# https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python

import math

# mutliphase book
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

nd = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
l1 = [2, 2.0, 2, 1.9, 1.6, 1.3, 1.1, 1, 1, 1, 1]

nd2 = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
l2 = [
    0.5,
    0.5,
    0.5,
    0.5,
    0.6,
    0.75,
    0.85,
    0.9,
    0.95,
    1,
    1.05,
    1.1,
    1.15,
    1.15,
    1.15,
    1.15,
    1.15,
    1.15,
    1.15,
    1.15,
    1.15,
]

m = (1.15 - 0.5) / (65 - 15)
b = 1.15 - m * 65

l2_new = []
for i, x in enumerate(nd2):
    if (x > 15) and (x < 65):
        y = m * x + b
    else:
        y = l2[i]
    l2_new.append(y)


def sigmoid(x: float | np.ndarray, L: float, x0: float, k: float, b: float) -> float | np.ndarray:
    """Sigmoid Function for Curve Fitting

    Args:
        x (float): Input data
        L (float): Scales Output Range from [0, 1] to [0, L]
        x0 (float): Middle point of Sigmoid on x-axis
        k (float): Scales the input, remains in (-inf, inf)
        b (float): Output bias, changing range from [0, L] to [b, L + b]

    Returns:
        y (float): Sigmoid Function Output
    """
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


p0 = [1.02, 43.392, -0.139, 1]  # this is an mandatory initial guess
popt, pcov = curve_fit(sigmoid, nd, l1, p0, method="dogbox")
L, x0, k, b = popt
print(f"Curve fit parameters for l1 are {popt}")

nd_test = np.linspace(1, 100, 25)
l1_test = sigmoid(nd_test, L, x0, k, b)
# good data, curve fits
# L = 1.02
# x0 = 43.4
# k = -0.139
# b = 0.99

n0 = [0.8, 35, 0.25, 0.5]
nopt, ncov = curve_fit(sigmoid, nd2, l2_new, n0, method="dogbox")
q, w, e, r = nopt
print(f"Curve fit parameters for l2 are {nopt}")
l2_test = sigmoid(nd_test, q, w, e, r)

# plt.scatter(nd2, l2, label="l2 old data")
plt.plot(nd, l1, linestyle="none", marker="o", label="L1 data")
plt.plot(nd_test, l1_test, linestyle="--", marker="none", label="L1 fit")
plt.plot(nd2, l2_new, linestyle="none", marker="o", label="L2 data")
plt.plot(nd_test, l2_test, linestyle="--", marker="none", label="L2 fit")
plt.xlabel("Ros Dimensionless Pipe Diameter Number")
plt.ylabel("Ros L1 and L2 Values")
plt.legend()
plt.show()
