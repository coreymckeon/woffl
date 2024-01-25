# figure out the equation for figure 5.4 on the Dun / Ros
# https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python

# mutliphase book
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

nd = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
l1 = [2, 2.0, 2, 1.9, 1.6, 1.3, 1.1, 1, 1, 1, 1]


# plt.show()


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


p0 = [1.02, 43.392, -0.139, 1]  # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, nd, l1, p0, method="dogbox")

print(popt)

L, x0, k, b = popt

nd_test = np.linspace(1, 100, 25)
L = 1.02
x0 = 43.4
k = -0.139
b = 0.99
l1_test = sigmoid(nd_test, L, x0, k, b)

plt.scatter(nd, l1)
plt.scatter(nd_test, l1_test)
plt.show()
