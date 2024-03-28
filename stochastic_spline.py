#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from regression import regressor
import scipy
import time


def normal(x, mu, sigma):
    return (
        1
        / (np.sqrt(2 * np.pi * sigma**2))
        * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )


min = 0.0
max = 10.0
bins = 10
histo = regressor.HistogramRegressor(min, max, bins)
spline = regressor.SplineRegressor(min, max, bins, 4)
bezier = regressor.BezierRegressor(min, max, 10)

for name, curve in [("spline", spline), ("bezier", bezier)]:
    plt.clf()
    curve.plot_bases()
    plt.savefig(f"{name}_bases.png")

mu = 5.0
sigma = 1.0
for i in range(1000):
    x = np.random.normal(mu, sigma)
    if x < min or x > max:
        continue
    for regressor in [histo, spline, bezier]:
        regressor.score(x)

plt.clf()
bezier.plot()
spline.plot()
histo.plot()
x = np.linspace(0, 10)
plt.plot(x, normal(x, mu, sigma), "k--", label="true curve")
plt.legend()
plt.savefig("regression.png")
