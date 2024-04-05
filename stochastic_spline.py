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
regressors = [regressor.HistogramRegressor(min, max, bins)]
for order in range(6):
    regressors.append(regressor.OrthoBezierRegressor(min, max, (order + 1) * 2))


mu = 5.0
sigma = 1.0
xs = np.linspace(0, 10)
fig, axes_rows = plt.subplots(5, 6, figsize=(30, 20))
for samples, ax_row in zip([10, 100, 1000, 10_000, 100_000], axes_rows):
    for i in range(samples):
        x = np.random.normal(mu, sigma)
        if x < min or x > max:
            continue
        for regressor in regressors:
            regressor.score(x)
    for i, (r, ax) in enumerate(zip(regressors[1:], ax_row)):
        ax.plot(xs, normal(xs, mu, sigma), "k--", label="true curve")
        regressors[0].plot(ax)
        r.plot(ax)
        ax.set_title(f"n={samples} order={(i+1)*2}")
    [r.reset() for r in regressors]

plt.savefig("regression.png")
