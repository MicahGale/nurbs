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

mu = 5.0
sigma = 1.0


def do_regression_series(func, inverse, name, min_x, max_x, ylim):
    regressors = [regressor.HistogramRegressor(min_x, max_x, bins)]
    for order in range(6):
        regressors.append(regressor.OrthoBezierRegressor(min_x, max_x, (order + 1) * 2))

    xs = np.linspace(min_x, max_x)
    fig, axes_rows = plt.subplots(5, 6, figsize=(30, 20))
    for samples, ax_row in zip([5, 10, 100, 10_00, 10_000], axes_rows):
        print(samples)
        for i in range(samples):
            x = inverse()
            if x < min_x or x > max_x:
                continue
            for regr in regressors:
                regr.score(x)
        for i, (r, ax) in enumerate(zip(regressors[1:], ax_row)):
            ax.plot(xs, func(xs), "k--", label="true curve")
            regressors[0].plot(
                ax,
            )
            r.plot(ax, func, ylim, 5)
            ax.set_title(f"n={samples} order={(i+1)*2}")
        [r.reset() for r in regressors]

    for ext in {"png", "svg"}:
        plt.savefig(f"{name}_regression.{ext}")


for trifecta in [
    (
        lambda x: normal(x, mu, sigma),
        lambda: np.random.normal(mu, sigma),
        "normal",
        0,
        10,
        (-0.1, 0.5),
    ),
    (
        np.exp,
        np.random.exponential,
        "exponential",
        0,
        5,
        (0, 1),
    ),
    (
        lambda x: 1 / 2 * np.cos(x),
        lambda: np.arcsin(np.random.uniform(0, 1) - 1 / 2),
        "cosine",
        -np.pi / 2,
        np.pi / 2,
        (0, 1),
    ),
]:
    print(trifecta[2])
    do_regression_series(*trifecta)
