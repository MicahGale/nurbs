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
    for order in range(3):
        regressors.append(regressor.BezierRegressor(min_x, max_x, order * 6 + 2))
        regressors.append(regressor.OrthoBezierRegressor(min_x, max_x, order  * 6 + 2))

    print(regressors)
    xs = np.linspace(min_x, max_x)
    fig, axes_rows = plt.subplots(5, 6, figsize=(30, 20))
    for row_idx, (samples, ax_row) in enumerate(
        zip([5, 10, 100, 10_00, 10_000], axes_rows)
    ):
        print(samples)
        for i in range(samples):
            x = inverse()
            if x < min_x or x > max_x:
                continue
            for regr in regressors:
                regr.score(x)
        for col_idx, (r, ax) in enumerate(zip(regressors[1:], ax_row)):
            ax.plot(xs, func(xs), "k--", label="true curve")
            regressors[0].plot(
                ax,
            )
            r.plot(ax, func, ylim, 5)
            if col_idx == 0:
                ax.set_ylabel(
                    f"Samples = {samples:,g}",
                    rotation=0,
                    size=16,
                    labelpad=100,
                    fontweight="bold",
                )
            if row_idx == 0:
                ortho = "Non-Ortho" if col_idx % 2 == 0 else "Ortho"
                ax.set_title(
                    f"{ortho} Order={int(col_idx / 2) * 6 + 2}", size=16, fontweight="bold"
                )
        [r.reset() for r in regressors]
    fig.tight_layout()

    for ext in {"png", "svg", "pdf"}:
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
        lambda x: np.exp(-x),
        np.random.exponential,
        "exponential",
        0,
        5,
        (0, 1),
    ),
    (
        lambda x: np.cos(x),
        lambda: np.arcsin(np.random.uniform(0, 1)),
        "cosine",
        0,
        np.pi / 2,
        (0, 1),
    ),
]:
    print(trifecta[2])
    do_regression_series(*trifecta)
