import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regression as rg
import scipy
from scipy.optimize import curve_fit
import matplotlib
from labellines import labelLines

matplotlib.rc("font", size=16)
plt.style.use("tableau-colorblind10")

def normal(x, mu, sigma):
    return (
        1 / (np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )


bins = 10
mu = 5.0
sigma = 1.0
O_14_LIMIT = 0.07651166303158025

def do_analytic(order, true_func):
    rgr = rg.BezierRegressor(0, 10, order)
    rgr.analytic_inner_prod(true_func)
    return rgr

def perform_regression(order, samples, inverse):
    rgr = rg.BezierRegressor(0, 10, order)
    for i in range(samples):
        sample = inverse()
        while sample < 0 or sample > 10:
            sample = inverse()
        rgr.score(sample)
    return rgr

def get_rmse(rgr, true_func, min_x, max_x):
    x = np.linspace(min_x, max_x, 1000)
    return np.sqrt(
        np.trapezoid((true_func(x) - rgr.evaluate(x)) ** 2, x) / (max_x - min_x)
    )

def plot(rgr, anal_rgr):
    norm = lambda x: normal(x, mu, sigma)
    rmse_x = []
    stoc_rmses = []
    anal_rmses = []
    for i in range(1, 11):
        rmse_x.append(i - 1/2)
        stoc_rmses.append(get_rmse(rgr, norm, i - 1, i) * 100)
        anal_rmses.append(get_rmse(anal_rgr, norm, i - 1, i) * 100)
    fig, (ax, ax2)= plt.subplots(2, 1, figsize=(16, 9))
    x = np.linspace(0, 10, 100)
    ax.plot(x, norm(x), "--", label="Normal distribution")
    ax.plot(x, anal_rgr.evaluate(x), label="Analytical FET")
    ax.plot(x, rgr.evaluate(x), label="stochastic FET")
    ax2.plot(x, norm(x) - anal_rgr.evaluate(x), "b-.", label="absolute error")
    ax3 = ax2.twinx()
    ax3.semilogy(x, np.abs(norm(x) - anal_rgr.evaluate(x))/norm(x), "k-.", label="relative error")
    labelLines(ax.get_lines(), align=False)
    labelLines(ax2.get_lines(), align=False)
    labelLines(ax3.get_lines(), align=False)
    ax.set_xlabel("x")
    ax.set_ylabel("p(x)")
    ax2.set_ylabel("absolute error [-]")
    ax3.set_ylabel("relative error [-]")
    plt.show()
    for ext in {"svg", "png", "pdf"}:
        plt.savefig(f"subdomain_rmse.{ext}")

anal_rgr = do_analytic(14, lambda x: normal(x, mu, sigma))
rgr = perform_regression(14, 1_000, lambda : np.random.normal(mu, sigma))
plot(rgr, anal_rgr)
