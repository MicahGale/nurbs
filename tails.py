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


def plot(rgrs, anal_rgr):
    norm = lambda x: normal(x, mu, sigma)
    stoc_rmse_trials = []
    numer_rmse_trials = []
    for rgr in rgrs:
        rmse_x = []
        stoc_rmses = []
        anal_rmses = []
        for i in range(1, 11):
            rmse_x.append(i - 1 / 2)
            integral = scipy.stats.norm.cdf(i) - scipy.stats.norm.cdf(i - 1)
            stoc_rmses.append(get_rmse(rgr, norm, i - 1, i) * 100)
            anal_rmses.append(get_rmse(anal_rgr, norm, i - 1, i) * 100)
        stoc_rmse_trials.append(stoc_rmses)
        numer_rmse_trials.append(anal_rmses)
    stoc_rmse_trials = np.array(stoc_rmse_trials)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    x = np.linspace(0, 10, 100)
    twin_ax = ax.twinx()
    twin_ax.violinplot(stoc_rmse_trials.T.tolist(), rmse_x, showmeans=True)
    twin_ax.plot(
        rmse_x,
        stoc_rmse_trials.mean(axis=0),
        "-.",
        color="tab:blue",
        label="Stochastic RMSE",
    )
    twin_ax.plot(rmse_x, anal_rmses, "-.^", label="numerical RMSE")
    twin_ax.set_yscale("log")
    twin_ax.set_ylabel("Absolute sub-domain RMSE [%]")
    ax.plot(x, norm(x), "k--", label="Normal distribution")
    ax.plot(x, rgr.evaluate(x), label="stochastic FET")
    ax.plot(x, anal_rgr.evaluate(x), label="Numerical FET")
    ax2.plot(x, norm(x) - rgr.evaluate(x), "b-.", label="absolute error")
    ax3 = ax2.twinx()
    ax3.semilogy(
        x,
        np.abs(norm(x) - anal_rgr.evaluate(x)) / norm(x),
        "k-.",
        label="relative error",
    )
    labelLines(ax.get_lines(), align=False, xvals=[3, 2, 8], yoffsets=-0.01)
    labelLines(twin_ax.get_lines(), align=False, xvals=[1, 9], yoffsets=[0.08, 0.03])
    labelLines(ax2.get_lines(), align=False, xvals=[2])
    labelLines(ax3.get_lines(), align=False, xvals=[8])
    ax.set_xlabel("x")
    ax.set_ylabel("p(x)")
    ax2.set_ylabel("absolute error [-]")
    ax3.set_ylabel("relative error [-]")
    for ext in {"svg", "png", "pdf"}:
        plt.savefig(f"subdomain_rmse.{ext}")


anal_rgr = do_analytic(14, lambda x: normal(x, mu, sigma))
rgrs = [
    perform_regression(14, 1_000, lambda: np.random.normal(mu, sigma))
    for _ in range(100)
]
plot(rgrs, anal_rgr)
