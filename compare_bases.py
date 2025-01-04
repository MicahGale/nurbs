import copy
import collections as co
import itertools as it
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import regression as rg
import pickle
import scipy
from scipy.optimize import curve_fit
import matplotlib
from threading import Thread

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


def generate_regressors(min_x, max_x, orders):
    # regressors = [rg.HistogramRegressor(min_x, max_x, bins)]
    regressors = []
    for order in orders:
        regressors.append(rg.BezierRegressor(min_x, max_x, order))
        regressors.append(rg.OrthoBezierRegressor(min_x, max_x, order))
    return regressors


def run_regression(rgrs, inverse, nums):
    ret = {}
    previous_num = 0
    for num in nums:
        print(num)
        for _ in range(num - previous_num):
            sample = inverse()
            while sample < 0 or sample > 10:
                sample = inverse()
            for rgr in rgrs:
                rgr.score(sample)
        ret[num] = copy.deepcopy(rgrs)
    return ret


def get_rmse(rgr, true_func, min_x, max_x):
    x = np.linspace(min_x, max_x, 1000)
    return np.sqrt(
        np.trapezoid((true_func(x) - rgr.evaluate(x)) ** 2, x) / (max_x - min_x)
    )


def converge_order():
    orders = range(2, 30)
    fig = plt.figure(figsize=(16, 9))
    norm = lambda x: normal(x, mu, sigma)
    rgrs = generate_regressors(0, 10, orders)
    rmses = []
    x = np.linspace(0, 10, 10_000)
    for order in orders:
        taylor = scipy.interpolate.approximate_taylor_polynomial(
            norm, mu, order, 10, 25
        )
        rmses.append(np.sqrt(np.trapezoid((norm(x) - taylor(x - 5)) ** 2) / 10) * 100)
    for rgr in rgrs[1:]:
        rgr.analytic_inner_prod(norm)
    no_rmses = []
    o_rmses = []
    multi_rmses = []
    for non_ortho, ortho, multi in it.batched(rgrs[1:], 3):
        no_rmses.append(get_rmse(non_ortho, norm, 0, 10) * 100)
        o_rmses.append(get_rmse(ortho, norm, 0, 10) * 100)
        multi_rmses.append(get_rmse(multi, norm, 0, 10) * 100)
    # get order 14
    index = list(orders).index(14)
    print(o_rmses[index])
    plt.semilogy(orders, rmses, label="Taylor Polynomial")
    plt.semilogy(orders, no_rmses, label="Non orthogonal Bernstein")
    plt.semilogy(orders, o_rmses, label="Orthonormal Bernstein")
    plt.semilogy(orders, multi_rmses, label="Multi-order Bernstein")
    plt.plot(plt.xlim(), [0.1, 0.1], "k--")
    plt.xlabel("Polynomial order")
    plt.ylabel("Truncation Error [%]")
    labelLines(plt.gca().get_lines(), align=False)
    for ext in {"svg", "png", "pdf"}:
        plt.savefig(f"order_all.{ext}")


def converge_samples():
    rgrs_groups = co.deque()
    handled_groups = co.deque()
    samples = np.logspace(1, 2, 20, dtype=int)
    for i in range(10):
        rgrs_groups.append(generate_regressors(0, 10, [14]))

    def regress(i):
        while True:
            try:
                rgrs = rgrs_groups.popleft()
                print(f"Thread {i}")
                handled_groups.append(
                    run_regression(rgrs, lambda: np.random.normal(mu, sigma), samples)
                )
            except IndexError:
                return

    thread_pool = []
    for i in range(12):
        t = Thread(target=regress, args=[i])
        t.start()
        thread_pool.append(t)
    for thread in thread_pool:
        thread.join()
    fig = plt.figure(figsize=(16, 9))
    norm = lambda x: normal(x, mu, sigma)
    no_rmses = []
    o_rmses = []
    multi_rmses = []
    x_samples = []

    def poly_fit(x, a, b):
        return a * x**b + 1

    for new_rgrs in handled_groups:
        for num_samples, rgrs in new_rgrs.items():
            non_ortho, ortho, multi = rgrs
            no_rmses.append(get_rmse(non_ortho, norm, 0, 10) * 100)
            o_rmses.append(get_rmse(ortho, norm, 0, 10) * 100)
            x_samples.append(num_samples)

    df = pd.DataFrame(
        {
            "samples": x_samples,
            "non_ortho": no_rmses,
            "ortho": o_rmses,
        }
    )
    df.to_excel("rmse_data.xlsx")
    for rmses, name, symbol in [
        (no_rmses, "Non-orthogonal Bernstein", "<"),
        (o_rmses, "Orthonormal Bernstein", ">"),
        (multi_rmses, "Multi-order Bernstein", "^"),
    ]:
        y = np.array(rmses) / O_14_LIMIT
        lines = plt.plot(x_samples, y, symbol, label=name)
        if "Non" in name:
            params, cov = curve_fit(poly_fit, x_samples, y)
            print(params)
            bound = lambda x: (poly_fit(x, *params))
            label = f"Non-ortho: ${params[0]:.1g}n^{{{params[1]:.1g}}} + 1.0$"
            residuals = y - bound(x_samples)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            label += f" $R^2={r2:.2g}$"
            plt.loglog(
                samples, bound(samples), "--", color=lines[0].get_color(), label=label
            )
        plt.plot(plt.xlim(), [1, 1], "k--")
        plt.xlabel("Number of samples")
        plt.ylabel("Monte Carlo RMSE / Analytical RMSE")
        plt.legend()
        for ext in {"png", "svg", "pdf"}:
            plt.savefig(f"samples.{ext}")


converge_samples()
