import copy
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import regression as rg
import scipy
from scipy.optimize import curve_fit
import matplotlib

from labellines import labelLines

matplotlib.rc("font", size=16)
plt.style.use('tableau-colorblind10')

def normal(x, mu, sigma):
    return (
        1 / (np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )


bins = 10
mu = 5.0
sigma = 1.0
O_14_LIMIT = 0.07651166303158025


def generate_regressors(min_x, max_x, orders):
    regressors = [rg.HistogramRegressor(min_x, max_x, bins)]
    for order in orders:
        regressors.append(rg.BezierRegressor(min_x, max_x, order))
        regressors.append(rg.OrthoBezierRegressor(min_x, max_x, order))
        regressors.append(rg.MultiOrderRegressor(min_x, max_x, order))
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
        taylor = scipy.interpolate.approximate_taylor_polynomial(norm, mu, order, 10, 25)
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
    plt.semilogy(orders, no_rmses, label = "Non orthogonal Bernstein")
    plt.semilogy(orders, o_rmses, label="Orthonormal Bernstein")
    plt.semilogy(orders, multi_rmses, label="Multi-order Bernstein")
    plt.plot(plt.xlim(), [0.1, 0.1], "k--")
    plt.xlabel("Polynomial order")
    plt.ylabel("Truncation Error [%]")
    labelLines(plt.gca().get_lines(), align=False)
    for ext in {"svg", "png", "pdf"}:
        plt.savefig(f"order_all.{ext}")


def converge_samples():
    rgrs = generate_regressors(0, 10, [14])
    fig = plt.figure(figsize=(16, 9))
    samples = np.logspace(1, 5, 5, dtype=int)
    new_rgrs = run_regression(rgrs, lambda: np.random.normal(mu, sigma), samples)
    samples.dtype = float
    norm = lambda x: normal(x, mu, sigma)
    no_rmses = []
    o_rmses = []
    multi_rmses = []
    def poly_fit(x, a, b, c):
        return a*x**2 + b*x + c
    for num_samples, rgrs in new_rgrs.items():
        non_ortho, ortho, multi = rgrs[1:]
        no_rmses.append(get_rmse(non_ortho, norm, 0, 10) * 100)
        o_rmses.append(get_rmse(ortho, norm, 0, 10) * 100)
        multi_rmses.append(get_rmse(multi, norm, 0, 10) * 100)
    params, cov = curve_fit(poly_fit, samples, o_rmses)
    print(params)
    bound = lambda x: poly_fit(x, *params)
    plt.loglog(samples, bound(samples), "k--")
    #plt.loglog(samples, np.array(no_rmses) / O_14_LIMIT, label="Non-Orthogonal Bernstein")
    plt.plot(samples, np.array(o_rmses) / O_14_LIMIT, label="Orthonormal Bernstein")
    #plt.plot(samples, np.array(multi_rmses) / O_14_LIMIT, label="Multi-Order Bernstein")
    plt.plot(plt.xlim(), [1,1], "k--")
    plt.xlabel("Number of samples")
    plt.ylabel("Monte Carlo RMSE / Analytical RMSE")
    plt.legend()
    plt.show()
    return
    for ext in {"png", "svg", "pdf"}:
        plt.savefig(f"samples.{ext}")


converge_samples()
