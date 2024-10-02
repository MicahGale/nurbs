import copy
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import regression as rg
import matplotlib

matplotlib.rc("font", size=16)


def normal(x, mu, sigma):
    return (
        1 / (np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )


bins = 10
mu = 5.0
sigma = 1.0


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
    x = np.linspace(min_x, max_x)
    return np.sqrt(
        np.trapezoid((true_func(x) - rgr.evaluate(x)) ** 2, x) / (max_x - min_x)
    )


def converge_order():
    orders = range(2, 25)
    fig = plt.figure(figsize=(16, 9))
    rgrs = generate_regressors(0, 10, orders)
    norm = lambda x: normal(x, mu, sigma)
    for rgr in rgrs[1:]:
        rgr.analytic_inner_prod(norm)
    no_rmses = []
    o_rmses = []
    multi_rmses = []
    for non_ortho, ortho, multi in it.batched(rgrs[1:], 3):
        no_rmses.append(get_rmse(non_ortho, norm, 0, 10) * 100)
        o_rmses.append(get_rmse(ortho, norm, 0, 10) * 100)
        multi_rmses.append(get_rmse(multi, norm, 0, 10) * 100)
    plt.semilogy(orders, no_rmses, label="Non-Orthogonal Bernstein")
    plt.plot(orders, o_rmses, label="Orthonormal Bernstein")
    plt.plot(orders, multi_rmses, label="Multi-order Bernstein")
    plt.xlabel("Order of Polynomial")
    plt.ylabel("Root Mean Squared Error [%]")
    plt.legend()
    for ext in {"png", "svg", "pdf"}:
        plt.savefig(f"order.{ext}")


def converge_samples():
    rgrs = generate_regressors(0, 10, [14])
    fig = plt.figure(figsize=(16, 9))
    samples = np.logspace(1, 6, 6, dtype=int)
    new_rgrs = run_regression(rgrs, lambda: np.random.normal(mu, sigma), samples)
    norm = lambda x: normal(x, mu, sigma)
    no_rmses = []
    o_rmses = []
    multi_rmses = []
    for num_samples, rgrs in new_rgrs.items():
        non_ortho, ortho, multi = rgrs[1:]
        no_rmses.append(get_rmse(non_ortho, norm, 0, 10) * 100)
        o_rmses.append(get_rmse(ortho, norm, 0, 10) * 100)
        multi_rmses.append(get_rmse(multi, norm, 0, 10) * 100)
    plt.loglog(samples, no_rmses, label="Non-Orthogonal Bernstein")
    plt.plot(samples, o_rmses, label="Orthonormal Bernstein")
    plt.plot(samples, multi_rmses, label="Multi-Order Bernstein")
    plt.xlabel("Number of samples")
    plt.ylabel("Root Mean Squared Error [%]")
    plt.legend()
    for ext in {"png", "svg", "pdf"}:
        plt.savefig(f"samples.{ext}")


converge_samples()
