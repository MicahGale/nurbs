import copy
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import regression as rg
import matplotlib

matplotlib.rc("font", size = 16)

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
        for _ in range(num - previous_num):
            sample = inverse()
            for rgr in rgrs:
                rgr.score(sample)
        ret[num] = copy.deepcopy(rgrs)
    return ret


def get_rmse(rgr, true_func, min_x, max_x):
    x = np.linspace(min_x, max_x)
    return np.sqrt(np.trapezoid((true_func(x) - rgr.evaluate(x)) ** 2, x) / (max_x - min_x))


def Converge_order():
    orders =  range(2, 20)
    rgrs = generate_regressors(0, 10, orders)
    new_rgrs = run_regression(rgrs, lambda: np.random.normal(mu, sigma), [10_000])
    norm = lambda x: normal(x, mu, sigma)
    for num_samples, rgrs in new_rgrs.items():
        no_rmses = []
        o_rmses = []
        for non_ortho, ortho in it.batched(rgrs[1:], 2):
            no_rmses.append(get_rmse(non_ortho, norm, 0, 10)*100)
            o_rmses.append(get_rmse(ortho, norm, 0, 10)*100)
        plt.semilogy(orders, no_rmses, label = "Non-Orthogonal Bernstein")
        plt.plot(orders, o_rmses, label = "Orthonormal Bernstein")
        plt.xlabel("Order of Polynomial")
        plt.ylabel("Root Mean Squared Error [%]")
        plt.legend()
        plt.show()

def converge_samples():
    rgrs = generate_regressors(0, 10, [14])
    samples = np.logspace(1,3,3, dtype=int)
    new_rgrs = run_regression(rgrs, lambda: np.random.normal(mu, sigma), samples)
    norm = lambda x: normal(x, mu, sigma)
    no_rmses = []
    o_rmses = []
    multi_rmses = []
    for num_samples, rgrs in new_rgrs.items():
        non_ortho, ortho, multi = rgrs[1:]
        no_rmses.append(get_rmse(non_ortho, norm, 0, 10)*100)
        o_rmses.append(get_rmse(ortho, norm, 0, 10)*100)
        multi_rmses.append(get_rmse(multi, norm, 0, 10)*100)
    plt.loglog(samples, no_rmses, label = "Non-Orthogonal Bernstein")
    plt.plot(samples, o_rmses, label = "Orthonormal Bernstein")
    plt.plot(samples, multi_rmses, label = "Multi-Order Bernstein")
    plt.xlabel("Number of samples")
    plt.ylabel("Root Mean Squared Error [%]")
    plt.legend()
    plt.show()
converge_samples()
