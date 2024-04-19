from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import scipy


class Regressor(ABC):
    def __init__(self, dimensions):
        self._n = 0
        self._coeffs = np.zeros(dimensions)
        self._dim = dimensions

    def score(self, value):
        self._n += 1
        self._internal_score(value)

    @abstractmethod
    def _internal_score(self, value):
        pass

    @abstractmethod
    def normalize(self):
        pass

    @abstractmethod
    def plot(self, ax=None):
        pass

    def reset(self):
        self._coeffs = np.zeros(self._dim)
        self._n = 0


class HistogramRegressor(Regressor):
    def __init__(self, x_min, x_max, n_bins):
        self._min = x_min
        self._max = x_max
        self._n_bins = n_bins
        self._bin_width = (x_max - x_min) / n_bins
        super().__init__(n_bins)

    def _internal_score(self, value):
        index = int(value / self._bin_width)
        self._coeffs[index] += 1

    def normalize(self):
        return np.array(self._coeffs) / self._n

    def plot(self, ax=None):
        coeffs = self.normalize()
        if ax is None:
            plotter = plt
        else:
            plotter = ax
        plotter.stairs(coeffs, np.array(range(0, self._n_bins + 1)) * self._bin_width)


class FETRegressor(Regressor):
    def __init__(self, x_min, x_max, n_bases):
        super().__init__(n_bases)
        self._moments = np.zeros(n_bases)
        self._min = x_min
        self._max = x_max
        # calculate norms
        norms = []
        x = np.linspace(x_min, x_max, 1000)
        for i in range(self._dim):
            norms.append(np.trapz(self.evaluate_basis(x, i) ** 2, x))
        self._norms = np.array(norms)

    @abstractmethod
    def evaluate_basis(self, x, i):
        pass

    def normalize(self):
        norms = np.array([n if n > 0 else 1.0 for n in self._norms])
        coeffs = self._coeffs / (self._norms * self._n)
        squares = self._moments / (self._norms * self._n)
        ret = [
            ufloat(float(coef), float(1 / (self._n - 1) * np.sqrt(square - coef**2)))
            for coef, square in zip(coeffs, squares)
        ]
        return ret

    def _internal_score(self, value):
        for i in range(self._dim):
            basis_eval = self.evaluate_basis(value, i)
            self._coeffs[i] += basis_eval
            self._moments[i] += basis_eval**2

    def plot(self, ax=None, true_func=None, ylim=None, error_bar=1):
        coeffs = self.normalize()
        x = np.linspace(self._min, self._max, 100)
        y = np.zeros_like(x, dtype=np.object_)
        for i, coef in enumerate(coeffs):
            y += coef * self.evaluate_basis(x, i)
        if ax is None:
            plotter = plt
            if ylim:
                plotter.ylim(ylim)
        else:
            plotter = ax
            if ylim:
                plotter.set_ylim(ylim)
        lines = []
        mean = self.mean()
        std = self.std()
        legends = [f"Coeff rel-error: {self.rel_err():.2g}"]
        if true_func:
            legends.append(f"R^2: {self.r2(true_func):.2g}")
        plotter.legend(labels=legends)
        plotter.errorbar(mean, 0.3, xerr=std, fmt="k^", capsize=3)
        lines.append(plotter.plot(x, [v.n for v in y], "b-"))
        for i in range(1, error_bar + 1):
            lines.append(
                plotter.fill_between(
                    x,
                    [v.n + i * v.s for v in y],
                    [v.n - i * v.s for v in y],
                    alpha=0.3 - i*0.05,
                )
            )
        return lines

    def evaluate(self, x):
        coeffs = self.normalize()
        y = np.zeros_like(x)
        for i, coef in enumerate(coeffs):
            y += coef.n * self.evaluate_basis(x, i)
        return y

    def plot_bases(self):
        lines = []
        x = np.linspace(self._min, self._max, 100)
        for i in range(self._dim):
            lines.append(plt.plot(x, self.evaluate_basis(x, i)))
        return lines

    def plot_sum_bases(self):
        x = np.linspace(self._min, self._max, 100)
        y = np.zeros_like(x)
        for i in range(self._dim):
            y += self.evaluate_basis(x, i)
        return plt.plot(x, y)

    def mean(self):
        x = np.linspace(self._min, self._max, 1000)
        y = self.evaluate(x)
        return np.trapz(x * y, x)

    def std(self):
        mean = self.mean()
        x = np.linspace(self._min, self._max, 1000)
        y = self.evaluate(x)
        return np.sqrt(np.trapz(y * (x - mean) ** 2, x))

    def r2(self, true_func):
        x = np.linspace(self._min, self._max, 1000)
        y = self.evaluate(x)
        ybar = y.mean()
        true_y = true_func(x)
        residual_sum = np.square(y - true_y).sum()
        total_sum = np.square(y - ybar).sum()
        return 1 - residual_sum / total_sum

    def rel_err(self):
        coeffs = self.normalize()
        return np.array([c.s for c in coeffs]).mean()


class BezierRegressor(FETRegressor):
    def __init__(self, x_min, x_max, order=3):
        bases = []
        for i in range(order + 1):
            bases.append(self._generate_basis_function(order, i))
        self._bases = bases
        super().__init__(x_min, x_max, order + 1)

    def do_affine_transform(self, x):
        if isinstance(x, float):
            assert x >= self._min and x <= self._max
        else:
            assert (x >= self._min).all() and (x <= self._max).all()
        return (x - self._min) / (self._max - self._min)

    @staticmethod
    def _generate_basis_function(n, i):
        assert n >= i
        binom_coeff = scipy.special.binom(n, i)
        return lambda t: binom_coeff * t**i * (1 - t) ** (n - i)

    def evaluate_basis(self, x, i):
        return self._bases[i](self.do_affine_transform(x))


class OrthoBezierRegressor(BezierRegressor):
    """

    Based on:

    M. A. Bellucci, “On the explicit representation of orthonormal Bernstein polynomials.” Apr. 09, 2014.
        Accessed: Mar. 14, 2024. [Online]. Available: https://arxiv.org/abs/1404.2293
    """

    def __init__(self, x_min, x_max, order=3):
        bases = []
        for i in range(order + 1):
            bases.append(self._generate_basis_function(order, i))
        self._bases = bases
        super().__init__(x_min, x_max, order)

    @staticmethod
    def _generate_basis_function(n, i):
        def ortho_bern(t):
            first_poly = np.sqrt(2 * (n - i) + 1) * (1 - t) ** (n - i)
            value = 0
            for k in range(i + 1):
                binom = scipy.special.binom
                value += (
                    (-1) ** k * binom(2 * n + 1 - k, i - k) * binom(i, k) * t ** (i - k)
                )
            return first_poly * value

        return ortho_bern


class SplineRegressor(FETRegressor):
    def __init__(self, x_min, x_max, n_bins, order=3):
        self._n_bins = n_bins
        self._bin_width = (x_max - x_min) / n_bins
        self._order = order

        self._knots = np.concatenate(
            (
                [x_min] * (order - 1),
                np.linspace(x_min, x_max, n_bins + 1),
                [x_max] * (order - 1),
            )
        )
        splines = []
        for row in np.identity(len(self._knots)):
            splines.append(scipy.interpolate.BSpline(self._knots, row, order))
        self._splines = splines
        super().__init__(x_min, x_max, len(self._knots))

    def evaluate_basis(self, value, i):
        return scipy.interpolate.splev(value, self._splines[i])

    def plot(self, ax=None):
        coeffs = self.normalize()
        x = np.linspace(self._min, self._max, 100)
        spline = scipy.interpolate.BSpline(self._knots, coeffs, self._order)
        y = scipy.interpolate.splev(x, spline)
        if ax is None:
            plotter = plt
        else:
            plotter = ax
        return plotter.plot(x, y)

    def calculate_orthogonal(self):
        x = np.linspace(self._min, self._max, 100)
        products = np.zeros((self._dim, self._dim))
        splev = scipy.interpolate.splev
        for i, spline_1 in enumerate(self._splines):
            for j, spline_2 in enumerate(self._splines[: i + 1]):
                product = np.trapz(
                    np.multiply(splev(x, spline_1), splev(x, spline_2)), x
                )
                products[i, j] = product
        return products
