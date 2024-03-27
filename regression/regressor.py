from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
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
    def plot(self):
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

    def plot(self):
        coeffs = self.normalize()
        plt.stairs(coeffs, np.array(range(0, self._n_bins + 1)) * self._bin_width)


class FETRegressor(Regressor):
    def __init__(self, x_min, x_max, n_bases):
        super().__init__(n_bases)
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
        return self._coeffs / (self._norms * self._n)

    def _internal_score(self, value):
        for i in range(self._dim):
            self._coeffs[i] += self.evaluate_basis(value, i)

    def plot(self):
        coeffs = self.normalize()
        x = np.linspace(self._min, self._max, 100)
        y = np.zeroslike(x)
        for i in range(self._dim):
            y += evaluate_basis(x, i)


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

    def plot(self):
        coeffs = self.normalize()
        x = np.linspace(self._min, self._max, 100)
        spline = scipy.interpolate.BSpline(self._knots, coeffs, self._order)
        y = scipy.interpolate.splev(x, spline)
        return plt.plot(x, y)

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
