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


class SplineRegressor(Regressor):
    def __init__(self, x_min, x_max, n_bins, order=3):
        self._min = x_min
        self._max = x_max
        self._n_bins = n_bins
        self._bin_width = (x_max - x_min) / n_bins
        self._knots = np.concatenate(
            (
                [x_min] * (order - 1),
                np.linspace(x_min, x_max, n_bins + 1),
                [x_max] * (order - 1),
            )
        )
        super().__init__(len(self._knots))
        splines = []
        for row in np.identity(len(self._knots)):
            splines.append(scipy.interpolate.BSpline(self._knots, row, order))

        self._splines = splines
        x = np.linspace(x_min, x_max, 1000)
        norms = []
        for spline in splines:
            norms.append(np.trapz(scipy.interpolate.splev(x, spline) ** 2, x))
        self._norms = np.array(norms)

    def _internal_score(self, value):
        for i, spline in enumerate(self._splines):
            self._coeffs += scipy.interpolate.splev(value, spline)

    def normalize(self):
        norms = np.array([n if n > 0 else 1.0 for n in self._norms])
        return np.array(self._coeffs) / (self._norms * self._n)

    def plot(self):
        coeffs = self.normalize()
