from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


class Kernel(ABC):
    @abstractmethod
    def generate(self, xa: np.array, xb: np.array) -> np.array:
        pass

    def plot_kernel(self, xa: np.array, xb: np.array) -> np.array:
        sigma = self.generate(xa=xa, xb=xb)

        plt.figure(figsize=(6, 6))
        plt.imshow(sigma)
        plt.show()

    def __add__(self, other: Kernel) -> Kernel:
        return AddedKernel(kernel_a=self, kernel_b=other)

    def __mul__(self, other: Kernel) -> Kernel:
        return MultipliedKernel(kernel_a=self, kernel_b=other)


class AddedKernel(Kernel):
    def __init__(self, kernel_a: Kernel, kernel_b: Kernel) -> None:
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b

    def generate(self, xa: np.array, xb: np.array) -> np.array:
        return self.kernel_a.generate(xa=xa, xb=xb) + self.kernel_b.generate(xa=xa, xb=xb)


class MultipliedKernel(Kernel):
    def __init__(self, kernel_a: Kernel, kernel_b: Kernel) -> None:
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b

    def generate(self, xa: np.array, xb: np.array) -> np.array:
        return self.kernel_a.generate(xa=xa, xb=xb) * self.kernel_b.generate(xa=xa, xb=xb)


class ExponentiatedQuadratic(Kernel):
    def __init__(self, sigma: float = 1.0, L: float = 1.0):

        self.sigma = sigma
        self.L = L

    def generate(self, xa: np.array, xb: np.array) -> np.array:
        sq_norm = -0.5 * cdist(xa, xb, "sqeuclidean") * self.L ** (-2)
        return self.sigma * np.exp(sq_norm)


class RationalQuadratic(Kernel):
    def __init__(self, sigma: float = 1.0, L: float = 1.0, alpha: float = 0.5):

        self.sigma = sigma
        self.L = L
        self.alpha = alpha

    def generate(self, xa: np.array, xb: np.array) -> np.array:
        sq_norm = 1 + cdist(xa, xb, "sqeuclidean") * (self.L**2 * 2 * self.alpha) ** (-1)
        return self.sigma * (sq_norm) ** (-self.alpha)


class RationalPeriodic(Kernel):
    def __int__(self, period: float = 4, sigma: float = 1.0, L: float = 1.0) -> None:

        self.period = period
        self.sigma = sigma
        self.L = L

    def generate(self, xa: np.array, xb: np.array) -> np.array:
        sq_norm = -2 * self.L ** (-2) * (np.sin(2 * np.pi * cdist(xa, xb, "minkowski", p=1) * self.period) ** 2)
        return self.sigma * np.exp(sq_norm)
