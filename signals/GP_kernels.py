from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

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
        super().__init__()
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b

    def generate(self, xa: np.array, xb: np.array) -> np.array:
        super().__init__()
        return self.kernel_a.generate(xa=xa, xb=xb) + self.kernel_b.generate(xa=xa, xb=xb)


class MultipliedKernel(Kernel):
    def __init__(self, kernel_a: Kernel, kernel_b: Kernel) -> None:
        super().__init__()
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b

    def generate(self, xa: np.array, xb: np.array) -> np.array:
        return self.kernel_a.generate(xa=xa, xb=xb) * self.kernel_b.generate(xa=xa, xb=xb)


class ExponentiatedQuadratic(Kernel):
    def __init__(self, sigma: float = 1.0, L: float = 1.0):
        super().__init__()
        self.sigma = sigma
        self.L = L

    def generate(self, xa: np.array, xb: np.array) -> np.array:
        sq_norm = -0.5 * cdist(xa, xb, "sqeuclidean") * self.L ** (-2)
        return self.sigma * np.exp(sq_norm)


class RationalQuadratic(Kernel):
    def __init__(self, sigma: float = 1.0, L: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.sigma = sigma
        self.L = L
        self.alpha = alpha

    def generate(self, xa: np.array, xb: np.array) -> np.array:
        sq_norm = 1 + cdist(xa, xb, "sqeuclidean") * (self.L**2 * 2 * self.alpha) ** (-1)
        return self.sigma * (sq_norm) ** (-self.alpha)


class RationalPeriodic(Kernel):
    def __init__(self, period: float = 4, sigma: float = 1.0, L: float = 1.0) -> None:
        super().__init__()
        self.period = period
        self.sigma = sigma
        self.L = L

    def generate(self, xa: np.array, xb: np.array) -> np.array:
        sq_norm = -2 * self.L ** (-2) * (np.sin(2 * np.pi * cdist(xa, xb, "minkowski", p=1) * self.period) ** 2)
        return self.sigma * np.exp(sq_norm)


unorganized_aa_args = {
    "lp1": 2.0,
    "lp2": 0.2,
    "l1": 0.2,
    "l2": 0.12,
    "p1": 6.6,
    "p2": 13,
    "freq": 80,
    "sigma": 0.7,
}

organized_aa_args = {
    "lp1": 2.0,
    "lp2": 0.2,
    "l1": 0.2,
    "l2": 0.12,
    "p1": 6.6,
    "p2": 13,
    "freq": 50,
    "sigma": 0.5,
}


class AAKernel(Kernel):
    def __init__(
        self,
        lp1: float = 2,
        lp2: float = 0.2,
        l1: float = 0.2,
        l2: float = 0.12,
        p1: float = 6.6,
        p2: float = 13,
        freq: int = 80,
        sigma: float = 0.7,
    ) -> None:
        super().__init__()
        self.lp1 = lp1
        self.lp2 = lp2
        self.l1 = l1
        self.l2 = l2
        self.p1 = p1 / 2
        self.p2 = p2 / 2
        self.freq = freq
        self.sigma = sigma

        self._kernel: Optional[Kernel] = None

    def generate(self, xa: np.array, xb: np.array) -> np.array:
        if self._kernel is None:

            kernel = ExponentiatedQuadratic(sigma=1.0, L=self.l1) * RationalPeriodic(
                period=self.p1, sigma=1, L=self.lp1
            )
            kernel = kernel + ExponentiatedQuadratic(sigma=self.sigma, L=self.l2) * RationalPeriodic(
                period=self.p2, sigma=1, L=self.lp2
            )
            self._kernel = kernel

        return self._kernel.generate(xa=xa, xb=xb)


class OrganizedAAKernel(AAKernel):
    def __init__(self) -> None:
        super().__init__(**organized_aa_args)  # type:  ignore


class UnorganizedAAKernel(AAKernel):
    def __init__(self) -> None:
        super().__init__(**unorganized_aa_args)  # type:  ignore
