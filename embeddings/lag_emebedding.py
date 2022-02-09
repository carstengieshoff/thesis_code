from abc import ABC, abstractmethod

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Embedding(ABC):
    @abstractmethod
    def embedd(self, signal: np.array) -> np.array:
        pass


class LagEmbedding(Embedding):
    def __init__(self, dim: int, lag: int):
        self._dim = dim
        self._lag = lag

    def embedd(self, signal: np.array) -> np.array:

        embedded_signal = sliding_window_view(signal, window_shape=self._dim * self._lag, axis=0).squeeze()
        embedded_signal = embedded_signal[:, 0 :: self._lag]

        return embedded_signal


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from signals.artificial_signals import Sinusoid

    sinusoid = Sinusoid(frequency=1, sampling_rate=100, sec=2)
    sinusoid_signal = sinusoid.generate()

    plt.plot(LagEmbedding(dim=2, lag=10).embedd(sinusoid_signal))
    plt.show()
