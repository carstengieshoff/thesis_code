from abc import ABC, abstractmethod

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Embedding(ABC):
    """Embedding a signal in some different space."""

    @abstractmethod
    def embedd(self, signal: np.array) -> np.array:
        pass


class LagEmbedding(Embedding):
    """Embedding a signal by replacing each datapoint by a set of (past) data points."""

    def __init__(self, dim: int, lag: int):
        assert dim >= 1, "`dim` is expected to be a positive integer"
        self._dim = dim
        self._lag = lag

    def embedd(self, signal: np.array) -> np.array:

        if len(signal.shape) >= 3:
            raise NotImplementedError("LagEmbedding for 3d signals not yet implemented")

        if self._lag == 0 or self._dim == 1:
            return signal

        embedded_signal = sliding_window_view(signal, window_shape=self._dim * self._lag, axis=0).squeeze()

        if signal.shape[1] > 1:
            embedded_signal = embedded_signal[:, :, self._lag * (1 - self._dim) - 1 :: self._lag]
        else:
            embedded_signal = embedded_signal[:, self._lag * (1 - self._dim) - 1 :: self._lag]

        return embedded_signal.reshape(-1, signal.shape[1], self._dim).squeeze()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from signals.artificial_signals import Sinusoid

    sinusoid = Sinusoid(frequency=1, sampling_rate=100, sec=2)
    sinusoid_signal = sinusoid.generate()

    plt.plot(LagEmbedding(dim=2, lag=10).embedd(sinusoid_signal))
    plt.show()
