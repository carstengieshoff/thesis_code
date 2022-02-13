from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

from embeddings.lag_emebedding import Embedding


class RecurrencePlot:
    def __init__(self, signal: np.array, metric: Callable[[np.array, np.array], float], embedding: Embedding):
        self.original_signal = signal.copy()
        self.embedded_signal = embedding.embedd(self.original_signal)
        self.metric = metric

        self.num_data_points, *self.shape_data_points = self.embedded_signal.shape
        self._recurrence_plot = np.zeros(shape=(self.num_data_points, self.num_data_points))

    def generate(self, normalize: bool = True) -> np.array:
        """Generating a RP from the provided signal according to the specifications.

        This creates only symmetric unthresholded RPs.
        """
        self._recurrence_plot = calculate_rp(
            signal=self.embedded_signal, num_data_points=self.num_data_points, metric=self.metric
        )

        self._recurrence_plot = self._recurrence_plot[::-1, :]

        if normalize:
            self._recurrence_plot /= np.max(self._recurrence_plot)

        return self._recurrence_plot

    def show(self, *args: Any, **kwargs: Any) -> None:
        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(self._recurrence_plot, *args, **kwargs)
        fig.colorbar(pos, ax=ax)
        plt.show()

    def hist(self, *args: Any, **kwargs: Any) -> None:
        fig, ax = plt.subplots(1, 1)
        ax.hist(self._recurrence_plot.flatten(), *args, **kwargs)
        plt.show()


@jit(nopython=True)
def calculate_rp(signal: np.array, num_data_points: int, metric: Callable[[np.array, np.array], float]) -> np.array:
    rp = np.zeros(shape=(num_data_points, num_data_points))

    for i in range(num_data_points):
        for j in range(i, num_data_points):
            dist = metric(signal[i, :], signal[j, :])
            rp[i, j] = dist
            rp[j, i] = dist

    return rp


if __name__ == "__main__":
    import time

    from embeddings.lag_emebedding import LagEmbedding
    from signals.artificial_signals import Sinusoid

    sinusoid = Sinusoid(frequency=5, sampling_rate=200, sec=5)
    sinusoid_signal = sinusoid.generate()
    embedding = LagEmbedding(dim=2, lag=2)

    @jit(nopython=True)
    def euclidean_dist(x: np.array, y: np.array) -> float:
        dist: float = np.linalg.norm(x - y)
        return dist

    rp = RecurrencePlot(signal=sinusoid_signal, embedding=embedding, metric=euclidean_dist)
    rp2 = RecurrencePlot(signal=sinusoid_signal, embedding=embedding, metric=euclidean_dist)

    start = time.time()
    rp.generate()
    end = time.time()
    print("Elapsed = %s" % (end - start))
    rp.show()

    start = time.time()
    rp2.generate()
    end = time.time()
    print("Elapsed = %s" % (end - start))
    rp2.show()

    start = time.time()
    rp.generate()
    end = time.time()
    print("Elapsed = %s" % (end - start))
    rp.show()
