from itertools import product
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from embeddings.lag_emebedding import Embedding


class RecurrencePlot:
    def __init__(self, signal: np.array, metric: Callable[[np.array, np.array], float], embedding: Embedding):
        self.original_signal = signal.copy()
        self.embedded_signal = embedding.embedd(self.original_signal)
        self.metric = metric

        self.num_data_points, *self.shape_data_points = self.embedded_signal.shape
        self._recurrence_plot = np.zeros(shape=(self.num_data_points, self.num_data_points))

    def generate(self, normalize: bool = True) -> np.array:
        for i, j in tqdm(product(range(self.num_data_points), range(self.num_data_points))):
            self._recurrence_plot[i, j] = self.metric(self.embedded_signal[i, :], self.embedded_signal[j, :])

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


if __name__ == "__main__":
    from embeddings.lag_emebedding import LagEmbedding
    from signals.artificial_signals import Sinusoid

    sinusoid = Sinusoid(frequency=5, sampling_rate=100, sec=1)
    sinusoid_signal = sinusoid.generate()
    embedding = LagEmbedding(dim=2, lag=2)

    def euclidean_dist(x: np.array, y: np.array) -> float:
        dist: float = np.linalg.norm(x - y)
        return dist

    rp = RecurrencePlot(signal=sinusoid_signal, embedding=embedding, metric=euclidean_dist)

    rp.generate()
    rp.show()
