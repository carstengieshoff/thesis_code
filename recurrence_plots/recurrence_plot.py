from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

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
        reshaped_signal = self.embedded_signal.reshape(self.num_data_points, -1)

        for i in range(self.num_data_points):
            dist = self.metric(reshaped_signal[i, :], reshaped_signal)
            self._recurrence_plot[:, i] = dist

        self._recurrence_plot = self._recurrence_plot[::-1, :]

        if normalize:
            self._recurrence_plot = (self._recurrence_plot - np.min(self._recurrence_plot)) / (
                np.max(self._recurrence_plot) - np.min(self._recurrence_plot)
            )

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
    import time

    from embeddings.lag_emebedding import LagEmbedding
    from embeddings.utils.fnn import fnn
    from embeddings.utils.mutual_information import mutual_information
    from metrics import cosine_dist
    from signals.artificial_signals import Sinusoid

    sinusoid = Sinusoid(frequency=1, sampling_rate=200, sec=5, noise_rate=0.1)
    sinusoid_signal = sinusoid.generate()
    sinusoid.show()
    lag = mutual_information(signal=sinusoid_signal)
    dim = fnn(signal=sinusoid_signal, lag=lag)
    embedding = LagEmbedding(dim=dim, lag=lag)

    rp = RecurrencePlot(signal=sinusoid_signal, embedding=embedding, metric=cosine_dist)

    start = time.time()
    rp.generate()
    end = time.time()
    print("Elapsed = %s" % (end - start))
    rp.show()

    start = time.time()
    rp.generate(normalize=False)
    end = time.time()
    print("Elapsed = %s" % (end - start))
    rp.show()
