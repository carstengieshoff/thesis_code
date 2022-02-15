from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from embeddings.lag_emebedding import Embedding


class RecurrencePlot:
    def __init__(self, rp_data: np.ndarray, info: Optional[str] = None):
        self._unthresholded_rp = rp_data
        self._info = info

    def normalize(self) -> RecurrencePlot:
        """Linear transform all distances in the RP to the interval (0,1)."""
        min_dist = np.min(self._unthresholded_rp)
        max_dist = np.max(self._unthresholded_rp)
        self._unthresholded_rp = (self._unthresholded_rp - min_dist) / (max_dist - min_dist)
        return self

    def get_rp(self, thresholded: bool = False, epsilon: Optional[float] = None) -> np.ndarray:
        if not thresholded:
            return self._unthresholded_rp
        else:
            if epsilon is None:
                raise RuntimeError("For the option `thresholded` a `epsilon` must be specified.")

            if isinstance(epsilon, float) and not (0 < epsilon < 1):
                raise ValueError(f"`epsilon` must lie in (0, 1), got {epsilon}")

            self.normalize()
            mask = self._unthresholded_rp <= epsilon
            thresholded_rp = np.zeros_like(self._unthresholded_rp)
            thresholded_rp[mask] = 1

            return thresholded_rp

    def show(self, thresholded: bool = False, epsilon: Optional[float] = None, *args: Any, **kwargs: Any) -> None:
        fig, ax = plt.subplots(1, 1)
        pos = ax.imshow(self.get_rp(thresholded=thresholded, epsilon=epsilon), *args, **kwargs)
        fig.colorbar(pos, ax=ax)
        plt.show()

    def hist(self, thresholded: bool = False, epsilon: Optional[float] = None, *args: Any, **kwargs: Any) -> None:
        fig, ax = plt.subplots(1, 1)
        ax.hist(self.get_rp(thresholded=thresholded, epsilon=epsilon).flatten(), *args, **kwargs)
        plt.show()

    @property
    def info(self) -> str:
        if self._info is None:
            raise ValueError("No `info` is available")
        return self._info

    @property
    def shape(self) -> Tuple[int, int]:
        shape: Tuple[int, int] = self._unthresholded_rp.shape
        return shape

    def __str__(self) -> str:
        return self.__class__.__name__ + self.info


class RecurrencePlotCalculator:
    """Calculate RP from a signal according to `embedding`, and `metric`.

    Args:
        embedding: :class:`embeddings.Embedding` to embedd signals with.
        metric: String option or custom metric
         (see `cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_).
    """

    def __init__(self, metric: Union[Callable[[np.array, np.array], np.ndarray], str], embedding: Embedding):
        self.embedding = embedding
        self.metric = metric

    def generate(self, signal: np.array) -> RecurrencePlot:
        """Generating a RP from the provided signal according to the specifications.

        This creates only symmetric unthresholded RPs.

        Args:
            signal: `np.array` of size (len_signal, dim_signal) representing a 1D or 2D signal.
        """
        embedded_signal = self.embedding.embedd(signal)
        reshaped_signal = embedded_signal.reshape(embedded_signal.shape[0], -1)

        recurrence_plot = cdist(reshaped_signal, reshaped_signal, metric=self.metric)

        recurrence_plot = recurrence_plot[::-1, :]

        return RecurrencePlot(recurrence_plot, info=self.info)

    def generate_dataset(self, signals: List[np.array]) -> List[RecurrencePlot]:
        """Generating a RP from the each signal in `signals` according to the specifications.

        Args:
            signals: List of `np.array` of size (len_signal, dim_signal) representing a 1D or 2D signals.
        """
        num_signals = len(signals)
        recurrence_plots: List[RecurrencePlot] = []
        for signal in tqdm(signals, total=num_signals):
            recurrence_plots.append(self.generate(signal=signal))

        return recurrence_plots

    @property
    def info(self) -> str:
        return f"(metric='{self.metric}', embedding={self.embedding})"

    def __str__(self) -> str:
        return self.__class__.__name__ + self.info


if __name__ == "__main__":
    import time

    from embeddings.lag_emebedding import LagEmbedding
    from embeddings.utils.fnn import fnn
    from embeddings.utils.mutual_information import mutual_information
    from signals.artificial_signals import Sinusoid

    sinusoid = Sinusoid(frequency=3, sampling_rate=1000, sec=2, noise_rate=1)
    sinusoid_signal = sinusoid.generate()
    sinusoid.show()

    lag = mutual_information(signal=sinusoid_signal)
    dim = fnn(signal=sinusoid_signal, lag=lag)
    embedding = LagEmbedding(dim=dim, lag=lag)
    calculator = RecurrencePlotCalculator(embedding=embedding, metric="cosine")
    print(calculator)

    start = time.time()
    rp = calculator.generate(signal=sinusoid_signal)
    end = time.time()
    print("Elapsed = %s" % (end - start))
    rp.normalize()
    rp.show(cmap="afmhot")
    rp.hist()
    rp.show(thresholded=True, epsilon=0.2, cmap="Greys")
    rp.hist(thresholded=True, epsilon=0.2)
    print(rp)
