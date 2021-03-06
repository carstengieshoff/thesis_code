from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from skimage.transform import resize
from tqdm import tqdm

from embeddings.lag_emebedding import Embedding
from recurrence_plots.utils import image_histogram_equalization
from visualizations import plot_rp

CDIST_OPTIONS = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulsinski",
    "kulczynski1",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]


def normalize(data: np.array) -> np.array:
    """Linear transform all distances in the RP to the interval (0,1)."""
    data /= data.max()
    return data


class RecurrencePlot:
    def __init__(self, rp_data: np.ndarray, info: Optional[str] = None):
        self._unthresholded_rp = rp_data
        self._info = info

    def normalize(self) -> RecurrencePlot:
        """Linear transform all distances in the RP to the interval (0,1)."""
        self._unthresholded_rp = normalize(self._unthresholded_rp)
        return self

    def sigmoid(self) -> RecurrencePlot:
        self.normalize()
        self._unthresholded_rp = 1 / (1 + np.exp(-1 * (self._unthresholded_rp + 0.5)))
        self.normalize()
        return self

    def square(self) -> RecurrencePlot:
        self.normalize()
        self._unthresholded_rp = self._unthresholded_rp**2
        self.normalize()
        return self

    def sqrt(self) -> RecurrencePlot:
        self.normalize()
        self._unthresholded_rp = self._unthresholded_rp**0.5
        self.normalize()
        return self

    def hist_eq(self) -> RecurrencePlot:
        self._unthresholded_rp = image_histogram_equalization(self._unthresholded_rp)
        return self

    def get_rp(
        self,
        threshold: Optional[str] = None,
        epsilon: Optional[float] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        if threshold is None:
            rp_data = self._unthresholded_rp
            if size is not None:
                rp_data = normalize(resize(rp_data, size))
            return rp_data
        else:
            if epsilon is None:
                raise RuntimeError("For the option `thresholded` a `epsilon` must be specified.")

            if isinstance(epsilon, float) and not (0 < epsilon < 1):
                raise ValueError(f"`epsilon` must lie in (0, 1), got {epsilon}")

            self.normalize()
            if threshold == "absolute":
                mask = self._unthresholded_rp <= epsilon
            elif threshold == "relative":
                mask = self._unthresholded_rp <= np.quantile(self._unthresholded_rp, epsilon)
            else:
                raise ValueError(
                    f"Unrecognized value for `threshold`, expected one of {', '.join(['relative', 'absolute'])}"
                )

            thresholded_rp = np.zeros_like(self._unthresholded_rp)
            thresholded_rp[mask] = 1

            if size:
                thresholded_rp = normalize(resize(thresholded_rp, size))

            return thresholded_rp

    def show(
        self,
        threshold: Optional[str] = None,
        epsilon: Optional[float] = None,
        size: Optional[Tuple[int, int]] = (512, 512),
        *args: Any,
        **kwargs: Any,
    ) -> None:
        rp_data = self.get_rp(threshold=threshold, epsilon=epsilon, size=size)
        # type ignore : See https://github.com/python/mypy/issues/6799
        plot_rp(rp_data=rp_data, *args, **kwargs)  # type: ignore

    def hist(self, threshold: Optional[str] = None, epsilon: Optional[float] = None, *args: Any, **kwargs: Any) -> None:
        fig, ax = plt.subplots(1, 1)
        ax.hist(self.get_rp(threshold=threshold, epsilon=epsilon).flatten(), *args, **kwargs)
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

    def __init__(
        self,
        metric: Union[Callable[[np.array, np.array], np.ndarray], str],
        embedding: Embedding,
        dtype: str = "float16",
    ):
        self.embedding = embedding
        self.metric = metric
        self._dtype = dtype

    def generate(self, signal: np.array) -> RecurrencePlot:
        """Generating a RP from the provided signal according to the specifications.

        This creates only symmetric unthresholded RPs.

        Args:
            signal: `np.array` of size (len_signal, dim_signal) representing a 1D or 2D signal.
        """
        embedded_signal = self.embedding.embedd(signal)
        reshaped_signal = embedded_signal.reshape(embedded_signal.shape[0], -1)

        recurrence_plot = cdist(reshaped_signal, reshaped_signal, metric=self.metric).astype(self._dtype)

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
    # from embeddings.utils.fnn import fnn
    # from embeddings.utils.mutual_information import mutual_information
    from signals.artificial_signals import AAGP
    from signals.GP_kernels import organized_aa_args

    s = AAGP(organized_aa_args, sampling_rate=200, sec=4)

    signal = s.generate()
    # s.show()

    # lag = mutual_information(signal=signal)
    # dim = fnn(signal=signal, lag=lag)
    embedding = LagEmbedding(dim=1, lag=1)
    calculator = RecurrencePlotCalculator(embedding=embedding, metric="euclidean")
    # print(calculator)

    start = time.time()
    for i in range(30):
        rp = calculator.generate(signal=signal)
        rp.get_rp(threshold="relative", epsilon=0.2)
    end = time.time()
    print("Elapsed = %s" % (end - start))
    rp.show()
    # fig = plt.figure(figsize=(15, 8))
#
# ax = plt.subplot(212)
# ax.plot(signal)
# ax.set_title("Signal", fontsize="large")
#
# ax = plt.subplot(231)
# plot_rp(rp.get_rp(), fig=fig, ax=ax, colorbar=True)
# ax.set_title("Unthresholed RP", fontsize="large")
#
# ax = plt.subplot(232)
# plot_rp(rp.get_rp(threshold="relative", epsilon=0.2), fig=fig, ax=ax, cmap="binary")
# ax.set_title("Thresholed RP (eps= 0.2)", fontsize="large")
#
# ax = plt.subplot(233)
# plot_rp(rp.get_rp(threshold="relative", epsilon=0.01), fig=fig, ax=ax, cmap="binary")
# ax.set_title("Thresholed RP (eps= 0.01)", fontsize="large")
#
# plt.savefig("./rp_example")
# plt.show()
#
# print(rp)
