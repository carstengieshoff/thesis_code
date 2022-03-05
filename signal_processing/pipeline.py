from typing import Any, List

import numpy as np
from tqdm import tqdm

from data_handling.data_reader import DataPoint
from embeddings.utils import fnn, mutual_information
from visualizations import plot_hist2d


class SignalProcessingPipeline:
    def __init__(self, dataset: List[DataPoint], Fs: int):
        self.dataset = dataset.copy()
        self.Fs = Fs
        self.description = ""

    def get_ebedding_info(self, num_samples: int = 0, *args: Any, **kwargs: Any) -> None:
        # TODO: Add option to plot this for any group of labels
        import random

        if num_samples > 0:
            sample = random.sample(self.dataset, k=num_samples)
        else:
            sample = self.dataset

        lags = []
        dims = []
        for x, y in tqdm(sample, total=len(sample)):
            lag = mutual_information(signal=x)
            dim = fnn(signal=x, lag=lag)

            lags.append(lag)
            dims.append(dim)
        lags = np.array(lags, dtype="uint8")
        dims = np.array(dims, dtype="uint8")

        plot_hist2d(lags, dims, *args, **kwargs)

    def _check_all_signals_same_shape(self) -> bool:
        """Check if all signals in `dataset` have the same length."""
        shapes = set(dp.x.shape for dp in self.dataset)
        return len(shapes) == 1
