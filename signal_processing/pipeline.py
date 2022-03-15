from typing import Any, List

import numpy as np
from scipy.signal import resample
from tqdm import tqdm

from data_handling.data_reader import DataPoint
from embeddings.utils import fnn, mutual_information
from recurrence_plots import RecurrencePlotCalculator
from signal_processing import QRSEstimator, get_r_peaks
from visualizations import plot_hist2d


class SignalProcessingPipeline:
    def __init__(self, dataset: List[DataPoint], Fs: int):
        self.dataset = dataset.copy()
        self.Fs = Fs
        self.description = ""

    def calc_rps(
        self, rp_calculator: RecurrencePlotCalculator, normalize: bool = True, *args: Any, **kwargs: Any
    ) -> None:
        self.description += "_" + str(rp_calculator)
        ds_new: List[DataPoint] = []
        for x, y in tqdm(self.dataset, total=len(self.dataset)):
            rp = rp_calculator.generate(signal=x)
            if normalize:
                rp.normalize()
            ds_new.append(DataPoint(np.expand_dims(rp.get_rp(*args, **kwargs), axis=0), y))

        self.dataset = ds_new

    def resample(self, new_size: int) -> None:

        ds_new = []
        for x, y in tqdm(self.dataset, total=len(self.dataset)):
            x_new = resample(x, new_size, axis=0)
            ds_new.append(DataPoint(x_new, y))

        self.dataset = ds_new

    def remove_qrs(self, qrs_estimator: QRSEstimator) -> None:

        ds_new: List[DataPoint] = []
        for x, y in tqdm(self.dataset, total=len(self.dataset)):
            try:
                qrs_locs = get_r_peaks(x[:, 0], self.Fs)
                _, b = qrs_estimator(x, qrs_locs)
                # for window in range(b.shape[1]):
                window = 0
                ds_new.append(DataPoint(b[:, window, :].T, y))
            except IndexError:
                pass

        self.dataset = ds_new

    def filter_signal(self) -> None:
        pass

    def log_message(self) -> None:
        pass

    def save_dataset(self) -> None:
        pass

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

        plot_hist2d(lags, dims, ylabel="lag", xlabel="dims", figsize=(20, 20))

    def _check_all_signals_same_shape(self) -> bool:
        """Check if all signals in `dataset` have the same length."""
        shapes = set(dp.x.shape for dp in self.dataset)
        return len(shapes) == 1
