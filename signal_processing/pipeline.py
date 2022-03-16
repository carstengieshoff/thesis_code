import logging
from typing import List, Optional

import numpy as np
from scipy.signal import filtfilt, resample
from tqdm import tqdm

from data_handling.data_reader import DataPoint
from signal_processing.fixed_window_signal_splitting import split_signal
from signal_processing.qrs_cancellation import QRSEstimator
from signal_processing.r_peak_detection import get_r_peaks


class SignalProcessingPipeline:
    def __init__(self, dataset: List[DataPoint], Fs: int):
        self.dataset = dataset.copy()
        self.Fs = Fs
        self.description = ""

    def filter_signals(self, b: np.array, a: np.array) -> None:

        ds_new: List[DataPoint] = []
        for x, y in tqdm(self.dataset, total=len(self.dataset)):
            x_new = filtfilt(b, a, x, axis=0)
            ds_new.append(DataPoint(x_new, y))

        self.dataset = ds_new

    def change_labels(self, keep_n: Optional[int] = None) -> None:
        new_labels = dict()

        labels = set([dp.y for dp in self.dataset])
        new_labels = {label: i for i, label in enumerate(labels)}

        keep_counter = {label: 0 for label in new_labels.keys()}
        ds_new = []
        for x, y in tqdm(self.dataset, total=len(self.dataset)):
            if keep_n is not None:
                keep_counter[y] += 1
                if keep_counter[y] > keep_n:
                    continue

            ds_new.append(DataPoint(x, new_labels[y]))

        self.dataset = ds_new

    def resample(self, new_size: int) -> None:

        ds_new: List[DataPoint] = []
        for x, y in tqdm(self.dataset, total=len(self.dataset)):
            x_new = resample(x, new_size, axis=0)
            ds_new.append(DataPoint(x_new, y))

        self.dataset = ds_new

    def remove_qrs(self, qrs_estimator: QRSEstimator) -> None:

        ds_new: List[DataPoint] = []
        excluded = 0
        for x, y in tqdm(self.dataset, total=len(self.dataset)):
            try:
                qrs_locs = get_r_peaks(x[:, 0], self.Fs)
                x_new = qrs_estimator.reconstruct(x, qrs_locs)
                ds_new.append(DataPoint(x_new, y))
            except IndexError:
                excluded += 1

            if excluded > 0:
                logging.info(f"{excluded} signals were excluded due to issues in determining r-peaks")

        self.dataset = ds_new

    def normalize(self, with_std: bool = False) -> None:
        ds_new: List[DataPoint] = []
        for x, y in tqdm(self.dataset, total=len(self.dataset)):
            x_new = x - x.mean(axis=0)
            if with_std:
                x_new = x_new / x_new.std(axis=0)
            ds_new.append(DataPoint(x_new, y))

        self.dataset = ds_new

    def split_signals(self, back: int, front: int) -> None:
        excluded = 0
        ds_new: List[DataPoint] = []
        for x, y in tqdm(self.dataset, total=len(self.dataset)):
            try:
                qrs_locs = get_r_peaks(x[:, 0], self.Fs)
                windowed_signal = split_signal(signal=x, r_peaks=qrs_locs, back=back, front=front)
                for window in range(windowed_signal.shape[1]):
                    ds_new.append(DataPoint(windowed_signal[:, window, :].T, y))
            except IndexError:
                excluded += 1

                if excluded > 0:
                    logging.info(f"{excluded} signals were excluded due to issues in determining r-peaks")

        self.dataset = ds_new

    def filter_signal(self) -> None:
        pass

    def log_message(self) -> None:
        pass

    def save_dataset(self) -> None:
        pass

    def _check_all_signals_same_shape(self) -> bool:
        """Check if all signals in `dataset` have the same length."""
        shapes = set(dp.x.shape for dp in self.dataset)
        return len(shapes) == 1
