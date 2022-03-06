from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

from data_handling.data_reader import DataPoint


class SignalDataset:
    def __init__(self, dataset: List[DataPoint]):
        self._signals, self._labels = self._list_data_to_arrays(dataset=dataset)
        self._recurrence_plots: Optional[np.array] = None

    @property
    def labels(self) -> np.array:
        return self._labels

    @property
    def signals(self) -> np.array:
        return self._signals

    @property
    def signal_shape(self) -> np.array:
        return self._signals[0].shape

    @property
    def recurrence_plots(self) -> np.array:
        if self._recurrence_plots is None:
            raise RuntimeError("Recurrence Plots have not yet been calculated")
        return self._recurrence_plots

    def train_test_split(self, percentage: float) -> Tuple[SignalDataset, SignalDataset]:
        num_test = int(self.__len__() * percentage)
        test_idxs = random.sample(list(range(self.__len__())), k=num_test)
        train_idxs = [idx for idx in list(range(self.__len__())) if idx not in test_idxs]
        ds_test = [DataPoint(self._signals[idx], self._labels[idx]) for idx in test_idxs]
        ds_train = [DataPoint(self._signals[idx], self._labels[idx]) for idx in train_idxs]
        return SignalDataset(dataset=ds_train), SignalDataset(dataset=ds_test)

    def _list_data_to_arrays(self, dataset: List[DataPoint]) -> Tuple[np.array, np.array]:
        x, y = list(map(list, zip(*dataset)))
        return np.stack(x), np.stack(y)

    def __getitem__(self, item: int) -> Tuple[np.array, int]:
        return self._signals[item], self._labels[item]

    def __len__(self) -> int:
        assert self._signals.shape[0] == self._labels.shape[0], "Number of signals and labels does not match"
        return int(self._labels.shape[0])


if __name__ == "__main__":
    N = 200
    test_raw_data = [DataPoint(np.random.rand(100, 4), np.random.randint(0, 4)) for _ in range(N)]

    ds = SignalDataset(test_raw_data)
    print(ds.signals.shape)
    print(ds.signal_shape)
    print(ds.labels.shape)
    print(len(ds))

    ds_train, ds_test = ds.train_test_split(0.25)
    print(len(ds_train), len(ds_test))
