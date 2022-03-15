from __future__ import annotations

import random
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from data_handling.data_reader import DataPoint
from embeddings.utils import fnn, mutual_information
from recurrence_plots import RecurrencePlotCalculator
from visualizations import plot_hist2d, plot_rp


class Dataset:
    def __init__(
        self,
        dataset: List[DataPoint],
        rps: Optional[np.array] = None,
    ):
        self._signals, self._labels = self._list_data_to_arrays(dataset=dataset)
        self._recurrence_plots: Optional[np.array] = rps

    def calc_rps(
        self, rp_calculator: RecurrencePlotCalculator, normalize: bool = True, *args: Any, **kwargs: Any
    ) -> None:
        rp_shape = rp_calculator.generate(np.zeros(shape=self.signal_shape)).get_rp().shape

        self._recurrence_plots = np.zeros(shape=(self.__len__(), 1, *rp_shape))
        for i, x in enumerate(tqdm(self._signals, total=len(self._signals))):
            rp = rp_calculator.generate(signal=x)
            if normalize:
                rp.normalize()

            self._recurrence_plots[i, :, :, :] = np.expand_dims(rp.get_rp(*args, **kwargs), axis=0)

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

    def train_test_split(self, percentage: float) -> Tuple[Dataset, Dataset]:
        num_test = int(self.__len__() * percentage)
        test_idxs = random.sample(list(range(self.__len__())), k=num_test)
        train_idxs = [idx for idx in list(range(self.__len__())) if idx not in test_idxs]
        ds_test = [DataPoint(self._signals[idx], self._labels[idx]) for idx in test_idxs]
        ds_train = [DataPoint(self._signals[idx], self._labels[idx]) for idx in train_idxs]
        if self._recurrence_plots is None:
            return Dataset(dataset=ds_train), Dataset(dataset=ds_test)
        else:
            return Dataset(dataset=ds_train, rps=self._recurrence_plots[train_idxs]), Dataset(
                dataset=ds_test, rps=self._recurrence_plots[test_idxs]
            )

    def _list_data_to_arrays(self, dataset: List[DataPoint]) -> Tuple[np.array, np.array]:
        x, y = list(map(list, zip(*dataset)))
        return np.stack(x), np.stack(y)

    def plot(self, item: int, *args: Any, **kwargs: Any) -> None:
        fig = plt.figure(*args, **kwargs)

        ax = plt.subplot(121)
        ax.plot(self._signals[item])
        ax.set_title("Signal")

        if self._recurrence_plots is not None:
            ax = plt.subplot(122)
            plot_rp(self._recurrence_plots[item].squeeze(), ax=ax, fig=fig)
            ax.set_title("RP")

        plt.suptitle(f"Instance of category {self._labels[item]}")
        plt.show()

    def get_embedding_info(self, num_samples: int = 0, *args: Any, **kwargs: Any) -> None:
        # TODO: Add option to plot this for any group of labels
        import random

        if num_samples > 0:
            sample_idxs = random.sample(list(range(self._signals.shape[0])), k=num_samples)
            samples = self._signals[sample_idxs]
        else:
            samples = self._signals

        lags = []
        dims = []
        for x in tqdm(samples, total=len(samples)):
            lag = mutual_information(signal=x)
            dim = fnn(signal=x, lag=lag)

            lags.append(lag)
            dims.append(dim)
        lags = np.array(lags, dtype="uint8")
        dims = np.array(dims, dtype="uint8")

        plot_hist2d(a=lags, b=dims, ylabel="lag", xlabel="dims")

    def __getitem__(self, item: int) -> Tuple[torch.tensor, torch.tensor]:
        if self._recurrence_plots is None:
            return self._signals[item], self._labels[item]
        else:
            return torch.from_numpy(self._recurrence_plots[item]), torch.tensor(self._labels[item])

    def __len__(self) -> int:
        assert self._signals.shape[0] == self._labels.shape[0], "Number of signals and labels does not match"
        return int(self._labels.shape[0])


if __name__ == "__main__":
    from embeddings import LagEmbedding

    N = 200
    test_raw_data = [DataPoint(np.random.rand(100, 4), np.random.randint(0, 4)) for _ in range(N)]

    ds = Dataset(test_raw_data)
    print(ds.signals.shape)
    print(ds.signal_shape)
    print(ds.labels.shape)
    print(len(ds))

    ds_train, ds_test = ds.train_test_split(0.25)
    print(len(ds_train), len(ds_test))

    embedding = LagEmbedding(dim=1, lag=5)
    calculator = RecurrencePlotCalculator(embedding=embedding, metric="cosine")

    ds.calc_rps(rp_calculator=calculator)
    ds.plot(0)
