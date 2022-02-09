from itertools import product
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class RecurrencePlot:
    def __init__(self, signal: np.array, metric: Callable[[np.array, np.array], float]):
        self.original_signal = signal.copy()
        self.metric = metric

        self.num_data_points, *self.shape_data_points = self.original_signal.shape
        self.recurrence_plot = np.zeros(shape=(self.num_data_points, self.num_data_points))

    def generate(self) -> np.array:
        for i, j in tqdm(product(range(self.num_data_points, self.num_data_points))):
            self.recurrence_plot[i, j] = self.metric(self.original_signal[i, :], self.original_signal[j, :])

        return self.recurrence_plot

    def show(self) -> None:
        if self.recurrence_plot.abs().sum() == 0:
            raise RuntimeError("Run `.generate` first to trigger calculation")
        plt.imshow(self.recurrence_plot)


if __name__ == "__main__":
    from signals.artificial_signals import Sinusoid

    sinusoid = Sinusoid(frequency=1, sampling_rate=100, sec=2)
