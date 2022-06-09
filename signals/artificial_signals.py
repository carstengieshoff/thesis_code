from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cheby2, chirp, filtfilt, iirnotch

from signals.GP_kernels import Kernel

af_type_a = {
    "f0": 6.0,
    "df": 0.2,
    "ff": 0.1,
    "M": 5,
    "al": np.array([150, 75, 45]),
    "dal": np.array([50, 25, 15]),
    "wa": 0.08,
}

af_type_b = {
    "f0": 8.0,
    "df": 0.3,
    "ff": 0.23,
    "M": 3,
    "al": np.array([60, 50, 40]),
    "dal": np.array([19, 15, 12]),
    "wa": 0.08,
}


class ArtificialSignal(ABC):
    """Generate an artificial signal."""

    def __init__(self, sampling_rate: int, sec: int, noise_rate: float = 0):
        self._sampling_rate = sampling_rate
        self._sec = sec
        self._x = np.linspace(start=0, stop=self._sec, num=self.__len__())
        self._noise_rate = noise_rate

        self._data: Optional[np.array] = None

    @abstractmethod
    def generate(self) -> np.array:
        pass

    def show(self, *args: Any, **kwargs: Any) -> None:
        if self._data is None:
            self.generate()
        plt.plot(self._x, self._data, *args, **kwargs)
        plt.xlabel("Seconds")
        plt.legend()
        plt.show()

    @staticmethod
    def add_noise(func: Callable[[Any], np.array]) -> Callable[[Any], np.array]:
        """Add mean zero gaussian noise with to output of wrapped function."""

        @wraps(func)
        def wrapper(self: ArtificialSignal, *args: Any, **kwargs: Any) -> np.array:
            output = func(self, *args, **kwargs)
            if self._noise_rate != 0:
                noise = np.random.normal(loc=0, scale=self._noise_rate, size=output.shape)
                output += noise
            return output

        return wrapper

    @property
    def data(self) -> np.array:
        """Generated data."""
        if self._data is None:
            raise RuntimeError("`.generate` needs to be run before data can be accessed")
        return self._data.copy()

    @property
    def sec(self) -> int:
        """Length of signal in seconds"""
        return self._sec

    @property
    def sampling_rate(self) -> int:
        """Sampling rate of the signal."""
        return self._sampling_rate

    @property
    def x(self) -> np.array:
        return self._x

    def __len__(self) -> int:
        return self._sec * self._sampling_rate


class Sinusoid(ArtificialSignal):
    """Create Sinusoid."""

    def __init__(self, frequency: float, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._frequency = frequency

    @ArtificialSignal.add_noise
    def generate(self) -> np.array:
        f_x = np.sin(2 * np.pi * self._frequency * self._x)
        self._data = f_x.reshape(-1, 1)
        return self._data

    def show(self, *args: Any, **kwargs: Any) -> None:
        super().show(label=f"Freq: {self._frequency} Hz", *args, **kwargs)


class GP(ArtificialSignal):
    """Create Gaussian process"""

    def __init__(
        self,
        kernel: Kernel,
        hp_filter_freq: Optional[float] = None,
        lp_filter_freq: Optional[float] = None,
        notch_filter_freq: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.kernel = kernel
        self.hp_filter_freq = hp_filter_freq
        self.lp_filter_freq = lp_filter_freq
        self.notch_filter_freq = notch_filter_freq

        X = np.expand_dims(self._x, 1)
        sigma = self.kernel.generate(X, X)
        U, S, V = np.linalg.svd(sigma, hermitian=True)
        self._sq_sigma = U @ np.diag(np.sqrt(S))

    @ArtificialSignal.add_noise
    def generate(self, num_samples: int = 1) -> np.array:

        f_x = np.random.standard_normal(size=(self._sec * self._sampling_rate, num_samples))
        f_x = self._sq_sigma @ f_x

        if self.hp_filter_freq is not None:
            [b, a] = cheby2(3, 20, self.hp_filter_freq, btype="highpass", fs=self.sampling_rate)
            f_x = filtfilt(b, a, f_x, axis=0)

        if self.lp_filter_freq is not None:
            [b, a] = cheby2(1, 10, self.lp_filter_freq, btype="lowpass", fs=self.sampling_rate)
            f_x = filtfilt(b, a, f_x, axis=0)

        if self.notch_filter_freq is not None:
            [b, a] = iirnotch(fs=self.sampling_rate, w0=self.notch_filter_freq, Q=20)
            f_x = filtfilt(b, a, f_x, axis=0)

        self._data = f_x.reshape(-1, num_samples)
        return self._data


class Chirp(ArtificialSignal):
    """Create Chirp"""

    def __init__(self, frequency_start: float, frequency_end: float, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._frequency_start = frequency_start
        self._frequency_end = frequency_end

    @ArtificialSignal.add_noise
    def generate(self) -> np.array:
        f_x = chirp(self._x, f0=self._frequency_start, f1=self._frequency_end, t1=self._sec)
        self._data = f_x.reshape(-1, 1)
        return self._data

    def show(self, *args: Any, **kwargs: Any) -> None:
        super().show(label=f"Freq: {self._frequency_start} Hz -> {self._frequency_end} Hz ", *args, **kwargs)


class AAStridh(ArtificialSignal):
    """Create simulated AF according to https://ieeexplore.ieee.org/document/900266."""

    def __init__(
        self,
        f0: float = 6.0,
        df: float = 0.2,
        ff: float = 0.1,
        M: int = 5,
        al: np.array = np.array([150, 75, 45]),
        dal: np.array = np.array([50, 25, 15]),
        wa: float = 0.08,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.f0 = f0
        self.df = df
        self.ff = ff
        self.M = M
        self.al = al
        self.dal = dal
        self.wa = wa

    @ArtificialSignal.add_noise
    def generate(self) -> np.array:
        x = np.arange(1, self.sec * self.sampling_rate + 1) / self.sampling_rate
        x = x.reshape(-1, 1)

        theta = 2 * np.pi * self.f0 * x + self.df / self.ff * np.sin(2 * np.pi * self.ff * x)
        self._data = -1 * sum(self._amplitude(i) * np.sin(i * theta) for i in range(1, self.M + 1))
        return self._data

    def _amplitude(self, i: int) -> np.array:
        x = np.arange(1, self.sec * self.sampling_rate + 1).reshape(-1, 1) / self.sampling_rate
        al = self.al.reshape(1, -1)
        dal = self.dal.reshape(1, -1)
        return 2 / (i * np.pi) * (al + dal * np.sin(2 * np.pi * self.wa * x))

    def show(self, *args: Any, **kwargs: Any) -> None:
        if self._data is None:
            self._data = self.generate()

        N = self._data.shape[1]
        fig, ax = plt.subplots(N, 1, figsize=(20, N * 2))
        for i in range(N):
            ax[i].plot(self._data[:, i])
        plt.suptitle("Simulated AF", fontsize="x-large")
        plt.show()


class Wavefront(ArtificialSignal):
    def __init__(
        self,
        freq_x: float = 0.1,
        freq_y: float = 0.2,
        freq_t: float = 10,
        size: Tuple[int, int] = (100, 100),
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._freq_x = freq_x
        self._freq_y = freq_y
        self._freq_t = freq_t
        self._size = size

    @ArtificialSignal.add_noise
    def generate(self) -> np.ndarray:
        signal = np.zeros(shape=(len(self._x), self._size[0], self._size[1]))
        for x_ in range(self._size[0]):
            for y_ in range(self._size[1]):
                signal[:, x_, y_] = np.sin(
                    2 * np.pi * self._freq_y * y_ + 2 * np.pi * self._freq_x * x_ + self._freq_t * self._x
                )

        self._data = signal
        return self._data

    def show(self, num_frames: int = 1, *args: Any, **kwargs: Any) -> None:
        if self._data is None:
            self._data = self.generate()
        for i in range(min(self._data.shape[0], num_frames)):
            plt.imshow(self._data[i, :, :], *args, **kwargs)
            plt.pause(1e-20)
        plt.show()


if __name__ == "__main__":
    pass
