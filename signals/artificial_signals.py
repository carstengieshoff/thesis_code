from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp


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


if __name__ == "__main__":
    c = Chirp(frequency_start=1, frequency_end=10, sampling_rate=100, sec=5, noise_rate=0.5)
    c.show()
