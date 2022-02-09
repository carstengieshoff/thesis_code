import matplotlib.pyplot as plt
import numpy as np


class Sinusoid:
    def __init__(self, frequency: float, sampling_rate: int, sec: int):
        self._frequency = frequency
        self._sampling_rate = sampling_rate
        self._sec = sec

    def generate(self, retx: bool = False) -> np.array:
        num_points = self._sec * self._sampling_rate
        x = np.linspace(start=0, stop=self._sec, num=num_points)

        f_x = np.sin(2 * np.pi * self._frequency * x)

        if retx:
            return x, f_x
        else:
            return f_x

    def show(self) -> None:
        x, f_x = self.generate(retx=True)

        plt.plot(x, f_x, label=f"Freq: {self._frequency} Hz")
        plt.xlabel("Seconds")
        plt.show()


if __name__ == "__main__":
    sinusoid = Sinusoid(frequency=1, sampling_rate=100, sec=2)
    sinusoid.show()
