from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq


def plot_fft(x: np.array, Fs: int, *args: Any, **kwargs: Any) -> None:
    signal_len = x.shape[0]

    xf = fftfreq(signal_len, 1 / Fs)[: signal_len // 2]
    freqs = fft(x)
    max_freq = np.argmax(freqs[: signal_len // 2]) / signal_len * Fs

    plt.figure(*args, **kwargs)
    ax = plt.subplot()

    ax.plot(xf, 2.0 / signal_len * np.abs(freqs[: signal_len // 2]))
    ax.grid()
    ax.set_title(f"FFT, Fs={Fs}, max at {max_freq} [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_xlabel("Frequency [Hz]")

    plt.show()
