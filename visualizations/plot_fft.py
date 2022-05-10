from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.fft import fftfreq, rfft


def plot_fft(
    x: np.array,
    Fs: int,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    *args: Any,
    **kwargs: Any,
) -> None:
    signal_len = x.shape[0]

    xf = fftfreq(signal_len, 1 / Fs)[: signal_len // 2]
    freqs = np.abs(rfft(x, axis=0))
    if len(freqs.shape) > 1:
        max_idx = np.argmax(freqs[: signal_len // 2, 0])
        max_freq = max_idx / signal_len * Fs
    else:
        max_idx = np.argmax(freqs[: signal_len // 2])
        max_freq = max_idx / signal_len * Fs

    if ax is None or fig is None:
        fig_and_ax_not_given = True
        fig, ax = plt.subplots(*args, **kwargs)
        ax = plt.subplot()
    else:
        fig_and_ax_not_given = False

    ax.plot(xf, 2.0 / signal_len * np.abs(freqs[: signal_len // 2]))
    ax.grid()
    ax.set_title(
        f"FFT, Fs={Fs}, max at {max_freq:.2f} [Hz] (lag = {np.ceil(signal_len/max_idx)})",
        fontsize="xx-large",
    )
    ax.set_ylabel("Magnitude [dB]", fontsize="x-large")
    ax.set_xlabel("Frequency [Hz]", fontsize="x-large")

    if fig_and_ax_not_given:
        plt.show()


if __name__ == "__main__":
    import numpy as np

    from signals import Chirp, Sinusoid

    FS = 200
    sec = 1
    sin1 = Sinusoid(frequency=5, sampling_rate=FS, sec=sec, noise_rate=0.1)
    sin2 = Sinusoid(frequency=2, sampling_rate=FS, sec=sec, noise_rate=0.2)
    sin3 = Sinusoid(frequency=5, sampling_rate=FS, sec=sec, noise_rate=0.5)
    chirp = Chirp(frequency_start=1, frequency_end=2, sampling_rate=FS, sec=sec, noise_rate=0.1)

    sin1.generate()
    sin2.generate()
    sin3.generate()
    chirp.generate()

    data = np.vstack([sin1.data.T, sin2.data.T, sin3.data.T]).T

    plot_fft(data, Fs=200)
