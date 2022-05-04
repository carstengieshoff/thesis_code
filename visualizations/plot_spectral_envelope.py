from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.fft import fftfreq

from embeddings.utils import spectral_envelope


def plot_spectral_envelope(
    x: np.array,
    Fs: int,
    h: Optional[np.array] = None,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    *args: Any,
    **kwargs: Any,
) -> None:
    signal_len = x.shape[0]

    xf = fftfreq(signal_len, 1 / Fs)[: signal_len // 2]
    freqs = spectral_envelope(x, h=h)
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
        f"Spectral Envelope, Fs={Fs}, max at {max_freq:.2f} [Hz] (lag = {np.ceil(signal_len/max_idx)})",
        fontsize="xx-large",
    )
    ax.set_ylabel("Magnitude [dB]", fontsize="x-large")
    ax.set_xlabel("Frequency [Hz]", fontsize="x-large")

    if fig_and_ax_not_given:
        plt.show()
