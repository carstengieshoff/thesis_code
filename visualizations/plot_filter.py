from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.signal import freqz


def plot_filter(
    b: np.array,
    a: np.array,
    Fs: int,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    *args: Any,
    **kwargs: Any,
) -> None:
    freq, h = freqz(b, a)

    if ax is None or fig is None:
        fig_and_ax_not_given = True
        fig, ax = plt.subplots(*args, **kwargs)
    else:
        fig_and_ax_not_given = False

    # Plot magnitude response of the filter
    ax.plot(freq * Fs / (2 * np.pi), 20 * np.log10(abs(h)), "r", label="Bandpass filter", linewidth="2")

    ax.set_xlabel("Frequency [Hz]", fontsize=15)
    ax.set_ylabel("Magnitude [dB]", fontsize=15)

    if title:
        plt.title("title", fontsize=15)

    plt.grid()

    if fig_and_ax_not_given:
        plt.show()


if __name__ == "__main__":
    from scipy.signal import cheby2

    [b, a] = cheby2(3, 20, [3, 100], btype="bandpass", fs=512)

    plot_filter(b, a, 500)

    print("Done")
