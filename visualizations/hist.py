from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def plot_hist2d(a: np.array, b: np.array, *args: Any, **kwargs: Any) -> Axes:
    """Plot 2D histrogram.

    Args:
        a: np.array of shape (len_a,)
        b: np.array of shape (len_b,)
    """
    # Todo: Add margin distrubutions
    # TODO: Log argmax(Hist)
    a_bins = np.arange(-1, max(a) + 1) + 0.5
    b_bins = np.arange(-1, max(b) + 1) + 0.5
    hist, xedges, yedges = np.histogram2d(a, b, bins=(a_bins, b_bins))

    fig, ax = plt.subplots(1, 1, *args, **kwargs)
    pos = ax.imshow(hist)
    ax.set_ylabel("lag")
    ax.set_xlabel("dim")
    ax.set_yticks((xedges[:-1] + xedges[1:]) / 2)
    ax.set_xticks((yedges[:-1] + yedges[1:]) / 2)
    fig.colorbar(pos, ax=ax)
    plt.show()
