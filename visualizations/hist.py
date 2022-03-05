from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt


def plot_hist2d(
    a: np.array, b: np.array, xlabel: Optional[str] = None, ylabel: Optional[str] = None, *args: Any, **kwargs: Any
) -> None:
    """Plot 2D histrogram.

    Args:
        a: np.array of shape (len_a,)
        b: np.array of shape (len_b,)
    """
    # Todo: Add margin distrubutions
    a_bins = np.arange(-1, max(a) + 1) + 0.5
    b_bins = np.arange(-1, max(b) + 1) + 0.5
    hist, xedges, yedges = np.histogram2d(a, b, bins=(a_bins, b_bins))

    fig, ax = plt.subplots(*args, **kwargs)
    pos = ax.imshow(hist)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_yticks((xedges[:-1] + xedges[1:]) / 2)
    ax.set_xticks((yedges[:-1] + yedges[1:]) / 2)
    fig.colorbar(pos, ax=ax)
    plt.show()

    id_x, id_y = (hist == hist.max()).nonzero()
    id_x, id_y = id_x.item(), id_y.item()
    print(f"Max at: ({id_x}, {id_y})")


if __name__ == "__main__":

    a = np.random.randint(0, 10, 100)
    b = np.random.randint(0, 10, 100)

    plot_hist2d(a=a, b=b)
