from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_rp(
    rp_data: np.array,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    *args: Any,
    **kwargs: Any,
) -> None:
    if ax is None or fig is None:
        fig, ax = plt.subplots(*args, **kwargs)

    cmap = kwargs.get("cmap")
    pos = ax.imshow(rp_data, cmap)
    fig.colorbar(pos, ax=ax)
    ax.set_axis_off()
    if ax is None or fig is None:
        plt.show()
