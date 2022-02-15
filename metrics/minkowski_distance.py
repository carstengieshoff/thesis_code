from typing import Union

import numpy as np


def minkowski_dist(x: np.array, y: np.array, ord: Union[int, str] = 2) -> np.array:
    """Calculate minkowski distance of order `ord` between `x` and `y`.

    This wraps https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html with the necessary axis
    assignment. This way the distance between one time stamp of a signal to all other time stamps is calculated in a
    vectorized fashion.

    Args:
        x: `np.array` of shape (dim_signal, )
        y: `np.array` of shape (dim_signal, ) or (len_signal, dim_signal).
        ord: Order of
    Returns:
        A single distance or a vector of distances of shape (len_signal, ).

    """
    return np.linalg.norm(x - y, axis=1, ord=ord)
