import numpy as np


def cosine_dist(x: np.array, y: np.array) -> np.array:
    """Calculate cosine distance between `x` and `y`.

    This implements a vectorized version of calculating the cosine distance. This imposes assumptions
    on the shape of the arguments, as described.

    Args:
        x: `np.array` of shape (dim_signal, )
        y: `np.array` of shape (dim_signal, ) or (len_signal, dim_signal).
    Returns:
        A single distance or a vector of distances of shape (len_signal, ).

    """
    denominator = np.sqrt(np.dot(x, x)) * np.sqrt(np.diag(np.dot(y, y.T)))
    numerator = np.dot(y, x)
    return 1 - numerator / denominator


if __name__ == "__main__":
    signal = np.eye(5)
    dist = cosine_dist(x=signal[0, :], y=signal)
    assert dist.shape == (5,)
    print(dist)
