import numpy as np


def normalize(
    signal: np.array,
) -> np.array:
    """Split an ECG signal into windows around the r-peaks.

    Args:
        signal: np.array of shape (signal_len, signal_dim) to be normalized.
    """
    return (signal - signal.mean(axis=0)) / signal.std(axis=0)
