import numpy as np


def split_signal(signal: np.array, r_peaks: np.array, front: int, back: int) -> np.array:
    """Split an ECG signal into windows around the r-peaks.

    Args:
        signal: np.array of shape (signal_len, signal_dim) to be split into windows.
        r_peaks: np.array of inidces to base splitting on.
        front: Number of datapoints before each peack to include in a window.
        back: Number of datapoints after each peack to include in a window.

    Retruns:
        A windowed version of the data of shape (signal_dim, num_windows, window_size), where `window_size` = `front` +
        'back' + 1.
    """
    signal_len, signal_dim = signal.shape
    r_diffs = r_peaks[1:] - r_peaks[:-1]
    window_size = front + back + 1

    if window_size > r_diffs.min():
        raise ValueError(f"The specified window size cannot extend {r_diffs.min()}, got {window_size}")

    r_peaks = r_peaks[front <= r_peaks]
    r_peaks = r_peaks[r_peaks < signal_len - back - 1]

    windowed_signal = np.zeros(shape=(signal_dim, r_peaks.shape[0], window_size))
    for i, peak in enumerate(r_peaks):
        window_start = peak - front
        window_end = peak + back + 1

        windowed_signal[:, i, :] = signal[window_start:window_end, :].T

    return windowed_signal


def get_rr_intervals(signal: np.array, r_peaks: np.array) -> np.array:
    """Split an ECG signal into rr-intervals of maximal equal length.


    Args:
        signal: np.array of shape (signal_len, signal_dim) to be split into windows.
        r_peaks: np.array of inidces to base splitting on.

    Retrurns:
        np.array of shape (signal_dim, num_rr_intervals, len_rr_intervals) where `len_rr_intervals` is the minimal
        distance betwenn indices in `r_peaks`.
    """
    r_diffs = r_peaks[1:] - r_peaks[:-1]

    return split_signal(signal=signal, r_peaks=r_peaks, front=0, back=r_diffs.min())
