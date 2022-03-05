import numpy as np
from scipy.signal import resample


def split_rr_to_equal_length(signal: np.array, r_peaks: np.array, new_window_size: int, offset: int = 0) -> np.array:
    """Split an ECG signal into windows around the r-peaks.

    Args:
        signal: np.array of shape (signal_len, signal_dim) to be split into windows.
        r_peaks: np.array of inidces to base splitting on.
        new_window_size: Integer of window length to sample all RR-intervals to.

    Retruns:
        A windowed version of the data of shape (signal_dim, num_windows, new_window_size), where each window
        corresponds to one RR-Interval.
    """
    signal_len, signal_dim = signal.shape
    num_windows = r_peaks.shape[0] - 1

    r_peaks = r_peaks - min(r_peaks.min(), offset)

    windowed_signal = np.zeros(shape=(signal_dim, num_windows, new_window_size))
    for i, (current_peak, next_peak) in enumerate(zip(r_peaks[:-1], r_peaks[1:])):
        window = signal[current_peak:next_peak, :]
        window = resample(x=window, num=new_window_size, axis=0)
        windowed_signal[:, i, :] = window.T

    return windowed_signal
