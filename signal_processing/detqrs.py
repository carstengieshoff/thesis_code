import numpy as np
from scipy.signal import firwin, lfilter


def detqrs3(signal: np.array, Fs: int, n: int = 4, Ft: int = 60, pth: int = 50) -> np.array:
    """Detect QRS peak in 1-lead ECG signal.

    Args:
        signal: ECG signal as `np.array` of shape (len_sigal, 1).
        Fs: Sampling rate of `signal`.
        n: Order of low-pass filter.
        Ft: Cut-off frequency of low-pass filter.
        pth: Percentage of QRS amplitude for computing threshold. This defaults to 50 (i.e. 50%).

    Returns:
        Locations (indices) of QRS peaks in `signal`.
    """
    cutoff = 2 * Ft / Fs

    impulse_response = firwin(numtaps=n + 1, cutoff=cutoff)
    signal = lfilter(b=impulse_response, a=1, x=signal.squeeze())

    signal[:4] = signal.mean(axis=0)
    diffs = np.zeros_like(signal)
    diffs[1:] = (signal[1:] - signal[:-1]) * 100
    diffs = np.where(diffs > 0, diffs**2, 0)

    diff_threshold = diffs.max(axis=0) * pth / 100

    peak_indicator = np.zeros_like(signal).squeeze()
    w = np.floor(0.16 * Fs).astype(int)
    s = np.floor(0.3 * Fs).astype(int)

    for i in range(w, signal.shape[0] - w):
        if (signal[i] > signal[i - w : i]).all() and (signal[i] > signal[i + 1 : i + w + 1]).all():
            if i <= s - 1:
                peak_indicator[i] = (peak_indicator[i - w : i] == 0).all() and diffs[
                    i - w : i + w + 1
                ].max() > diff_threshold
            else:
                peak_indicator[i] = (peak_indicator[i - s : i] == 0).all() and diffs[
                    i - w : i + w + 1
                ].max() > diff_threshold

    if (peak_indicator[:s] == 0).all():
        for i in range(1, s):
            peak_indicator[i] = (signal[i] > signal[:i]).all() and (signal[i] > signal[i + 1 : s + 1]).all()
            if peak_indicator[i]:
                break

    peak_indices = np.argwhere(peak_indicator)
    return peak_indices


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    # from scipy.signal import cheby2

    data = loadmat("../matlab_src/12leadecg.mat")["data"].T
    x = loadmat("../matlab_src/Y1.mat")["x"].T

    fs = 1000
    # data_centered = data - data.mean(axis=0)
    # b, a = cheby2(N=3, rs=20.0, Wn=[0.5/(fs/2), 100/(fs/2)], output='ba', analog=False, btype="bandpass")
    # data_centered = lfilter(b=b, a=a, x=data_centered, axis=0)
    qrs_locs = detqrs3(x, Fs=fs)

    fig, ax = plt.subplots(12, 2, figsize=(10, 40))

    for i in range(12):
        ax[i, 0].plot(data[:, i])
        ax[i, 1].plot(data[:, i])
        ax[i, 1].scatter(qrs_locs, data[qrs_locs, i], marker="o", color="red")
        ax[i, 0].set_title(f"lead_{i + 1}")
        ax[i, 1].set_title(f"lead_{i + 1}_filtered")

    plt.show()
