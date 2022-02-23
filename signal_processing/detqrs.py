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

    wt = 2 * Ft / Fs

    b = firwin(numtaps=n + 1, cutoff=wt)
    Y1 = lfilter(b=b, a=1, x=signal.squeeze())

    Y1[:4] = Y1.mean(axis=0)
    Y2 = np.zeros_like(Y1)
    Y2[1:] = (Y1[1:] - Y1[:-1]) * 100
    Y3 = np.where(Y2 > 0, Y2**2, 0)

    th = Y3.max(axis=0) * pth / 100

    p = np.zeros_like(Y1).squeeze()
    w = np.floor(0.16 * Fs).astype(int)
    s = np.floor(0.3 * Fs).astype(int)

    for i in range(w, Y1.shape[0] - w):
        if i > w:
            if (Y1[i] > Y1[i - w : i]).all() and (Y1[i] > Y1[i + 1 : i + w + 1]).all():
                if i <= s - 1:
                    p[i] = (p[i - w : i] == 0).all() and Y3[i - w : i + w + 1].max() > th
                else:
                    p[i] = (p[i - s : i] == 0).all() and Y3[i - w : i + w + 1].max() > th

        else:  # This should not be necessary at all
            if (Y1[i] > Y1[i - w : i]).all() and (Y1[i] > Y1[i + 1 : i + 1 + w]).all():
                p[i] = (p[i - w : i] == 0).all() and Y3[i - w : i + w + 1].min() > th

    if (p[:s] == 0).all():
        for i in range(1, s):
            p[i] = (Y1[i] > Y1[:i]).all() and (Y1[i] > Y1[i + 1 : s + 1]).all()
            if p[i]:
                break

    return np.argwhere(p)


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
        ax[i, 1].scatter(qrs_locs - 2, data[qrs_locs - 2, i], marker="o", color="red")
        ax[i, 0].set_title(f"lead_{i + 1}")
        ax[i, 1].set_title(f"lead_{i + 1}_filtered")

    plt.show()
