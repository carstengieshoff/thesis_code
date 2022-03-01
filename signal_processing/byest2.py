import numpy as np
from scipy.linalg import toeplitz


def byest(
    Y: np.array,
    r_peaks: np.array,
    Fs: int,
    nbvec: int = 5,
    continuous_connections: bool = True,
    return_last_window: bool = True,
) -> np.array:
    """Bayesian estimation of AF signal.


    Args:
        Y: n-leads input signal (signals organized on rows, samples on columns
         (Y = (n x N) matrix))
        r_peaks: R-wave time instant series (vector containing time instants of all R
        waves in the signal)
        Fs: sampling frequency
        leadAF: number of the lead we want to extract the atrial activity from
         (generally V1)
        nbvec: number of principal components considered
        continuous_connections: Indicator whether to return a continuous signal. This defaults to True.
        return_last_window: Indicator whether to omit the last incomplete window. This defaults to True.

    Returns:
        Extracted atrial activity signal
    """
    nblead = min(Y.shape)

    r_peak_dist = r_peaks[1:] - r_peaks[:-1]

    indmin = np.floor(0.04 * Fs).astype(int)
    indmax = np.floor(r_peak_dist.min()).astype(int) - indmin - 1
    window_length = indmin + indmax + 1
    num_windows = r_peaks.shape[0]

    r_peaks = r_peaks[r_peaks > indmin]

    X = np.zeros(shape=(nblead, num_windows, window_length))

    if Y.shape[0] > Y.shape[1]:
        Y = Y.T

    len_y = max(Y.shape)

    for k in range(num_windows):
        peak = r_peaks[k]
        if peak + indmax < len_y:
            X[:, k, :] = Y[:, peak - indmin : peak + indmax + 1]
        else:
            remainder = Y[:, peak - indmin :]
            fill = np.zeros(shape=(nblead, window_length - remainder.shape[1]))
            X[:, k, :] += np.vstack([remainder.T, fill.T]).T

    X_mean = X.mean(axis=1)

    X_mean = X_mean.T

    # Original
    Un = np.ones(shape=(window_length, 1)) / np.sqrt(window_length)
    A = np.vstack([Un.T, np.arange(1, window_length + 1).reshape(-1, 1).T]).squeeze().T
    U1, _, _ = np.linalg.svd((np.eye(window_length) - A @ np.linalg.pinv(A)) @ X_mean)
    H = np.vstack([U1[:, :nbvec].T, A.T]).T

    # Option 2:
    # Un = np.ones(shape=(window_length, 1)) / np.sqrt(window_length)
    # A = Un
    # U1, S, V = np.linalg.svd((np.eye(window_length) - A @ np.linalg.pinv(A)) @ X_mean)
    # K_a = U1[:, :nbvec]
    # H = np.vstack([K_a.T, np.arange(1, window_length + 1).reshape(-1, 1).T]).T

    # LS estimated noise
    IH = np.linalg.pinv(H)

    b_LS = np.einsum("ij,lnj->lni", (np.eye(window_length) - H @ IH), X)
    b_LS = b_LS[:, :-1, :]  # Why is the last window not used -> It is incopmplete

    # covariance matrix estimation
    auto_corr = np.zeros(
        shape=(
            nblead,
            2 * window_length - 1,
        )
    )

    for lead in range(nblead):
        for k in range(num_windows - 1):
            # MATLAB xcorr(s,"biased") = np.correlate(s,s)/len(s)
            auto_corr[lead, :] = (
                auto_corr[lead, :] + np.correlate(a=b_LS[lead, k, :], v=b_LS[lead, k, :], mode="full") / window_length
            )

    # Scaling is irrelevant due to later use (always products using matrix and its inverse)
    auto_corr = auto_corr[:, window_length - 1 :]  # / ((window_length-1) * window_length)

    Cb = np.zeros(shape=(nblead, window_length, window_length))
    ICb = np.zeros(shape=(nblead, window_length, window_length))
    b_BLUE = np.zeros_like(X)

    for lead in range(nblead):
        Cb[lead, :, :] = toeplitz(c=auto_corr[lead, :])
        ICb[lead, :, :] = np.linalg.inv(Cb[lead, :, :])

        # BLUE estimation
        IH = np.linalg.inv(H.T @ ICb[lead, :, :] @ H) @ H.T @ ICb[lead, :, :]
        b_BLUE[lead, :, :] = np.einsum("ij,nj->ni", (np.eye(window_length) - H @ IH), X[lead, :, :])

    b_BLUE = b_BLUE[:, :-1, :]  # Remove incomplete window

    if return_last_window:
        cleaned_signal = np.zeros_like(Y)
    else:
        cleaned_signal = np.zeros(shape=(nblead, r_peaks[-2] + indmax + 1))

    for lead in range(nblead):
        re = np.concatenate([Y[lead, : r_peaks[0] - indmin].T, b_BLUE[lead, 0, :]])
        if continuous_connections:
            # reconstitution with continuity at connection points
            for k in range(1, num_windows - 1):
                end_point_last = b_BLUE[lead, k - 1, indmin + indmax]  # last inices might need to be changed
                start_point_next = b_BLUE[lead, k, 0]  # last inices might need to be changed
                num_points_gap = r_peaks[k] - r_peaks[k - 1] - indmin - indmax + 1
                bet = (X[lead, k, 0] - X[lead, k - 1, -1] - start_point_next + end_point_last) / (num_points_gap - 1)
                alp = X[lead, k - 1, -1] - end_point_last - bet
                vec = Y[lead, r_peaks[k - 1] + indmax : r_peaks[k] - indmin + 1] - (
                    alp + bet * np.arange(1, num_points_gap + 1)
                )
                re = np.concatenate([re, vec[1:-1].T, b_BLUE[lead, k, :]])

        else:
            # reconstitution of AF signal
            for k in range(1, r_peaks.shape[0]):
                re = np.concatenate(
                    [re, Y[lead, r_peaks[k - 1] + indmax + 1 : r_peaks[k] - indmin].T, b_BLUE[lead, k, :]]
                )
        if return_last_window:
            re = np.concatenate([re, Y[lead, re.shape[0] :].T]).T
        cleaned_signal[lead, :] = re

    return cleaned_signal, X, b_BLUE


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    data = loadmat("../tests/data/detqrs_data.mat")
    qrs_locs = data["qrs_locs"].squeeze() - 1
    data_centered = data["data_centered"].T
    fs = 1000
    # for lead in range(12):
    #    data_af[:, lead] = byest(data_centered, qrs_locs, fs, lead)
    data_af = byest(data_centered, qrs_locs, fs, nbvec=4, return_last_window=False).T

    fig, ax = plt.subplots(12, 2, figsize=(10, 40))

    for i in range(12):
        ax[i, 0].plot(data_centered[:, i], label="original")
        ax[i, 0].plot(data_af[:, i], label="corrected")
        ax[i, 0].scatter(qrs_locs - 2, data_centered[qrs_locs - 2, i], marker="o", color="red")
        ax[i, 0].set_title(f"lead_{i + 1}")
        ax[i, 0].legend()
        ax[i, 1].plot(data_centered[: data_af.shape[0], i] - data_af[:, i])
        ax[i, 1].set_title("denoised")

    plt.show()
