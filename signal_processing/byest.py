import numpy as np
from scipy.linalg import toeplitz


def byest(Y: np.array, r_peaks: np.array, Fs: int, leadAF: int = 1, nbvec: int = 5) -> np.array:
    """Byesian estimation of AF signal.


    Args:
        Y: n-leads input signal (signals organized on rows, samples on columns
         (Y = (n x N) matrix))
        r_peaks: R-wave time instant series (vector containing time instants of all R
        waves in the signal)
        Fs: sampling frequency
        leadAF: number of the lead we want to extract the atrial activity from
         (generally V1)
        nbvec: number of principal components considered

    Returns:
        Extracted atrial activity signal
    """
    nblead = min(Y.shape)

    r_peak_dist = r_peaks[1:] - r_peaks[:-1]

    indmin = np.floor(0.04 * Fs).astype(int)
    indmax = np.floor(r_peak_dist.min()).astype(int) - indmin - 1

    r_peaks = r_peaks[r_peaks > indmin]

    X = np.zeros(shape=(nblead, indmax + indmin + 1))

    if Y.shape[0] > Y.shape[1]:
        Y = Y.T

    len_y = max(Y.shape)

    for k in range(r_peaks.shape[0]):
        peak = r_peaks[k]
        if r_peaks[k] + indmax < len_y:
            X += Y[:, peak - indmin : peak + indmax + 1]
        else:
            remainder = Y[:, peak - indmin :]
            fill = np.zeros(shape=(nblead, X.shape[1] - remainder.shape[1]))
            X += np.vstack([remainder.T, fill.T]).T

    X = X / r_peaks.shape[0]

    X = X.T

    # Original
    Un = np.ones(shape=(indmax + indmin + 1, 1)) / np.sqrt(indmax + indmin + 1)
    A = np.vstack([Un.T, np.arange(1, indmax + indmin + 1 + 1).reshape(-1, 1).T]).squeeze().T
    U1, S, V = np.linalg.svd((np.eye(indmax + indmin + 1) - A @ np.linalg.pinv(A)) @ X)
    H = np.vstack([U1[:, :nbvec].T, A.T]).T

    # Option 2:
    # A = Un
    # U1, S, V = np.linalg.svd((np.eye(indmax + indmin + 1) - A @ np.linalg.pinv(A)) @ X)
    # K_a = U1[:, :nbvec]
    # H = np.vstack([U1[:, :nbvec].T, np.arange(1, indmax + indmin + 1 + 1).reshape(-1, 1).T]).T

    # LS estimated noise
    IK = np.linalg.pinv(H)
    ri = np.empty(shape=(indmax + indmin + 1,))
    L = r_peaks.shape[0] - 1
    for ll in range(L):
        x = Y[:, r_peaks[ll] - indmin : r_peaks[ll] + indmax + 1].T
        AF = (np.eye(indmin + indmax + 1) - H @ IK) @ x
        if ll == 0:
            ri = AF[:, leadAF]
        else:
            ri = np.vstack([ri.T, AF[:, leadAF].T]).T

    # rb = np.concatenate([Y[leadAF, : r_peaks[0] - indmin - 1 + 1], ri[:, 0]])
    # for k in range(1, L):
    #    rb = np.concatenate([rb, Y[leadAF, r_peaks[k - 1] + indmax + 1: r_peaks[k] - indmin - 1 + 1], ri[:, k]])
    # rb = np.concatenate([rb, Y[leadAF, rb.shape[0] + 1 - 1 :]])

    # covariance matrix estimation
    c = np.zeros(shape=((indmax + indmin) * 2 + 1,))
    LL = L

    for k in range(LL):
        c = c + np.correlate(a=ri[:, k], v=ri[:, k], mode="full") / ri.shape[0]
    c = c / LL

    Cb = toeplitz(c=c[indmax + indmin : 2 * (indmax + indmin) + 1 + 1] / (indmax + indmin + 1))
    ICb = np.linalg.inv(Cb)

    # BLUE estimation
    IK = np.linalg.inv(H.T @ ICb @ H) @ H.T @ ICb
    for ll in range(LL):
        x = Y[:, r_peaks[ll] - indmin : r_peaks[ll] + indmax + 1].T
        AF = (np.eye(indmin + indmax + 1) - H @ IK) @ x
        if ll == 0:
            rib = AF[:, leadAF]
        else:
            rib = np.vstack([rib.T, AF[:, leadAF].T]).T

    # reconstitution of AF signal
    re = np.concatenate([Y[leadAF, 0 : r_peaks[0] - indmin - 1 + 1].T, rib[:, 0]])
    for k in range(1, L):
        re = np.concatenate([re, Y[leadAF, r_peaks[k - 1] + indmax + 1 : r_peaks[k] - indmin - 1 + 1].T, rib[:, k]])
    re = np.concatenate([re.T, Y[leadAF, re.shape[0] :]])

    # reconstitution with continuity at connection points
    re2 = np.concatenate([Y[leadAF, : r_peaks[0] - indmin - 1 + 1].T, rib[:, 0]])
    for k in range(1, L):
        c1 = rib[indmin + indmax, k - 1]
        c2 = rib[0, k]
        N = r_peaks[k] - r_peaks[k - 1] - indmin - indmax + 1
        bet = (Y[leadAF, r_peaks[k] - indmin] - Y[leadAF, r_peaks[k - 1] + indmax] - c2 + c1) / (N - 1)
        alp = Y[leadAF, r_peaks[k - 1] + indmax] - c1 - bet
        vec = Y[leadAF, r_peaks[k - 1] + indmax : r_peaks[k] - indmin + 1] - (alp + bet * np.arange(1, N + 1))
        re2 = np.concatenate([re2, vec[1:-1].T, rib[:, k]])

    re2 = np.concatenate([re2, Y[leadAF, re2.shape[0] :].T]).T
    return re2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    data = loadmat("../tests/data/detqrs_data.mat")
    qrs_locs = data["qrs_locs"].squeeze() - 1
    data_centered = data["data_centered"].T
    fs = 1000
    data_af = np.zeros(shape=(12240, 12))
    for lead in range(12):
        data_af[:, lead] = byest(data_centered, qrs_locs, fs, lead)

    fig, ax = plt.subplots(12, 1, figsize=(10, 40))

    for i in range(12):
        ax[i].plot(data_centered[:, i], label="original")
        ax[i].plot(data_af[:, i], label="corrected")
        ax[i].scatter(qrs_locs - 2, data_centered[qrs_locs - 2, i], marker="o", color="red")
        ax[i].set_title(f"lead_{i + 1}")
        ax[i].legend()

    plt.show()
