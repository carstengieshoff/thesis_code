from typing import Tuple

import numpy as np
from scipy.linalg import toeplitz


def _get_cov(s: np.array) -> np.array:
    num_windows, window_length = s.shape
    auto_corr = np.zeros(2 * window_length - 1)
    for k in range(num_windows):
        # MATLAB xcorr(s,"biased") = np.correlate(s,s)/len(s)
        auto_corr = auto_corr + np.correlate(a=s[k, :], v=s[k, :], mode="full") / window_length

    auto_corr = auto_corr[window_length - 1 :]
    cov = toeplitz(c=auto_corr)
    return cov


class QRSEstimator:
    """Bayesian estimation of AF signal.

    Args:
        Fs: sampling frequency.
        nbvec: Nnumber of principal components considered.

    Returns:
        Extracted atrial activity signal.
    """

    def __init__(self, Fs: int, nbvec: int = 5) -> None:
        self.Fs = Fs
        self.nbvec = nbvec
        self.indmin = np.floor(0.04 * self.Fs).astype(int)

    def __call__(self, Y: np.array, r_peaks: np.array) -> Tuple[np.ndarray, np.ndarray]:
        """Extract contact and ambient noise from ECG signals.


        Args:
        Y: n-leads input signal (signals organized on rows, samples on columns
         (Y = (n x N) matrix))
        r_peaks: R-wave time instant series (vector containing time instants of all R
        waves in the signal)

        Returns:
            X: np.array of shape (n, num_peaks, window_length) representing windowed versions of the signal.
            b: np.array of shape (n, num_peaks, window_length) representing the estimations of contact and ambient noise

        """
        r_peaks = r_peaks[r_peaks > self.indmin]

        X = self._get_X(signal=Y.T, r_peaks=r_peaks)
        H = self._get_model_matrix(X=X, original=True)
        X = np.delete(X, -1, axis=1)
        b_ls = self._get_LS_estimates(X=X, H=H)
        b_blue = self._get_BLUE_estimates(X=X, LS_estimates=b_ls, H=H)
        return X, b_blue

    def reconstruct(self, Y: np.array, r_peaks: np.array) -> np.ndarray:
        """"""
        X, b = self.__call__(Y=Y, r_peaks=r_peaks)
        num_leads, num_windows, window_length = X.shape
        indmax = window_length - self.indmin - 1

        if Y.shape[0] > Y.shape[1]:
            Y = Y.T

        r_peaks = r_peaks[r_peaks > self.indmin]
        re = np.concatenate([Y[:, : r_peaks[0] - self.indmin].T, b[:, 0, :].T]).T

        # reconstitution with continuity at connection points
        for k in range(1, num_windows):
            c1 = b[:, k - 1, -1]
            c2 = b[:, k, 0]
            N = r_peaks[k] - r_peaks[k - 1] - window_length + 2
            gap = np.repeat(np.arange(1, N + 1).reshape(1, -1), repeats=num_leads, axis=0)
            bet = (Y[:, r_peaks[k] - self.indmin] - Y[:, r_peaks[k - 1] + indmax] - c2 + c1) / (N - 1)
            alp = Y[:, r_peaks[k - 1] + indmax] - c1 - bet

            vec = Y[:, r_peaks[k - 1] + indmax : r_peaks[k] - self.indmin + 1] - (
                alp.reshape(-1, 1) + bet.reshape(-1, 1) * gap
            )
            re = np.concatenate([re.T, vec[:, 1:-1].T, b[:, k, :].T]).T

        re = np.concatenate([re.T, Y[:, re.shape[1] :].T]).T
        return re.T

    def _get_X(self, signal: np.array, r_peaks: np.array) -> np.array:
        """Return a windowed version of `Y`.

        If `Y` is of shape (num_leads, signal_length), and `r_peaks` is of size (num_peaks,) this returns
        `X` of size (num_leads, num_peaks, window_length). This constitutes a window of length 'window_length' around
        each peak for each lead of the signal `Y`.

        """
        nblead = signal.shape[0]
        num_peaks = r_peaks.shape[0]
        r_peak_dist = r_peaks[1:] - r_peaks[:-1]

        indmax = np.floor(r_peak_dist.min()).astype(int) - self.indmin - 1
        window_length = self.indmin + indmax + 1

        X = np.zeros(shape=(nblead, num_peaks, window_length))
        for k, peak in enumerate(r_peaks):
            if peak + indmax < signal.shape[1]:
                X[:, k, :] = signal[:, peak - self.indmin : peak + indmax + 1]
            else:
                remainder = signal[:, peak - self.indmin :]
                fill = np.zeros(shape=(nblead, window_length - remainder.shape[1]))
                X[:, k, :] += np.vstack([remainder.T, fill.T]).T

        return X

    def _get_model_matrix(self, X: np.array, original: bool = False) -> np.array:
        X_mean = X.mean(axis=1)
        window_length = X_mean.shape[1]

        unit_vec = np.ones(shape=(window_length, 1)) / np.sqrt(window_length)
        n_vec = np.arange(1, window_length + 1).reshape(-1, 1)

        if original:
            A = np.vstack([unit_vec.T, n_vec.T]).squeeze().T
            U, _, _ = np.linalg.svd((np.eye(window_length) - A @ np.linalg.pinv(A)) @ X_mean.T)
            H = np.vstack([U[:, : self.nbvec].T, A.T]).T
        else:
            A = unit_vec
            U, _, _ = np.linalg.svd((np.eye(window_length) - A @ np.linalg.pinv(A)) @ X_mean.T)
            H = np.vstack([U[:, : self.nbvec].T, n_vec.T]).T

        return H

    def _get_LS_estimates(self, X: np.array, H: np.array) -> np.array:
        """Estimate contact and ambient noise using LS."""
        IH = np.linalg.pinv(H)
        b_LS = np.einsum("ij,lnj->lni", (np.eye(X.shape[2]) - H @ IH), X)
        return b_LS

    def _get_BLUE_estimates(self, X: np.array, LS_estimates: np.array, H: np.array) -> np.array:
        """Estimate contact and ambient noise using BLUE."""
        nblead, num_windows, window_length = X.shape
        b = np.zeros_like(LS_estimates)

        for lead in range(nblead):
            Cb = _get_cov(
                s=LS_estimates[lead, :, :],
            )
            ICb = np.linalg.inv(Cb)

            IH = np.linalg.inv(H.T @ ICb @ H) @ H.T @ ICb
            b[lead, :, :] = np.einsum("ij,nj->ni", (np.eye(window_length) - H @ IH), X[lead, :, :])

        return b


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    data = loadmat("../tests/data/detqrs_data.mat")
    qrs_locs = data["qrs_locs"].squeeze() - 1
    data_centered = data["data_centered"].T
    fs = 1000

    byest = QRSEstimator(Fs=fs, nbvec=4)

    data_af = byest.reconstruct(Y=data_centered, r_peaks=qrs_locs)
    fig, ax = plt.subplots(12, 2, figsize=(10, 40))

    for i in range(12):
        ax[i, 0].plot(data_centered[:, i], label="original")
        ax[i, 0].plot(data_af[:, i], label="corrected")
        ax[i, 0].scatter(qrs_locs - 2, data_centered[qrs_locs - 2, i], marker="o", color="red")
        ax[i, 0].set_title(f"lead_{i + 1}")
        ax[i, 0].legend()
        ax[i, 1].plot(data_centered[:, i] - data_af[:, i])
        ax[i, 1].set_title("denoised")

    plt.show()
