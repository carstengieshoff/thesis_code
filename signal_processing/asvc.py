import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import gaussian

from signal_processing.fixed_window_signal_splitting import split_signal

logging.basicConfig(level=logging.INFO)


class ASVCancellator:
    def __call__(
        self,
        original_signal: np.array,
        r_peaks: np.array,
        verbose: bool = False,
        fit: str = "normal",
        *args: Any,
        **kwargs: Any,
    ) -> np.array:

        # PreProcessing w.r.t R-peaks
        r_peak_dist = r_peaks[1:] - r_peaks[:-1]
        r_peak_min = r_peak_dist.min()

        front = int(0.3 * r_peak_min)
        back = int(0.7 * r_peak_min)

        # Pad signal
        pad_front = max(0, front - r_peaks.min())
        pad_back = max(0, r_peaks.max() + back - original_signal.shape[0] + 1)
        r_peaks_shifted = r_peaks + pad_front

        signal_padded = np.vstack(
            [
                np.zeros(shape=(pad_front, original_signal.shape[1])),
                original_signal,
                np.zeros(shape=(pad_back, original_signal.shape[1])),
            ]
        )

        # Windowing
        X = split_signal(signal=signal_padded, r_peaks=r_peaks_shifted, front=front, back=back)

        # Create Template (lead by lead)
        # Find subset of windows to use (e.g. by similarity, neighbours, clustering)
        template = self._get_template(X, plot_templates=verbose)

        # Fit template to window
        # Fit by amplitude
        if fit == "normal":
            template_fitted = self._fit_template_to_windows_lstsq(X, template, verbose=verbose)
        elif fit == "shifted":
            template_fitted = self._fit_template_to_windows_shift_lstsq(X, template, verbose=verbose)
        else:
            raise RuntimeError(f"Unrecognized option for `fit`. Expected one of `normal`, `shifted`: got {fit}")

        # fit transitions
        aa_signal = self._subtract_template(
            windowed_signal=X, template=template_fitted, P=40, M=20, smooth_transitions=False
        )

        aa_signal_reconstructed = self._reconstruct(aa_signal, signal_padded, r_peaks_shifted, front, back)

        aa_signal_reconstructed_depadded = aa_signal_reconstructed[pad_front:-pad_back]
        # Evaluate (optionally)

        if verbose:
            self._plot(original_signal, aa_signal_reconstructed_depadded, r_peaks, template_fitted, front, back)

        return aa_signal_reconstructed_depadded

    def _plot(
        self,
        original_signal: np.array,
        transformed_signal: np.array,
        r_peaks: np.array,
        template: np.array,
        front: int,
        back: int,
    ) -> None:
        signal_len, n_leads = original_signal.shape

        fig, ax = plt.subplots(n_leads, 4, figsize=(50, 60))
        plt.title("QRST-cancellation using Adaptive Singluar Value Cancellation", fontsize=30)
        if n_leads > 1:
            for i in range(n_leads):
                ax[i, 0].plot(original_signal[:, i], label="original")
                ax[i, 0].scatter(r_peaks, original_signal[r_peaks, i], marker="o", color="red", label="r-peaks")
                ax[i, 0].set_title(f"lead_{i + 1}")
                ax[i, 0].legend()

                ax[i, 1].plot(transformed_signal[:, i])
                ax[i, 1].set_title("AA")

                ax[i, 2].plot(
                    original_signal[:, i] - transformed_signal[:, i],
                )
                ax[i, 2].set_title("VA (orig-AA)")

                ax[i, 3].plot(template[i, 0, :])
                ax[i, 3].set_title("lead-template")

                for j in range(3):
                    for peak in r_peaks:
                        ax[i, j].axvspan(
                            peak - front, peak + back, facecolor="gray", alpha=0.2, label="considered window"
                        )

            plt.show()

        else:
            ax[0].plot(original_signal, label="original")
            ax[0].scatter(r_peaks, original_signal[r_peaks, :], marker="o", color="red", label="r-peaks")
            ax[0].set_title()
            ax[0].legend()

            ax[1].plot(transformed_signal)
            ax[1].set_title("AA")

            ax[2].plot(
                original_signal - transformed_signal,
            )
            ax[2].set_title("VA (orig-AA)")

            ax[3].plot(template[0, 0, :])
            ax[3].set_title("lead-template")

            for j in range(3):
                for peak in r_peaks:
                    ax[j].axvspan(peak - front, peak + back, facecolor="gray", alpha=0.2, label="considered window")

        plt.show()

    def _get_template(self, windowed_signal: np.array, plot_templates: bool = True) -> np.array:

        n_leads, n_windows, window_size = windowed_signal.shape

        # template = np.zeros(shape=(n_leads, window_size))
        template = np.zeros_like(windowed_signal)

        # np.linalg.svd can be vectorized // only if we do not take subsets
        for lead in range(n_leads):
            # subset_idxs = list(range(n_windows)) # to be made more sophistciated

            U, _, _ = np.linalg.svd(windowed_signal[lead, :, :].T)
            # U = PCA(windowed_signal[lead, :, :].T, var=0.7)

            template[lead, :, :] = np.broadcast_to(U[:, 0], shape=(n_windows, window_size))
            # template[lead, :, :] = np.broadcast_to(U.sum(axis=1), shape=(n_windows, window_size))

        return template

    def _fit_template_to_windows_lstsq(
        self, windowed_signal: np.array, template: np.array, verbose: bool = False
    ) -> np.array:
        assert windowed_signal.shape == template.shape
        n_leads, n_windows, window_size = windowed_signal.shape

        coeffs = np.zeros(shape=(n_leads, n_windows, 2))

        for lead in range(n_leads):

            design_matrix = np.stack([template[lead, 0, :], np.ones_like(template[lead, 0, :])]).T
            lstq_results = np.linalg.lstsq(a=design_matrix, b=windowed_signal[lead, :, :].T)
            coeffs[lead, :, :] = lstq_results[0].T

            if verbose:
                logging.info(
                    f"Fitting lead {lead+1}: a,b = {lstq_results[0].mean(axis=1)}, residual = {lstq_results[1]}"
                )

        template_aligned = template * np.expand_dims(coeffs[:, :, 0], axis=2) + np.expand_dims(coeffs[:, :, 1], axis=2)
        return template_aligned

    def _fit_template_to_windows_shift_lstsq(
        self, windowed_signal: np.array, template: np.array, verbose: bool = False
    ) -> np.array:
        assert windowed_signal.shape == template.shape
        n_leads, n_windows, window_size = windowed_signal.shape

        coeffs = np.zeros(shape=(n_leads, n_windows, 2))
        template_shifted = template.copy()
        for lead in range(n_leads):
            shifts = np.argmax(np.abs(windowed_signal[lead, :, :]), axis=1) - np.argmax(
                np.abs(template[lead, :, :]), axis=1
            )
            shifts = np.where(np.abs(shifts) < 100, shifts, 0)
            for window in range(n_windows):
                shift = shifts[window]
                template_shifted[lead, window, :] = np.roll(template_shifted[lead, window, :], shift=shift)

                design_matrix = np.stack([template_shifted[lead, window, :], np.ones_like(template[lead, window, :])]).T
                lstq_results = np.linalg.lstsq(a=design_matrix, b=windowed_signal[lead, window, :].T)
                coeffs[lead, window, :] = lstq_results[0].T

                if verbose:
                    logging.info(
                        f"Fitting lead {lead+1}, window {window}:"
                        f" a,b = {lstq_results[0]},"
                        f" residual = {lstq_results[1]},"
                        f" shift {shift}"
                    )

        template_aligned = template_shifted * np.expand_dims(coeffs[:, :, 0], axis=2) + np.expand_dims(
            coeffs[:, :, 1], axis=2
        )
        return template_aligned

    def _subtract_template(
        self,
        windowed_signal: np.array,
        template: np.array,
        P: int,
        M: int,
        smooth_transitions: bool = True,
    ) -> np.array:
        n_leads, n_windows, window_size = windowed_signal.shape

        aa_signal = windowed_signal.copy()

        for lead in range(n_leads):
            for window in range(n_windows):

                diff = np.abs(windowed_signal[lead, window, :] - template[lead, window, :])
                start = np.argmin(diff[:P])
                end = np.argmin(diff[-P:]) + window_size - P

                template[lead, window, :start] = 0
                template[lead, window, end:] = 0

                aa_signal[lead, window, :] -= template[lead, window, :]

                # post-process
                if start >= 1:
                    M_ = min(M, start)
                    gaussian_window = gaussian(2 * M_, np.sqrt(2 * M_))
                    ks = (aa_signal[lead, window, start - 1] - aa_signal[lead, window, start]) / 2
                    aa_signal[lead, window, start - M_ : start] -= ks * gaussian_window[:M_]
                    aa_signal[lead, window, start : start + M_] += ks * gaussian_window[M_:]

                if end < window_size - 1:
                    M_ = min(M, window_size - end - 1)
                    gaussian_window = gaussian(2 * M_, np.sqrt(2 * M_))
                    ke = (aa_signal[lead, window, end] - aa_signal[lead, window, end + 1]) / 2
                    aa_signal[lead, window, end - M_ + 1 : end + 1] -= ke * gaussian_window[:M_]
                    aa_signal[lead, window, end + 1 : end + M_ + 1] += ke * gaussian_window[M_:]

        return aa_signal

    def _reconstruct(
        self,
        aa_signal: np.array,
        original_signal: np.array,
        r_peaks: np.array,
        front: int,
        back: int,
    ) -> np.array:
        reconstructed_signal = original_signal.copy()

        for i, peak in enumerate(r_peaks):
            window_start = peak - front
            window_end = peak + back + 1
            reconstructed_signal[window_start:window_end, :] = aa_signal[:, i, :].T

        return reconstructed_signal


if __name__ == "__main__":
    from scipy.io import loadmat

    from signal_processing import detqrs3

    data = loadmat("../tests/data/detqrs_data.mat")
    # qrs_locs = data["qrs_locs"].squeeze() - 1
    data_centered = data["data_centered"].T
    data_centered = data_centered - data_centered.mean(axis=0)
    fs = 1000

    qrs_locs = detqrs3(data_centered[:, 0], fs)  # get_r_peaks(data_centered[:,0], fs)
    asvc = ASVCancellator()

    data_af = asvc(original_signal=data_centered, r_peaks=qrs_locs, verbose=True, fit="shifted")
