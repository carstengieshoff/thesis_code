import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import gaussian
from sklearn.cluster import SpectralClustering

from signal_processing.fixed_window_signal_splitting import split_signal

logging.basicConfig(level=logging.INFO)


class ASVCancellator:
    def __call__(
        self,
        original_signal: np.array,
        r_peaks: np.array,
        verbose: bool = False,
        fit: str = "normal",
        P: int = 40,
        M: int = 20,
        use_clustering: bool = False,
        min_cluster_size: Optional[int] = None,
        savefig: bool = False,
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
        min_cluster_size = int(X.shape[1] / 4) if min_cluster_size is None else min_cluster_size
        template, cluster_labels = self._get_template(
            X, plot_templates=verbose, use_clustering=use_clustering, min_cluster_size=min_cluster_size
        )

        # Fit template to window
        # Fit by amplitude
        if fit == "normal":
            template_fitted = self._fit_template_to_windows_lstsq(X, template, verbose=verbose)
        elif fit == "shifted":
            template_fitted = self._fit_template_to_windows_shift_lstsq(X, template, verbose=verbose)
        else:
            raise RuntimeError(f"Unrecognized option for `fit`. Expected one of `normal`, `shifted`: got {fit}")

        # fit transitions
        aa_signal, starts_ends = self._subtract_template(
            windowed_signal=X, template=template_fitted, P=P, M=M, smooth_transitions=False
        )

        aa_signal_reconstructed = self._reconstruct(
            aa_signal, signal_padded, r_peaks_shifted, front, back, starts_ends, M=M
        )

        aa_signal_reconstructed_depadded = aa_signal_reconstructed[pad_front:-pad_back]
        # Evaluate (optionally)

        if verbose:

            self._plot(
                original_signal,
                aa_signal_reconstructed_depadded,
                r_peaks,
                template_fitted,
                front,
                back,
                cluster_labels,
                savefig=savefig,
            )

        return aa_signal_reconstructed_depadded

    def _plot(
        self,
        original_signal: np.array,
        transformed_signal: np.array,
        r_peaks: np.array,
        template: np.array,
        front: int,
        back: int,
        cluster_labels: np.array,
        savefig: bool = False,
    ) -> None:
        signal_len, n_leads = original_signal.shape

        fig, ax = plt.subplots(n_leads, 4, figsize=(50, 60))
        plt.title("QRST-cancellation using Adaptive Singluar Value Cancellation", fontsize=30)
        if n_leads > 1:
            for lead in range(n_leads):
                ax[lead, 0].plot(original_signal[:, lead], label="original")
                ax[lead, 0].scatter(r_peaks, original_signal[r_peaks, lead], marker="o", color="red", label="r-peaks")
                ax[lead, 0].set_title(f"lead_{lead + 1} Original ECG")
                ax[lead, 0].legend()

                ax[lead, 1].plot(transformed_signal[:, lead])
                ax[lead, 1].set_title("Estimated AA")

                ax[lead, 2].plot(
                    original_signal[:, lead] - transformed_signal[:, lead],
                )
                ax[lead, 2].set_title("Estimated VA (orig- Estimated AA)")

                ax[lead, 3].plot(template[lead, :, :].T)
                ax[lead, 3].set_title(f"lead_{lead+1}-templates")

                for w, peak in enumerate(r_peaks):
                    for j in range(3):
                        add_intensity = cluster_labels[lead, w] / 5
                        ax[lead, j].axvspan(
                            peak - front,
                            peak + back,
                            facecolor="gray",
                            alpha=0.2 + add_intensity,
                            label="considered window",
                        )
                        ax[lead, j].text(x=peak - front, y=transformed_signal[:, lead].max() * 0.8, s=str(w + 1))
            if savefig:
                plt.savefig("./asvc.png")
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
                for w, peak in enumerate(r_peaks):
                    ax[j].axvspan(peak - front, peak + back, facecolor="gray", alpha=0.2, label="considered window")
                    ax[j].text(peak - front, -0.5, str(w + 1), fontsize="x-small")

        plt.show()

    def _get_template(
        self,
        windowed_signal: np.array,
        plot_templates: bool = True,
        use_clustering: bool = True,
        min_cluster_size: int = 5,
    ) -> np.array:

        n_leads, n_windows, window_size = windowed_signal.shape

        # template = np.zeros(shape=(n_leads, window_size))
        template = np.zeros_like(windowed_signal)

        # np.linalg.svd can be vectorized // only if we do not take subsets
        cluster_labels = np.zeros(shape=(n_leads, n_windows))
        for lead in range(n_leads):
            if use_clustering:
                cluster_labels[lead, :] = self._cluster_complexes(
                    windowed_signal[lead, :, :], min_cluster_size=min_cluster_size
                )
                logging.info(
                    f"Clustering lead {lead+1}:"
                    f" into {cluster_labels[lead, :].sum(), n_windows-cluster_labels[lead, :].sum()}"
                )
            else:
                cluster_labels[lead, :] = np.zeros(shape=(n_windows,))

            for i in range(2):
                cluster_size = (cluster_labels[lead, :] == i).sum()
                cluster_idxs = cluster_labels[lead, :] == i
                U, _, _ = np.linalg.svd(windowed_signal[lead, cluster_idxs, :].T)
                template[lead, cluster_idxs, :] = np.broadcast_to(U[:, 0], shape=(cluster_size, window_size))

        return template, cluster_labels

    def _cluster_complexes(self, windows: np.array, min_cluster_size: int) -> np.array:
        signals = windows[:, :]
        # dists = cdist(signals, signals, metric ="correlation")
        model = SpectralClustering(n_clusters=2).fit(signals)
        if model.labels_.sum() < min_cluster_size or len(model.labels_) - model.labels_.sum() < min_cluster_size:
            labels = np.zeros_like(model.labels_)
        else:
            labels = model.labels_
        return labels

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

        # N = 100
        # weights = gaussian(2*N, std = 2*np.sqrt(N))
        # template_aligned[:,:,:N] *=weights[:N]
        # template_aligned[:, :, -N:] *= weights[-N:]

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

                # Ensuring "np.roll" does not create artifacts; values rolled over from the oter end of the array are
                # replaced more adequately by the last feasible value
                if shift > 0:
                    template_shifted[lead, window, :shift] = template_shifted[lead, window, shift]
                elif shift < 0:
                    template_shifted[lead, window, shift:] = template_shifted[lead, window, shift - 1]
                else:
                    pass

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

        # N = 100
        # weights = gaussian(2*N, std = 2*np.sqrt(N))
        # template_aligned[:,:,:N] *=weights[:N]
        # template_aligned[:, :, -N:] *= weights[-N:]
        return template_aligned

    def _subtract_template(
        self,
        windowed_signal: np.array,
        template: np.array,
        P: int,
        M: int,
        smooth_transitions: bool = True,
    ) -> np.array:
        template = template.copy()
        n_leads, n_windows, window_size = windowed_signal.shape

        aa_signal = windowed_signal.copy()

        starts_ends = np.zeros(shape=(n_leads, n_windows, 2), dtype="int")

        for lead in range(n_leads):
            for window in range(n_windows):

                diff = np.abs(windowed_signal[lead, window, :] - template[lead, window, :])
                start = np.argmin(diff[:P])
                end = np.argmin(diff[-P:]) + window_size - P

                template[lead, window, :start] = 0
                template[lead, window, end:] = 0

                aa_signal[lead, window, :] -= template[lead, window, :]

                starts_ends[lead, window, :] = np.array([start, end])

                # post-process
                # if start >= 1:
                #    M_ = min(M, start)
                #    gaussian_window = gaussian(2 * M_, np.sqrt(M_))
                #    ks = (aa_signal[lead, window, start - 1] - aa_signal[lead, window, start]) / 2
                #    aa_signal[lead, window, start - M_ : start] -= ks * gaussian_window[:M_]
                #    aa_signal[lead, window, start : start + M_] += ks * gaussian_window[M_:]
        #
        # if end < window_size - 1:
        #    M_ = min(M, window_size - end - 1)
        #    gaussian_window = gaussian(2 * M_, np.sqrt(M_))
        #    ke = (aa_signal[lead, window, end] - aa_signal[lead, window, end + 1]) / 2
        #    aa_signal[lead, window, end - M_ + 1 : end + 1] -= ke * gaussian_window[:M_]
        #    aa_signal[lead, window, end + 1 : end + M_ + 1] += ke * gaussian_window[M_:]
        #
        return aa_signal, starts_ends

    def _reconstruct(
        self,
        aa_signal: np.array,
        original_signal: np.array,
        r_peaks: np.array,
        front: int,
        back: int,
        starts_ends: np.array,
        M: int,
    ) -> np.array:
        reconstructed_signal = original_signal.copy()
        last_window_end_idx = 0

        for window, peak in enumerate(r_peaks):
            window_start = peak - front
            window_end = peak + back + 1

            if window_start > last_window_end_idx + 1:
                gap = reconstructed_signal[last_window_end_idx + 1 : window_start, :]
                gap -= self._linreg(data=gap)
                reconstructed_signal[last_window_end_idx + 1 : window_start, :] = gap

            last_window_end_idx = window_end

            reconstructed_signal[window_start:window_end, :] = aa_signal[:, window, :].T

        if last_window_end_idx < reconstructed_signal.shape[0]:
            gap = reconstructed_signal[last_window_end_idx + 1 :, :]
            gap -= self._linreg(data=gap)
            reconstructed_signal[last_window_end_idx + 1 :, :] = gap

        for window, peak in enumerate(r_peaks):
            window_start = peak - front
            window_end = peak + back + 1

            for lead in range(reconstructed_signal.shape[1]):
                start = window_start + starts_ends[lead, window, 0]
                end = window_start + starts_ends[lead, window, 1]

                M_ = M
                gaussian_window = gaussian(2 * M_, np.sqrt(2 * M_))

                if start > M_:
                    ks = (reconstructed_signal[start - 1, lead] - reconstructed_signal[start, lead]) / 2
                    reconstructed_signal[start - M_ : start, lead] -= ks * gaussian_window[:M_]
                    reconstructed_signal[start : start + M_, lead] += ks * gaussian_window[M_:]

                if end < reconstructed_signal.shape[0] - M_:
                    ke = (reconstructed_signal[end, lead] - reconstructed_signal[end + 1, lead]) / 2
                    reconstructed_signal[end - M_ + 1 : end + 1, lead] -= ke * gaussian_window[:M_]
                    reconstructed_signal[end + 1 : end + M_ + 1, lead] += ke * gaussian_window[M_:]

        return reconstructed_signal

    def _linreg(self, data: np.array) -> np.array:

        length, dims = data.shape

        features = np.vstack([np.arange(1, length + 1), np.ones(length)]).T
        lstq_results = np.linalg.lstsq(a=features, b=data)[0]

        fitted_data = np.dot(features, lstq_results)

        return fitted_data


if __name__ == "__main__":
    from scipy.io import loadmat

    from signal_processing import detqrs3

    data = loadmat("../tests/data/detqrs_data.mat")
    # qrs_locs = data["qrs_locs"].squeeze() - 1
    data_centered = data["data_centered"].T
    data_centered = data_centered - data_centered.mean(axis=0)
    fs = 1000

    qrs_locs = detqrs3(data_centered[:, 9], fs)  # get_r_peaks(data_centered[:,0], fs)
    asvc = ASVCancellator()

    data_af = asvc(
        original_signal=data_centered,
        r_peaks=qrs_locs[1:],
        verbose=True,
        fit="normal",
        P=40,
        M=20,
        use_clustering=False,
        savefig=True,
    )
