import logging
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import filtfilt
from scipy.signal.windows import gaussian
from sklearn.cluster import SpectralClustering

from signal_processing.fixed_window_signal_splitting import split_signal

logging.basicConfig(level=logging.INFO)


class ASVCancellator:
    def __init__(
        self,
        with_shift: bool = True,
        use_clustering: bool = False,
        min_cluster_size: Optional[int] = None,
        smooth_template: Optional[int] = None,
        pos_neg_fit: bool = False,
        smooth_transitions: bool = True,
        P: int = 40,
        M: int = 20,
    ):

        self.with_shift = with_shift
        self.use_clustering = use_clustering
        self.min_cluster_size = min_cluster_size
        self.smooth_template = smooth_template
        self.pos_neg_fit = pos_neg_fit
        self.smooth_transitions = smooth_transitions
        self.P = P
        self.M = M

    def reconstruct(self, *args: Any, **kwargs: Any) -> np.array:
        return self.__call__(*args, **kwargs)

    def __call__(
        self,
        original_signal: np.array,
        r_peaks: np.array,
        verbose: bool = False,
        plot_all: bool = True,
        savefig: bool = False,
        plot_single_windows: Optional[List[Tuple[int, int]]] = None,
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
        rr_windows = split_signal(signal=signal_padded, r_peaks=r_peaks_shifted, front=front, back=back)

        # Create Template (lead by lead)
        # Find subset of windows to use (e.g. by similarity, neighbours, clustering)
        min_cluster_size = int(rr_windows.shape[1] / 4) if self.min_cluster_size is None else self.min_cluster_size
        template, cluster_labels = self._get_template(
            rr_windows, plot_templates=verbose, use_clustering=self.use_clustering, min_cluster_size=min_cluster_size
        )

        if self.smooth_template is not None:
            h = np.ones(self.smooth_template) / self.smooth_template
            template = filtfilt(h, 1, template, axis=2)

        # Fit template to window
        template_fitted = self._fit_template_to_windows(
            rr_windows, template, verbose=verbose, pos_neg_fit=self.pos_neg_fit
        )

        # fit transitions
        aa_signal, starts_ends = self._subtract_template(
            windowed_signal=rr_windows, template=template_fitted, P=self.P, M=self.M, smooth_transitions=False
        )

        # Fit to original signal shape and smooth transitions
        aa_signal_reconstructed = self._reconstruct(
            aa_signal, signal_padded, r_peaks_shifted, front, back, starts_ends, M=self.M
        )

        if pad_front > 0:
            aa_signal_reconstructed = aa_signal_reconstructed[pad_front:]

        if pad_back > 0:
            aa_signal_reconstructed = aa_signal_reconstructed[:-pad_back]

        # Evaluate (optionally)
        if plot_all:
            self._plot_all(
                original_signal,
                aa_signal_reconstructed,
                r_peaks,
                template_fitted,
                front,
                back,
                cluster_labels,
                savefig=savefig,
            )

        if plot_single_windows is not None:
            for lead, window in plot_single_windows:
                self._plot_window(
                    lead=lead,
                    window=window,
                    windowed_signal=rr_windows,
                    template=template,
                    template_fitted=template_fitted,
                )

        return aa_signal_reconstructed

    def _plot_window(
        self, lead: int, window: int, windowed_signal: np.array, template: np.array, template_fitted: np.array
    ) -> None:

        lead -= 1
        window -= 1

        plt.figure(figsize=(15, 8))
        plt.plot(range(len(template[lead, window, :])), template[lead, window, :], label="Template")
        plt.plot(range(len(windowed_signal[lead, window, :])), windowed_signal[lead, window, :], label="Original data")
        plt.plot(
            range(len(template_fitted[lead, window, :])), template_fitted[lead, window, :], label="Fitted Template (VA)"
        )
        plt.plot(
            range(len(template_fitted[lead, window, :])),
            windowed_signal[lead, window, :] - template_fitted[lead, window, :],
            color="red",
            label="Diff (AA)",
        )
        plt.legend()
        plt.grid()
        plt.title(f"Lead {lead+1}, window {window+1} | {str(self)}", fontsize="xx-large")
        plt.savefig(f"./{str(self)}.png")
        plt.show()

    def _plot_all(
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
        plt.title("QRST-cancellation using" + str(self), fontsize="x-large")
        if n_leads > 1:
            for lead in range(n_leads):
                ax[lead, 0].plot(original_signal[:, lead], label="original")
                ax[lead, 0].scatter(r_peaks, original_signal[r_peaks, lead], marker="o", color="red", label="r-peaks")
                ax[lead, 0].set_title(f"lead_{lead + 1} Original ECG", fontsize="large")
                ax[lead, 0].legend()

                ax[lead, 1].plot(transformed_signal[:, lead])
                ax[lead, 1].set_title("Estimated AA", fontsize="large")

                ax[lead, 2].plot(
                    original_signal[:, lead] - transformed_signal[:, lead],
                )
                ax[lead, 2].set_title("Estimated VA (orig- Estimated AA)", fontsize="large")

                ax[lead, 3].plot(template[lead, :, :].T)
                ax[lead, 3].set_title(f"lead_{lead+1}-templates", fontsize="large")

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
            ax[1].set_title("AA", fontsize="large")

            ax[2].plot(
                original_signal - transformed_signal,
            )
            ax[2].set_title("VA (orig-AA)", fontsize="large")

            ax[3].plot(template[0, 0, :])
            ax[3].set_title("lead-template", fontsize="large")

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

    def _fit_template_to_windows(
        self, windowed_signal: np.array, template: np.array, verbose: bool = False, pos_neg_fit: bool = False
    ) -> np.array:
        assert windowed_signal.shape == template.shape
        n_leads, n_windows, window_size = windowed_signal.shape

        templates_aligned = template.copy()

        for lead in range(n_leads):
            for window in range(n_windows):
                if self.with_shift:
                    corr = np.correlate(
                        np.abs(windowed_signal[lead, window, :]), np.abs(template[lead, window, :]), mode="same"
                    )
                    shift = np.argmax(corr) - window_size // 2
                    templates_aligned[lead, window, :] = np.roll(templates_aligned[lead, window, :], shift=shift)

                    # Ensuring "np.roll" does not create artifacts; values rolled over from the oter end of the array
                    # are replaced more adequately by the last feasible value
                    if shift > 0:
                        templates_aligned[lead, window, :shift] = templates_aligned[lead, window, shift]
                    elif shift < 0:
                        templates_aligned[lead, window, shift:] = templates_aligned[lead, window, shift - 1]
                    else:
                        pass

                if pos_neg_fit:
                    template_pos = np.where(
                        templates_aligned[lead, window, :] >= 0, templates_aligned[lead, window, :], 0
                    )
                    template_neg = np.where(
                        templates_aligned[lead, window, :] < 0, templates_aligned[lead, window, :], 0
                    )
                    design_matrix = np.stack([template_pos, template_neg, np.ones_like(template[lead, window, :])]).T
                else:
                    design_matrix = np.stack(
                        [templates_aligned[lead, window, :], np.ones_like(template[lead, window, :])]
                    ).T

                lstq_results = np.linalg.lstsq(a=design_matrix, b=windowed_signal[lead, window, :].T)
                templates_aligned[lead, window, :] = np.dot(design_matrix, lstq_results[0])

                if verbose:
                    info_msg = (
                        f"Fitting lead {lead+1},"
                        f" window {window},"
                        f" coeffs = {lstq_results[0]},"
                        f" residual = {lstq_results[1]},"
                    )
                    if self.with_shift:
                        info_msg += f" shift {shift}"
                    logging.info(info_msg)

        return templates_aligned

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

        if self.smooth_transitions:
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

    def __str__(self) -> str:
        string_repr: List[str] = []
        if self.with_shift:
            string_repr.append("Shift")
        if self.use_clustering:
            string_repr.append(" Clustering")
        if self.pos_neg_fit:
            string_repr.append(" Pos-Neg fit")

        concatenator = "" if len(string_repr) == 0 else "with "

        return "ASCV " + concatenator + ", ".join(string_repr)


if __name__ == "__main__":
    from scipy.io import loadmat

    from signal_processing import detqrs3

    data = loadmat("../tests/data/detqrs_data.mat")
    # qrs_locs = data["qrs_locs"].squeeze() - 1
    data_centered = data["data_centered"].T
    data_centered = data_centered - data_centered.mean(axis=0)
    fs = 1000

    qrs_locs = detqrs3(data_centered[:, 0], fs)  # get_r_peaks(data_centered[:,0], fs)
    asvc = ASVCancellator(
        with_shift=False,
        P=40,
        M=20,
        use_clustering=False,
        pos_neg_fit=False,
        smooth_template=None,
    )

    data_af = asvc(
        original_signal=data_centered[:, :5],
        r_peaks=qrs_locs[1:],
        verbose=True,
        savefig=False,
        plot_all=False,
        plot_single_windows=[(5, 10)],
    )
