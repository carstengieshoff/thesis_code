import logging
from typing import Any, List, Optional

import numpy as np
from scipy.signal import filtfilt
from scipy.signal.windows import gaussian
from sklearn.cluster import SpectralClustering

from signal_processing.fixed_window_signal_splitting import split_signal
from visualizations import plot_ecg_plotly

logging.basicConfig(level=logging.INFO)


class ASVCancellator:
    """Implementing QRST-Cancellation according to https://iopscience.iop.org/article/10.1088/0967-3334/29/12/001.

    Args:
        with_shift: Boolean indicating whether to allow shifting of templates to better fit the individual windows.
        use_clustering: Boolean indicating whether to cluster QRST complexes before calculating templates.
        min_cluster_size: Only used if `use_clustering` is True. Minimal amount of QRST complexes to form a cluster.
        pos_neg_fit: Boolean indicating whether to fit positiveand negative parts of the signal individuall
        smooth_transitions: Boolean indicating whether to smooth signals at border between windows.,
        M: Window size (one-side) to use for signal smoothing. To be provided as milliseconds.
        post_processing_threshold: Whether to apply post-processing to reduce r-peak-residuals
        post_processing_type: Post-processing type to use. Defaults to "gaussian".
        H: Window size (one-sided) to consider for post-processing. To be provided as milliseconds. This defaults to 50
        fs: The underlying sampling rate of the signals to be  processed.
        front: Amount of milliseconds in front of r-peak to start QRST-Windows at.
        back: Percentage of minimal RR-distance to use after r-peak for QRST-Windows. This defaults to 0.6.
        P:
    """

    def __init__(
        self,
        with_shift: bool = True,
        use_clustering: bool = False,
        min_cluster_size: Optional[int] = None,
        pos_neg_fit: bool = False,
        smooth_transitions: bool = True,
        post_processing_threshold: Optional[float] = None,
        post_processing_type: str = "gaussian",
        fs: int = 500,
        M: int = 40,
        H: int = 50,
        front: int = 50,
        back: float = 0.6,
    ):

        self.with_shift = with_shift
        self.use_clustering = use_clustering
        self.min_cluster_size = min_cluster_size
        self.pos_neg_fit = pos_neg_fit
        self.smooth_transitions = smooth_transitions
        self.fs = fs
        self.M = int(M / 1000 * self.fs)
        self.H = int(H / 1000 * self.fs)
        self.post_processing_threshold = post_processing_threshold
        self.post_processing_type = post_processing_type

        self.front = front
        self.back = back

    def reconstruct(self, *args: Any, **kwargs: Any) -> np.array:
        return self.__call__(*args, **kwargs)

    def __call__(
        self,
        original_signal: np.array,
        r_peaks: np.array,
        verbose: bool = False,
        plot_all: bool = False,
        savefig: bool = False,
        plot_templates: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> np.array:

        # PreProcessing w.r.t R-peaks
        r_peak_dist = r_peaks[1:] - r_peaks[:-1]
        r_peak_min = r_peak_dist.min()

        front = int(self.front / 1000 * self.fs)
        back = int(self.back * r_peak_min)

        # Pad signal
        pad_front = max(0, front - r_peaks.min())
        pad_back = max(0, r_peaks.max() + back - original_signal.shape[0] + 1)
        r_peaks_shifted = r_peaks + pad_front

        signal_padded = np.vstack(
            [
                original_signal[r_peaks[1] - front : r_peaks[1] - front + pad_front, :],
                original_signal,
                original_signal[r_peaks[-2] + back - pad_back : r_peaks[-2] + back, :],
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

        # Fit template to window
        template_fitted = self._fit_template_to_windows(
            rr_windows, template, verbose=verbose, pos_neg_fit=self.pos_neg_fit
        )

        # fit transitions
        aa_signal = self._subtract_template(windowed_signal=rr_windows, template=template_fitted)

        # Fit to original signal shape and smooth transitions
        aa_signal_reconstructed = self._reconstruct(aa_signal, signal_padded, r_peaks_shifted, front, back, M=self.M)

        if pad_front > 0:
            aa_signal_reconstructed = aa_signal_reconstructed[pad_front:]

        if pad_back > 0:
            aa_signal_reconstructed = aa_signal_reconstructed[:-pad_back]

        if self.post_processing_threshold is not None:
            if verbose:
                plot_ecg_plotly(
                    original=original_signal,
                    aa=aa_signal_reconstructed,
                    peaks=r_peaks,
                    front=front,
                    back=back,
                    H=self.H,
                    title="QRST-cancellation pre-post processing",
                )

            aa_signal_reconstructed = post_process(
                aa_signal_reconstructed,
                r_peaks=r_peaks,
                threshold=self.post_processing_threshold,
                H=self.H,
                type=self.post_processing_type,
                front=front,
                back=back,
            )
        # Plot (optionally)
        if plot_all:
            plot_ecg_plotly(
                original=original_signal,
                aa=aa_signal_reconstructed,
                peaks=r_peaks,
                front=front,
                back=back,
                H=self.H,
                title="QRST-cancellation using" + str(self),
            )

        if plot_templates:
            plot_ecg_plotly(original=template_fitted[:, 0, :].T, title="Templates")

        # Evaluate (optionally)
        # VR = self._evaluate_VR(windowed_signal=aa_signal)
        return aa_signal_reconstructed

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
        self,
        windowed_signal: np.array,
        template: np.array,
        verbose: bool = False,
        pos_neg_fit: bool = False,
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

                X = design_matrix.copy()
                b = windowed_signal[lead, window, :].T.copy()

                lstq_results = np.linalg.lstsq(a=X, b=b)
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
    ) -> np.array:
        aa_signal = windowed_signal - template
        return aa_signal

    def _reconstruct(
        self,
        aa_signal: np.array,
        original_signal: np.array,
        r_peaks: np.array,
        front: int,
        back: int,
        M: int,
    ) -> np.array:
        reconstructed_signal = original_signal.copy()
        last_window_end_idx = 0

        for window, peak in enumerate(r_peaks):
            window_start = peak - front
            window_end = peak + back + 1

            if window_start > last_window_end_idx + 1:
                gap = reconstructed_signal[last_window_end_idx:window_start, :]
                gap -= self._linreg(data=gap)
                reconstructed_signal[last_window_end_idx:window_start, :] = gap

            last_window_end_idx = window_end

            reconstructed_signal[window_start:window_end, :] = aa_signal[:, window, :].T

        if last_window_end_idx < reconstructed_signal.shape[0]:
            gap = reconstructed_signal[last_window_end_idx:, :]
            gap -= self._linreg(data=gap)
            reconstructed_signal[last_window_end_idx:, :] = gap

        if self.smooth_transitions:
            for window, peak in enumerate(r_peaks):
                start = peak - front
                reconstructed_signal = close_gap(reconstructed_signal, start - 1, M)

                end = peak + back
                if end + 1 < reconstructed_signal.shape[0]:
                    reconstructed_signal = close_gap(reconstructed_signal, end, M)

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
        if self.post_processing_threshold is not None:
            string_repr.append(
                f" Post-Pr: (Threshold {self.post_processing_threshold}, Type: {self.post_processing_type})"
            )

        concatenator = "" if len(string_repr) == 0 else "with "

        return "ASCV " + concatenator + ", ".join(string_repr)


def evaluate_VR(aa_signal: np.array, r_peaks: np.array, H: int = 50, front: int = 100, back: int = 350) -> np.array:
    signal_len, n_leads = aa_signal.shape

    aa_signal = aa_signal.copy()
    aa_signal = aa_signal - aa_signal.mean(axis=0, keepdims=True)

    denom_ = []
    for peak in r_peaks:
        denom_.append(aa_signal[peak - front : peak - H, :])
        denom_.append(aa_signal[peak + H : peak + back, :])
    denom = np.concatenate(denom_)
    denom = np.power(denom, 2)
    denom = denom.mean(axis=0, keepdims=True)

    numerator = np.zeros(shape=(len(r_peaks), n_leads))
    for i, peak in enumerate(r_peaks):
        start = max(0, peak - H)
        end = min(signal_len, peak + H)

        numerator[i, :] = np.sqrt(np.power(aa_signal[start:end, :], 2).mean(axis=0)) * np.max(
            np.abs(aa_signal[start:end, :]), axis=0, keepdims=True
        )
    if n_leads > 1:
        return numerator / denom  # .mean(axis=0)
    else:
        return numerator / denom


def post_process(
    aa_signal: np.array,
    r_peaks: np.array,
    threshold: float = 2,
    H: int = 50,
    type: str = "factor",
    front: int = 100,
    back: int = 350,
) -> np.array:
    signal_len, n_leads = aa_signal.shape
    aa_signal = aa_signal.copy()
    scores = evaluate_VR(aa_signal=aa_signal, r_peaks=r_peaks, front=front, back=back)

    for i, peak in enumerate(r_peaks):
        start = max(0, peak - H)
        end = min(signal_len, peak + H)
        poor_leads = np.argwhere(scores[i, :] >= threshold)

        threshold = 1
        if type == "factor":
            aa_signal[start:end, poor_leads] *= 1 / (5 * threshold)
        elif type == "zero":
            aa_signal[start:end, poor_leads] = 0
        elif type == "gaussian":
            n = end - start
            weights = 1 - (5 * threshold - 1) / (5 * threshold) * gaussian(n, np.sqrt(2 * n))
            weights = np.repeat(weights, axis=0, repeats=len(poor_leads)).reshape(n, -1, 1)
            aa_signal[start:end, poor_leads] *= weights
        elif type == "linear":
            n = end - start
            gap = np.repeat(np.arange(0, n).reshape(1, -1), repeats=len(poor_leads), axis=0) / n
            gap = aa_signal[start, poor_leads] + gap * (aa_signal[end, poor_leads] - aa_signal[start, poor_leads])
            gap = gap.T  # reshape(aa_signal[start:end, poor_leads].shape)
            aa_signal[start:end, poor_leads] = np.expand_dims(
                gap, axis=2
            )  # .reshape(aa_signal[start:end, poor_leads].shape)
        else:
            raise ValueError("Unknown argument for `type`")

    return aa_signal


def close_gap(signal: np.array, pos: int, M: int = 10, alpha: float = 1.0) -> np.array:
    "Close gap between sample `pos` and `pos+1`"
    if len(signal.shape) == 1:
        signal = signal.reshape(-1, 1)
    M = min(M, pos + 1, signal.shape[0] - (pos + 1))

    gaussian_window = gaussian(2 * M, np.sqrt(M)).reshape(-1, 1)

    delta = alpha * (signal[pos, :] - signal[pos + 1, :]) / 2

    signal[pos + 1 - M : pos + 1, :] -= delta * gaussian_window[:M]
    signal[pos + 1 : pos + M + 1, :] += delta * gaussian_window[M:]

    return signal


if __name__ == "__main__":
    from scipy.io import loadmat
    from scipy.signal import cheby2

    from signal_processing import detqrs3

    # from visualizations import plot_filter, plot_spectral_envelope

    try:
        data = loadmat("./tests/data/detqrs_data.mat")
    except FileNotFoundError:
        data = loadmat("../tests/data/detqrs_data.mat")

    # qrs_locs = data["qrs_locs"].squeeze() - 1
    data_centered = data["data_centered"].T
    data_centered = data_centered - data_centered.mean(axis=0)
    fs = 1000

    # plot_spectral_envelope(data_centered, Fs=fs)

    # plt.plot(data_centered)
    # plt.show()

    # Create and review filter
    [b, a] = cheby2(3, 20, [1, 100], btype="bandpass", fs=fs)
    # plot_filter(b, a, fs)
    # plt.show()
    data_centered = filtfilt(b, a, data_centered, axis=0)

    [b, a] = cheby2(3, 20, 5, btype="highpass", fs=fs)
    # plot_filter(b, a, fs)
    # plt.show()
    data_centered = filtfilt(b, a, data_centered, axis=0)

    # Filter all signals and show example
    # plot_spectral_envelope(data_centered, Fs=fs)
    # plt.show()

    # plt.plot(data_centered)
    # plt.show()

    qrs_locs = detqrs3(data_centered[:, 0], fs)  # get_r_peaks(data_centered[:,0], fs)
    asvc = ASVCancellator(
        with_shift=True,
        M=20,
        use_clustering=False,
        pos_neg_fit=False,
        post_processing_threshold=4,
        post_processing_type="linear",
    )

    data = data_centered
    qrs_locs = detqrs3(data[:, 0], fs)
    for i in range(1):
        data_af = asvc(
            original_signal=data[:, :3],
            r_peaks=qrs_locs[:],
            verbose=False,
            savefig=True,
            plot_all=False,
            plot_single_windows=[],
        )

        # print(f"VR original signal: {evaluate_VR(data, r_peaks=qrs_locs)}")
        vr = evaluate_VR(data_af, r_peaks=qrs_locs)
        print(f"VR processed signal: {vr}")
