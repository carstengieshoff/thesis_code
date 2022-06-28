from typing import Dict, Optional

import numpy as np
from scipy.stats import entropy

from embeddings.utils import spectral_envelope
from signal_processing import split_signal


class ParamCalculator:
    def __init__(self, leads: Optional[slice] = None, lead: int = 0) -> None:
        self.leads = leads
        self.lead = lead
        pass

    def get_params(
        self,
        signal: np.array,
        Fs: int,
        h: Optional[np.array] = None,
        start_interval_every: int = 1.0,
        interval_length: int = 2.5,
    ) -> Dict[str, float]:

        if self.leads:
            x = signal[:, self.leads]
        else:
            x = signal

        params: Dict[str, float] = {}

        if x.shape[1] >= 2:
            params.update(
                self._get_svd_based_measures(x=x, Fs=Fs),
            )

        params.update(
            self._get_se_based_measures(
                x=x, Fs=Fs, h=h, start_interval_every=start_interval_every, interval_length=interval_length
            ),
        )
        params.update(
            self._get_further_coeffs(x=x, Fs=Fs, h=h),
        )

        return params

    def _get_svd_based_measures(self, x: np.array, Fs: int) -> Dict[str, float]:
        n, dims = x.shape

        windowed_signal = split_signal(
            signal=x, r_peaks=np.arange(0, n, Fs), front=0, back=Fs - 1
        )  # dims, num_windows_ window_length

        num_windows = windowed_signal.shape[1]

        ks = []
        for s in range(num_windows):
            U, S, V = np.linalg.svd(windowed_signal[:, s, :])
            cumulative_rel_pcs = np.cumsum(S / S.sum())
            num_pcs = np.where(cumulative_rel_pcs >= 0.95)[0].min() + 1
            ks.append(num_pcs)

            if s == 0:
                M1 = U[:, :3] @ np.diag(S[:3])
                M1_inv = np.linalg.pinv(M1)
                M_hat = M1 @ M1_inv

        NMSEs = []
        for s in range(1, num_windows):
            y = windowed_signal[:, s, :]
            y_hat = M_hat @ y

            y_hat = y_hat[self.lead]
            y = y[self.lead]

            nmse = np.power(y - y_hat, 2).sum() / np.power(y, 2).sum()
            NMSEs.append(nmse)

        return {
            "k_95": np.array(ks).mean(dtype="float32"),
            "NMSE_3": np.array(NMSEs).mean(dtype="float32"),
        }

    def _get_se_based_measures(
        self,
        x: np.array,
        Fs: int,
        h: Optional[np.array] = None,
        start_interval_every: int = 1.0,
        interval_length: int = 2.5,
    ) -> Dict[str, float]:
        n, dims = x.shape

        windowed_signal = split_signal(
            signal=x,
            r_peaks=np.arange(0, n, int(start_interval_every * Fs)),
            front=0,
            back=int(interval_length * Fs - 1),
        )  # dims, num_windows_ window_length

        num_windows = windowed_signal.shape[1]
        window_length = windowed_signal.shape[2]

        ses = []
        dafs = []
        for s in range(num_windows):
            se = spectral_envelope(signal=windowed_signal[:, s, :].T, h=h)
            se = se / se.sum()
            ses.append(se)

            max_idx = np.argmax(se)
            max_freq = max_idx / window_length * Fs
            dafs.append(max_freq)

        ses = np.vstack(ses).mean(axis=0)
        ses = ses / ses.sum()
        peaks = np.sort(np.argsort(ses)[-5:])

        dF = Fs / window_length
        window_side = int(0.5 * 1 / dF)

        area_under_peaks = 0
        end = 0
        for peak in peaks:
            start = max(end, peak - window_side)  # no-overlap
            # start = max(0, peak - window_side)  # overlap
            end = min(len(ses), peak + window_side)
            area_under_peaks += ses[start:end].sum()

        return {
            "MOI": area_under_peaks,
            "MSE": entropy(pk=ses),
            "Fib_mean": np.mean(dafs),
            "Fib_std": np.std(dafs),
        }

    def _get_further_coeffs(self, x: np.array, Fs: int, h: Optional[np.array] = None) -> np.array:

        n, dims = x.shape

        se = spectral_envelope(signal=x, h=h)
        se /= se.sum()
        max_idx = np.argmax(se)
        max_freq = max_idx / n * Fs

        dF = Fs / n
        window_side = int(0.5 * 1 / dF)

        harmonics = {}
        for i in range(2, 4):
            start = i * max_idx - window_side
            end = i * max_idx + window_side
            harmonics[i] = se[start:end].sum()

        power_per_lead = np.power(x, 2).mean(axis=0)

        return {"DAF": max_freq, "Fwave-power": power_per_lead.mean(), "harm2": harmonics[2], "harm3": harmonics[3]}


if __name__ == "__main__":
    from signals import AAGP
    from signals.GP_kernels import organized_aa_args, unorganized_aa_args

    Fs = 200
    aa = AAGP(organized_aa_args, sec=10, sampling_rate=Fs)

    aa.generate(num_samples=2)

    calculator = ParamCalculator(leads=slice(0, 4), lead=0)

    params = calculator.get_params(signal=aa.data, Fs=Fs, h=np.ones(5))

    print(params)

    aa = AAGP(organized_aa_args, sec=10, sampling_rate=400)
    aa.generate(num_samples=2)

    params = calculator.get_params(signal=aa.data, Fs=400, h=np.ones(5))

    print(params)

    print("Done")
