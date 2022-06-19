from typing import Optional

import numpy as np
from scipy.stats import entropy

from embeddings.utils import spectral_envelope
from signal_processing import split_signal


def get_K95(signal: np.array, Fs: int) -> float:
    """Implementing k_95 from [Bonizzi, 2010].

    Args:
        signal: Signal to analyse
        Fs: sampling rate
    """
    n, dims = signal.shape

    windowed_signal = split_signal(
        signal=signal, r_peaks=np.arange(0, n, Fs), front=0, back=Fs - 1
    )  # dims, num_windows_ window_length

    num_windows = windowed_signal.shape[1]

    ks = []
    for s in range(num_windows):
        U, S, V = np.linalg.svd(windowed_signal[:, s, :])
        cumulative_rel_pcs = np.cumsum(S / S.sum())
        num_pcs = np.where(cumulative_rel_pcs >= 0.95)[0].min() + 1
        ks.append(num_pcs)

    return np.array(ks).mean(dtype="float32")


def get_NMSE(signal: np.array, Fs: int, lead: int = 1, k: Optional[int] = None) -> float:
    """Implementing NMSE from [Bonizzi, 2010].

    Args:
        signal: Signal to analyse
        Fs: sampling rate
        lead: lead to evaluate NSME on
        k: Components to use. If None takes number of pcs to describe 95% of variance.
    """
    n, dims = signal.shape

    windowed_signal = split_signal(
        signal=signal, r_peaks=np.arange(0, n, Fs), front=0, back=Fs - 1
    )  # dims, num_windows_ window_length

    num_windows = windowed_signal.shape[1]

    U, S, V = np.linalg.svd(windowed_signal[:, 0, :])
    if k is None:
        cumulative_rel_pcs = np.cumsum(S / S.sum())
        num_pcs = np.where(cumulative_rel_pcs >= 0.95)[0].min() + 1
    else:
        num_pcs = k

    M1 = U[:, :num_pcs] @ np.diag(S[:num_pcs])
    M1_inv = np.linalg.pinv(M1)

    NMSEs = []
    for s in range(1, num_windows):

        y = windowed_signal[:, s, :]
        y_hat = M1 @ M1_inv @ y

        y_hat = y_hat[lead]
        y = y[lead]

        nmse = np.power(y - y_hat, 2).sum() / np.power(y, 2).sum()
        NMSEs.append(nmse)

    return np.array(NMSEs).mean(dtype="float32")


def get_MOI(signal: np.array, Fs: int, h: Optional[np.array] = None) -> float:
    """Implementing MOI from [Uldry, 2012].

    Args:
        signal: Signal to analyse
        Fs: sampling rate
    """
    n, dims = signal.shape

    windowed_signal = split_signal(
        signal=signal, r_peaks=np.arange(0, n, 5 * Fs), front=0, back=10 * Fs - 1
    )  # dims, num_windows_ window_length

    num_windows = windowed_signal.shape[1]

    ses = []
    for s in range(num_windows):
        se = spectral_envelope(signal=windowed_signal[:, s, :].T, h=h)
        se = se / se.sum()
        ses.append(se)

    ses = np.vstack(ses).mean(axis=0)
    ses = ses / ses.sum()
    peaks = np.sort(np.argsort(ses)[-5:])

    dF = Fs / signal.shape[0]
    window_side = int(0.5 * 1 / dF)

    area_under_peaks = 0
    end = 0
    for peak in peaks:
        start = max(end, peak - window_side)  # no-overlap
        # start = max(0, peak - window_side)  # overlap
        end = min(len(ses), peak + window_side)
        area_under_peaks += ses[start:end].sum()

    return area_under_peaks


def get_MSE(signal: np.array, Fs: int, h: Optional[np.array] = None) -> float:
    """Implementing MOI from [Uldry, 2012].

    Args:
        signal: Signal to analyse
        Fs: sampling rate
    """
    n, dims = signal.shape

    windowed_signal = split_signal(
        signal=signal, r_peaks=np.arange(0, n, 5 * Fs), front=0, back=10 * Fs - 1
    )  # dims, num_windows_ window_length

    num_windows = windowed_signal.shape[1]

    ses = []
    for s in range(num_windows):
        se = spectral_envelope(signal=windowed_signal[:, s, :].T, h=h)
        se = se / se.sum()
        ses.append(se)

    ses = np.vstack(ses).mean(axis=0)
    ses = ses / ses.sum()

    return entropy(pk=ses)


if __name__ == "__main__":
    from signals import AAGP
    from signals.GP_kernels import organized_aa_args, unorganized_aa_args

    Fs = 200
    aa = AAGP(unorganized_aa_args, sec=20, sampling_rate=Fs)

    aa.generate(num_samples=12)

    # print(f"$K_95$: {get_K95(signal= aa.data, Fs=Fs)}")
    # print(f"$NMSE_3$: {get_NMSE(signal=aa.data, Fs=Fs,lead=0, k=3)}")
    print(f"$MOI$: {get_MOI(signal=aa.data, Fs=Fs)}")
    print(f"$MSE$: {get_MSE(signal=aa.data, Fs=Fs)}")

    Fs = 200
    aa = AAGP(organized_aa_args, sec=20, sampling_rate=Fs)

    aa.generate(num_samples=12)

    # print(f"$K_95$: {get_K95(signal= aa.data, Fs=Fs)}")
    # print(f"$NMSE_3$: {get_NMSE(signal=aa.data, Fs=Fs,lead=0, k=3)}")
    print(f"$MOI$: {get_MOI(signal=aa.data, Fs=Fs)}")
    print(f"$MSE$: {get_MSE(signal=aa.data, Fs=Fs)}")
