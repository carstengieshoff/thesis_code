from typing import Optional

import numpy as np

from signal_processing import split_signal


def get_K95(signal: np.array, Fs: int) -> float:

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


if __name__ == "__main__":
    from signals import AAGP
    from signals.GP_kernels import unorganized_aa_args

    Fs = 200
    aa = AAGP(unorganized_aa_args, sec=10, sampling_rate=Fs)

    aa.generate(num_samples=12)

    print(f"$K_95$: {get_K95(signal= aa.data, Fs=Fs)}")
    print(f"$NMSE_3$: {get_NMSE(signal=aa.data, Fs=Fs,lead=0, k=3)}")
