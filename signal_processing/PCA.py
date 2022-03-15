from typing import Optional

import numpy as np


def PCA(signal: np.ndarray, num_pcs: int = 1, var: Optional[float] = None) -> np.ndarray:
    """Perfrom PCA on signal.

    Args:
        signal: Signal as `np.ndarray` of shape (len_signal, dim_signal).
        num_pcs: Number of PCs to keep in the returned signal.
        var: Optional amount of variance to keep in the output signal. If this is specified, it overwrites `num_pcs`.

    Returns:
        Signal as `np.ndarray` of same shape as input `signal`, reduced to the sepcified number of PCs.
    """
    signal = signal - signal.mean(axis=0)  # / signal.std(axis=0)
    U, S, V = np.linalg.svd(signal)

    if var:
        cumulative_rel_pcs = np.cumsum(S / S.sum())
        num_pcs = np.where(cumulative_rel_pcs >= var)[0].min() + 1

    return U[:, :num_pcs] @ np.diag(S[:num_pcs])
