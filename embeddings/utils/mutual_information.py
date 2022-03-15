from typing import Any, Union

import numpy as np
from scipy.stats import mode
from teaspoon.parameter_selection.MI_delay import MI_for_delay


def mutual_information(signal: np.array, mi_type: str = "mean", *args: Any, **kwargs: Any) -> Union[int, np.array]:
    """Calculating mutual information on each dimension of a (multivariate) signal.

    For a signal of shape (n_samples, n_dims), `MI_for_delay` from `teaspoon` to each of the n_dims signals of shape
    (n_samples, 1). The mode and mean of the n_dims delays can be returned.

    """
    n_dims = signal.shape[1]
    delays = np.zeros(shape=n_dims, dtype=int)
    for dim in range(n_dims):
        delays[dim] = MI_for_delay(signal[:, dim], *args, **kwargs)

    if mi_type == "mean":
        return np.round(delays.mean()).astype(int).item()
    elif mi_type == "mode":
        return mode(delays)[0].astype(int).item()
    elif mi_type == "complete":
        return delays.astype(int)
    else:
        valid_args = [f"`{arg}`" for arg in ["mean", "mode", "complete"]]
        raise RuntimeError(f"Unknown value `{mi_type}` for `mi_type`, expected one of {', '.join(valid_args)}")


if __name__ == "__main__":
    from signals.artificial_signals import Chirp

    c = Chirp(frequency_start=5, frequency_end=10, sampling_rate=100, sec=5, noise_rate=0.5)
    print(mutual_information(c.generate(), mi_type="mean", plotting=True, ranking=True))
