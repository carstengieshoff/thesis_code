from typing import Any, Literal, Union, get_args

import numpy as np
from scipy.stats import mode
from teaspoon.parameter_selection.FNN_n import FNN_n

FNN_type = Literal["mean", "mode", "complete"]


def fnn(signal: np.array, lag: int, fnn_type: FNN_type = "mean", *args: Any, **kwargs: Any) -> Union[int, np.array]:
    """Calculating mutual information on each dimension of a (multivariate) signal.

    For a signal of shape (n_samples, n_dims), `MI_for_delay` from `teaspoon` to each of the n_dims signals of shape
    (n_samples, 1). The mode and mean of the n_dims returned delays are returned.

    """
    n_dims = signal.shape[1]
    embedding_dims = np.zeros(shape=n_dims, dtype=int)
    for dim in range(n_dims):
        embedding_dims[dim] = FNN_n(signal[:, dim], tau=lag, *args, **kwargs)[1]

    if fnn_type == "mean":
        return np.round(embedding_dims.mean()).item()
    elif fnn_type == "mode":
        return mode(embedding_dims)[0].item()
    elif fnn_type == "complete":
        return embedding_dims
    else:
        valid_args = [f"`{arg}`" for arg in get_args(FNN_type)]
        raise RuntimeError(f"Unknown value `{fnn_type}` for `mi_type`, expected one of {', '.join(valid_args)}")


if __name__ == "__main__":
    from embeddings.utils.mutual_information import mutual_information
    from signals.artificial_signals import Chirp

    c = Chirp(frequency_start=5, frequency_end=10, sampling_rate=100, sec=5, noise_rate=0.5)
    lag = mutual_information(c.generate(), mi_type="mean", plotting=True, ranking=True)
    print(fnn(c.generate(), lag=lag, fnn_type="mean", plotting=True))
