import numpy as np
import pytest

from embeddings.lag_emebedding import LagEmbedding


@pytest.mark.parametrize("size", [100, 101])
@pytest.mark.parametrize("lag", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
def test_lag_embedding_for_1d_signals(size: int, lag: int, dim: int) -> None:
    # given
    signal = np.random.random(size=(size, 1))
    embedding = LagEmbedding(lag=lag, dim=dim)
    # when
    embedded_signal = embedding.embedd(signal=signal)

    # then
    if dim == 1:
        expected_shape = signal.shape
    else:
        expected_shape = (signal.shape[0] - lag * dim + 1, dim)

    # then
    assert embedded_signal.shape == expected_shape
    #  -1 ist the last position (and the first dimension), dim-1 lags still miss and mus be added (subtracted)
    expected_last_embedding = signal[lag * (1 - dim) - 1 :: lag, :].reshape(-1, 1)
    actual_last_embedding = embedded_signal[-1, :].reshape(-1, 1)
    assert np.array_equal(actual_last_embedding, expected_last_embedding)


@pytest.mark.parametrize("size", [100, 101])
@pytest.mark.parametrize("lag", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("signal_dim", [2, 3, 10])
def test_lag_embedding_for_2d_signals(size: int, lag: int, dim: int, signal_dim: int) -> None:
    # given
    signal = np.random.random(size=(size, signal_dim))
    embedding = LagEmbedding(lag=lag, dim=dim)
    # when
    embedded_signal = embedding.embedd(signal=signal)

    # then
    if dim == 1:
        expected_shape = signal.shape
    else:
        expected_shape = (signal.shape[0] - lag * dim + 1, signal_dim, dim)

    # then
    assert embedded_signal.shape == expected_shape
    #  -1 ist the last position (and the first dimension), dim-1 lags still miss and mus be added (subtracted)
    expected_last_embedding = signal[lag * (1 - dim) - 1 :: lag, :].T.squeeze()
    actual_last_embedding = embedded_signal[-1, :]
    assert np.array_equal(actual_last_embedding, expected_last_embedding)
