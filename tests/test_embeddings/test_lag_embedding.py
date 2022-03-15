import numpy as np
import pytest

from embeddings.lag_emebedding import LagEmbedding


@pytest.mark.parametrize("size", [100, 101])
@pytest.mark.parametrize("lag", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dim", [2, 3, 4, 5])
def test_lag_embedding_for_1d_signals(size: int, lag: int, dim: int) -> None:
    # given
    signal = np.random.random(size=(size, 1))
    embedding = LagEmbedding(lag=lag, dim=dim)

    # when
    embedded_signal = embedding.embedd(signal=signal)

    # then
    expected_shape = (signal.shape[0] - lag * dim + 1, dim)
    assert embedded_signal.shape == expected_shape
    #  -1 ist the last position (and the first dimension), dim-1 lags still miss and mus be added (subtracted)
    expected_last_embedding = signal[lag * (1 - dim) - 1 :: lag, :].reshape(-1, 1)
    actual_last_embedding = embedded_signal[-1, :].reshape(-1, 1)
    assert np.array_equal(actual_last_embedding, expected_last_embedding)
    assert np.array_equal(embedded_signal[:, -1].reshape(-1, 1), signal[lag * dim - 1 :, :])


@pytest.mark.parametrize("size", [100, 101])
@pytest.mark.parametrize("lag", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dim", [2, 3, 4, 5])
@pytest.mark.parametrize("signal_dim", [2, 3, 10])
def test_lag_embedding_for_2d_signals(size: int, lag: int, dim: int, signal_dim: int) -> None:
    # given
    signal = np.random.random(size=(size, signal_dim))
    embedding = LagEmbedding(lag=lag, dim=dim)

    # when
    embedded_signal = embedding.embedd(signal=signal)

    # then
    expected_shape = (signal.shape[0] - lag * dim + 1, signal_dim, dim)
    assert embedded_signal.shape == expected_shape
    #  -1 ist the last position (and the first dimension), dim-1 lags still miss and mus be added (subtracted)
    expected_last_embedding = signal[lag * (1 - dim) - 1 :: lag, :].T.squeeze()
    actual_last_embedding = embedded_signal[-1, :]
    assert np.array_equal(actual_last_embedding, expected_last_embedding)
    assert np.array_equal(embedded_signal[:, :, -1], signal[lag * dim - 1 :, :])


@pytest.mark.parametrize("size", [100, 101])
@pytest.mark.parametrize("lag", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("signal_dim", [1, 3, 10])
def test_lag_embedding_returns_original_signal_for_dim1(size: int, lag: int, signal_dim: int) -> None:
    # given
    signal = np.random.random(size=(size, signal_dim))
    embedding = LagEmbedding(lag=lag, dim=1)

    # when
    embedded_signal = embedding.embedd(signal=signal)

    # then
    assert np.array_equal(embedded_signal, signal)


@pytest.mark.parametrize("size", [100, 101])
@pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("signal_dim", [1, 3, 10])
def test_lag_embedding_returns_original_signal_for_lag0(size: int, dim: int, signal_dim: int) -> None:
    # given
    signal = np.random.random(size=(size, signal_dim))
    embedding = LagEmbedding(lag=0, dim=dim)

    # when
    embedded_signal = embedding.embedd(signal=signal)

    # then
    assert np.array_equal(embedded_signal, signal)


@pytest.mark.parametrize("lag", [2, 3])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("signal_dim", [1, 3])
def test_lag_embedding_raises_if_signal_too_short(lag: int, dim: int, signal_dim: int) -> None:
    # given
    size = lag * dim - 2
    signal = np.random.random(size=(size, signal_dim))
    embedding = LagEmbedding(lag=lag, dim=dim)

    # when then
    with pytest.raises(RuntimeError, match="The signal length has to be at least"):
        embedding.embedd(signal)


def test_lag_embedding_raises_for_ed_signal() -> None:
    # given
    signal = np.random.random(size=(2, 2, 2))
    embedding = LagEmbedding(lag=1, dim=1)

    # when then
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        embedding.embedd(signal)
