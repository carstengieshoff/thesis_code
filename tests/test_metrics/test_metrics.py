from typing import Callable

import numpy as np
import pytest

from metrics.cosine_distance import cosine_dist
from metrics.minkowski_distance import minkowski_dist


@pytest.mark.parametrize("metric", [minkowski_dist, cosine_dist])
@pytest.mark.parametrize("len_signal", [0, 1, 100, 101])
@pytest.mark.parametrize("dim_signal", [0, 1, 2, 10, 11])
def test_metric_output_shape_with_broadcasting(
    len_signal: int,
    dim_signal: int,
    metric: Callable[[np.array, np.array], np.array],
) -> None:
    # given
    x = np.random.rand(dim_signal)
    y = np.random.rand(len_signal, dim_signal)

    # when
    dist = metric(x, y)

    # then
    assert dist.shape == (len_signal,)


def test_minkowski_dist_on_dummy_data() -> None:
    # given
    N = 10
    x = np.eye(N)

    for n in range(N):
        # when
        dist = minkowski_dist(x[n, :], x, ord=2)

        # then
        assert np.array_equal(dist, np.sqrt(2) * (1 - x[n, :]))


def test_cosine_dist_on_dummy_data() -> None:
    # given
    N = 10
    x = np.eye(N)

    for n in range(N):
        # when
        dist = cosine_dist(x[n, :], x)

        # then
        assert np.array_equal(dist, 1 - x[n, :])
