import numpy as np
import pytest

from metrics.minkowski_distance import minkowski_dist


@pytest.mark.parametrize("len_signal", [0, 1, 100, 101])
@pytest.mark.parametrize("dim_signal", [0, 1, 2, 10, 11])
def test_metric_output_shape_with_broadcasting(len_signal: int, dim_signal: int) -> None:
    # given
    x = np.random.rand(dim_signal)
    y = np.random.rand(len_signal, dim_signal)

    # when
    dist = minkowski_dist(x=x, y=y)

    # then
    assert dist.shape == (len_signal,)
