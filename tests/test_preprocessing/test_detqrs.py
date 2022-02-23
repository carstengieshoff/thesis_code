import numpy as np
from scipy.io import loadmat

from signal_processing.detqrs import detqrs3


def test_detqrs3() -> None:
    # given
    detqrs_data = loadmat("../data/detqrs_data.mat")
    signal = detqrs_data["data_centered"].T
    expected_qrs_locs = detqrs_data["qrs_locs"].T
    expected_qrs_locs -= 1
    fs = 1000

    # when
    actual_qrs_locs = detqrs3(signal[:, 0], Fs=fs)

    # then
    assert np.array_equal(actual_qrs_locs, expected_qrs_locs)
