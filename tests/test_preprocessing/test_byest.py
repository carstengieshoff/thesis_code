import numpy as np
from scipy.io import loadmat

from signal_processing.byest import Byest


def test_byest_close_to_MATLAB() -> None:
    # given
    detqrs_data = loadmat("../data/detqrs_data.mat")
    signal = detqrs_data["data_centered"].T
    qrs_locs = detqrs_data["qrs_locs"].astype(int).squeeze() - 1
    fs = 1000
    expected_data = detqrs_data["data_af"]

    # when
    actual_data = Byest(Fs=fs, nbvec=5).reconstruct(Y=signal, r_peaks=qrs_locs)

    # then
    for idx in range(12):
        assert np.allclose(actual_data[idx, :], expected_data[idx, :], atol=1e-10)
