import neurokit2 as nk
import numpy as np


def get_r_peaks(signal: np.array, sampling_rate: int, return_bools: bool = False) -> np.array:
    """Wrapper for `neurokit2.ecg_peaks`"""

    r_peaks_bool, r_peaks_idxs = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
    r_peaks_idxs = r_peaks_idxs["ECG_R_Peaks"]
    r_peaks_bool = r_peaks_bool.values.squeeze()

    if return_bools:
        return r_peaks_bool
    else:
        return r_peaks_idxs
