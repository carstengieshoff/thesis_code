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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    data = loadmat("../matlab_src/12leadecg.mat")["data"].T
    x = loadmat("../matlab_src/Y1.mat")["x"].T

    fs = 1000

    qrs_locs = get_r_peaks(x.squeeze(), fs)

    fig, ax = plt.subplots(12, 2, figsize=(10, 40))

    for i in range(12):
        ax[i, 0].plot(data[:, i])
        ax[i, 1].plot(data[:, i])
        ax[i, 1].scatter(qrs_locs, data[qrs_locs, i], marker="x", color="red")
        ax[i, 0].set_title(f"lead_{i + 1}")
        ax[i, 1].set_title(f"lead_{i + 1}_filtered")

    plt.show()
