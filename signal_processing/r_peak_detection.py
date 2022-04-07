import logging

import neurokit2 as nk
import numpy as np
from ecgdetectors import Detectors
from scipy.signal import cheby2, filtfilt


def get_r_peaks(signal: np.array, sampling_rate: int, return_bools: bool = False) -> np.array:
    """Wrapper for `neurokit2.ecg_peaks`"""

    r_peaks_bool, r_peaks_idxs = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
    r_peaks_idxs = r_peaks_idxs["ECG_R_Peaks"]
    r_peaks_bool = r_peaks_bool.values.squeeze()

    if return_bools:
        return r_peaks_bool
    else:
        return r_peaks_idxs


class RPeakDetector(Detectors):  # type: ignore
    def __init__(
        self,
        sampling_frequency: int,
        method: str = "neuro",
        min_dist: float = 0.2,
        max_dist: float = 2,
        max_num: int = 20,
        min_num: int = 4,
    ) -> None:
        super().__init__(sampling_frequency=sampling_frequency)
        self.min_dist = int(min_dist * sampling_frequency)
        self.max_dist = int(max_dist * sampling_frequency)
        self.max_num = max_num
        self.min_num = min_num
        self.method = method

        self.removals = 0

        self.detector = {
            "Elgendi et al (Two average)": self.two_average_detector,
            "Matched filter": self.matched_filter_detector,
            "Kalidas & Tamil (Wavelet transform)": self.swt_detector,
            "Engzee": self.engzee_detector,
            "Christov": self.christov_detector,
            "Hamilton": self.hamilton_detector,
            "Pan Tompkins": self.pan_tompkins_detector,
            "WQRS": self.wqrs_detector,
            "neuro": lambda x: get_r_peaks(x, sampling_frequency),
        }

    def detect(self, signal: np.array, correct_peaks: bool = True, threshold: float = 1.95) -> np.array:
        r_peaks = self.detector[self.method](signal.T)
        r_peaks = np.asarray(r_peaks)

        if correct_peaks:
            r_peaks = self.correct(signal, r_peaks)

        if len(r_peaks) < self.min_num:
            self.removals += 1
            logging.info(f"Too few r-peaks detected: {len(r_peaks)}, need at least {self.min_num}")
            return None

        if len(r_peaks) > self.max_num:
            self.removals += 1
            logging.info(f"Too may r-peaks detected: {len(r_peaks)}, can have at most {self.max_num}")
            return None

        rr_dist = r_peaks[1:] - r_peaks[:-1]
        if rr_dist.min() < self.min_dist:
            self.removals += 1
            logging.info("Too short r-peak distance ")
            return None

        if rr_dist.max() > self.max_dist:
            self.removals += 1
            logging.info("Too long distance between r-peaks")
            return None

        if rr_dist.max() / rr_dist.min() > threshold:
            self.removals += 1
            logging.info("Too heterogenous  distances between r-peaks")
            return None

        return r_peaks

    def correct(self, signal: np.array, peaks: np.array) -> np.array:
        # from https://dsp.stackexchange.com/questions/58155/how-to-filter-ecg-and-detect-r-peaks
        num_peak = peaks.shape[0]
        peaks_corrected_list = list()
        for index in range(num_peak):
            i = peaks[index]
            cnt = i
            if cnt - 1 < 0:
                break
            if signal[cnt] < signal[cnt - 1]:
                while signal[cnt] < signal[cnt - 1]:
                    cnt -= 1
                    if cnt < 0:
                        break
            elif signal[cnt] < signal[cnt + 1]:
                while signal[cnt] < signal[cnt + 1]:
                    cnt += 1
                    if cnt < 0 or cnt == max(signal.shape) - 1:
                        break
            peaks_corrected_list.append(cnt)
        peaks_corrected = np.asarray(peaks_corrected_list)
        return peaks_corrected


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    data = loadmat("../matlab_src/12leadecg.mat")["data"].T
    x = loadmat("../matlab_src/Y1.mat")["x"].T

    fs = 1000

    [b, a] = cheby2(3, 20, [1, 100], btype="bandpass", fs=fs)
    data_new = filtfilt(b, a, data, axis=0)

    detector = RPeakDetector(fs)
    # qrs_locs = get_r_peaks(x.squeeze(), fs)
    qrs_locs = detector.detect(data_new[:, 2])

    fig, ax = plt.subplots(12, 2, figsize=(10, 40))

    for i in range(12):
        ax[i, 0].plot(data[:, i])
        ax[i, 1].plot(data_new[:, i])
        ax[i, 1].scatter(qrs_locs, data_new[qrs_locs, i], marker="x", color="red")
        ax[i, 0].set_title(f"lead_{i + 1}")
        ax[i, 1].set_title(f"lead_{i + 1}_filtered")

    plt.show()
