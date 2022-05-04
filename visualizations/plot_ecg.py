from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_ecg(
    signal: np.array,
    r_peaks: Optional[np.array] = None,
    q_locs: Optional[np.array] = None,
    front: Optional[int] = None,
    back: Optional[int] = None,
    legend: bool = True,
    title: Optional[str] = None,
    show: bool = True,
    xmin: Optional[int] = None,
    xmax: Optional[int] = None,
) -> None:
    if len(signal.shape) == 1:
        signal = signal.reshape(-1, 1)

    signal_len, num_leads = signal.shape

    xmin = 0 if xmin is None else xmin
    xmax = signal_len if xmax is None else xmax

    fig, ax = plt.subplots(num_leads, 1, figsize=(30, 2 * num_leads))
    for lead in range(num_leads):
        ax[lead].plot(signal[:, lead], label="ECG")
        if r_peaks is not None:
            ax[lead].scatter(r_peaks, signal[r_peaks, lead], marker="+", color="red", label="R-Peak")
            if front is not None and back is not None:
                for s, peak in enumerate(r_peaks):
                    ax[lead].axvspan(
                        peak - front, peak + back, alpha=0.3, facecolor="grey", label="_" * s + "QRST-Window"
                    )
        if q_locs is not None:
            ax[lead].scatter(q_locs, signal[q_locs, lead], marker="*", color="red", label="Q-loc")

        ax[lead].set_xlim(xmin, xmax)

    if legend:
        ax[0].legend(fontsize="x-large")

    if title is not None:
        plt.suptitle(title, fontsize="xx-large", fontweight="bold")

    if show:
        plt.show()


if __name__ == "__main__":
    from wfdb.processing import gqrs_detect, xqrs_detect

    ecg = np.load("../data/SR1.npy")
    Fs = 500
    ecg = (ecg - ecg.mean(axis=0)) * 10  # /ecg.std(axis=0)
    calc_lead = 0

    q_locs = gqrs_detect(np.concatenate([ecg[:, calc_lead], ecg[:Fs, calc_lead]]), fs=Fs)
    r_peaks = xqrs_detect(ecg[:, calc_lead], Fs, verbose=False)
    rr_min = (r_peaks[1:] - r_peaks[:-1]).min()

    plot_ecg(
        signal=ecg[:, :4],
        r_peaks=None,  # r_peaks[r_peaks <= ecg.shape[0]],
        q_locs=q_locs[q_locs <= ecg.shape[0]],
        front=25,
        back=int(0.8 * rr_min),
        xmin=1000,
        xmax=3000,
    )
