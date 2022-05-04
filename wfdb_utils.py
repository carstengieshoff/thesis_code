import logging
from copy import deepcopy
from typing import List, Optional

import numpy as np
import wfdb
from scipy.signal import filtfilt, resample
from wfdb.processing import correct_peaks, gqrs_detect, xqrs_detect

SNOMED_CODE_2_LABEL = {
    "270492004": "1st degree av block",
    "164889003": "atrial fibrillation",
    "164909002": "left bundle branch block",
    "284470004": "premature atrial contraction",
    "59118001": "right bundle branch block",
    "426783006": "sinus rhythm",
    "429622005": "st depression",
    "164931005": "st elevation",
    "164884008": "ventricular ectopics",
}

logging.basicConfig(level=logging.INFO)


def filter_record(record: wfdb.Record, b: np.array, a: np.array) -> wfdb.Record:
    """Filter `record.d_signal` with coefficients `b` and `a`."""
    record.d_signal = filtfilt(b, a, record.d_signal, axis=0)
    return record


def sample_record_to(record: wfdb.Record, fs_target: int = 500) -> wfdb.Record:
    """Resample `record.d_signal` to frequency `fs_target`."""
    if record.fs == fs_target:
        return record

    sample_ratio = fs_target / record.fs
    record.d_signal = resample(record.d_signal, int(record.sig_len * sample_ratio), axis=0)

    record.fs = fs_target
    record.sig_len = record.d_signal.shape[0]
    return record


def shorten_record(record: wfdb.Record, start: int = 0, signal_length: int = 5000) -> Optional[wfdb.Record]:
    if record.sig_len < start + signal_length:
        return record

    record.d_signal = record.d_signal[start : start + signal_length]
    record.sig_len = signal_length

    return record


def divide_record(record: wfdb.Record, signal_length: int = 5000) -> List[wfdb.Record]:

    num_parts = record.sig_len // signal_length

    records: List[wfdb.Record] = []
    for chunk in range(num_parts):
        new_record = shorten_record(record, start=chunk * signal_length, signal_length=signal_length)
        if new_record is None:
            raise RuntimeError("`shorten_record` returned too short signal")
        new_record = deepcopy(new_record)
        new_record.record_name += f"_{chunk}"
        new_record.file_name = [new_record.record_name + ".mat" for _ in new_record.file_name]

        records.append(new_record)

    return records


def get_rpeaks(
    record: wfdb.Record, channel: int = 0, with_correction: bool = False, access_option: str = "infer"
) -> np.array:
    """Wrapping `wfdb.processing.xqrs_detect`."""
    if access_option == "infer":
        r_peaks = xqrs_detect(sig=record.d_signal[:, channel], fs=record.fs, verbose=False, learn=False)
        if with_correction:
            try:
                r_peaks = correct_peaks(
                    record.d_signal[:, 0], r_peaks, search_radius=record.fs // 100, smooth_window_size=3
                )
            except IndexError:
                logging.info(f"Cannot optimize peaks for Record {record.record_name} due to `IndexError`")

    elif access_option == "read":
        r_peaks = _get_comment_line(record, indicator="r_peaks:")
        r_peaks = np.array([int(s) for s in r_peaks.split(",")])
    else:
        raise ValueError(
            f"`get_qrs_locs` expected `access_option` to be one of 'infer', 'read', but got {access_option}"
        )

    return r_peaks


def get_qr_locs(record: wfdb.Record, channel: int = 0, with_correction: bool = False) -> np.array:
    """Wrapping `wfdb.processing.xqrs_detect`."""
    fs = record.fs
    r_peaks = get_rpeaks(record=record, channel=channel, with_correction=with_correction)

    ecg = record.d_signal[:, channel]
    ecg = (ecg - ecg.mean()) / ecg.std()

    # Adding one second at the end, to delay an issue of the alogorith, where the last Q is located poorly
    q_locs = gqrs_detect(np.concatenate([ecg, ecg[:fs]]), fs=fs)
    # Removing Qs in added second
    q_locs = q_locs[q_locs <= len(ecg)]

    if r_peaks[0] - q_locs[0] < 0:
        q_locs = np.insert(q_locs, 0, max(0, r_peaks.min() - int(0.033 * fs)))
        logging.info(f"Adding a q-location for record {record.record_name} at location {q_locs[0]}")
        s = min(len(q_locs), len(r_peaks))
        q_locs = q_locs[:s]
        r_peaks = r_peaks[:s]

    q_locs[r_peaks - q_locs < 0.015 * fs] -= int(0.015 * fs)

    return r_peaks, q_locs


def get_label(record: wfdb.Record) -> int:
    labels = _get_comment_line(record, indicator="Dx:").split(",")
    labels = [SNOMED_CODE_2_LABEL[label] for label in labels if label in SNOMED_CODE_2_LABEL.keys()]

    if "atrial fibrillation" in labels:
        label = 0
    elif "sinus rhythm" in labels:
        label = 1
    else:
        label = 2

    return label


def _get_comment_line(record: wfdb.Record, indicator: str) -> str:
    line: Optional[str] = None
    for s in record.comments:
        if s.startswith(indicator):
            line = s.replace(indicator, "")
            break

    if line is None:
        raise RuntimeError(f"No `{indicator}` found for in comments of record {record.record_name}")
    return line


def write_record(record: wfdb.Record, write_dir: str) -> None:
    """Wrapping 'wfdb.wrsamp' to ensure properties such as 'fs' are written to '.hea' correctly."""
    wfdb.wrsamp(
        record_name=record.record_name,
        fs=record.fs,
        units=record.units,
        sig_name=record.sig_name,
        p_signal=record.p_signal,
        d_signal=record.d_signal,
        fmt=record.fmt,
        adc_gain=record.adc_gain,
        baseline=record.baseline,
        comments=record.comments,
        base_time=record.base_time,
        base_date=record.base_date,
        write_dir=write_dir,
    )


if __name__ == "__main__":
    record = wfdb.rdrecord("./JS00001", physical=False)

    # write_record(record, "wfdb_utils/out_data")

    from scipy.signal import cheby2

    from visualizations import plot_ecg

    [b, a] = cheby2(3, 20, [0.5, 30], btype="bandpass", fs=500)
    record = filter_record(record, b, a)
    channel = 1

    r_peaks, q_locs = get_qr_locs(record, with_correction=True, channel=channel)
    rr_diff = r_peaks[1:] - r_peaks[:-1]
    rr_min = rr_diff.min()

    r_peak_values = record.d_signal[r_peaks, channel]
    if np.abs(r_peak_values)[0] < 0.5 * np.quantile(np.abs(r_peak_values), q=0.5):
        print("R-peak Correction")
        r_peaks = r_peaks[1:]
        q_locs = q_locs[1:]
        rr_diff = r_peaks[1:] - r_peaks[:-1]
        rr_min = rr_diff.min()

    plot_ecg(
        signal=record.d_signal[:, :4],
        r_peaks=r_peaks,  # r_peaks[r_peaks <= ecg.shape[0]],
        q_locs=q_locs,
        front=int(0.05 * record.fs),
        back=int(0.8 * rr_min),
        xmin=None,
        xmax=None,
    )

    print("Done")
