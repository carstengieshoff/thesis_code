import logging
from copy import deepcopy
from typing import List, Optional

import numpy as np
import wfdb
from scipy.signal import filtfilt, resample
from wfdb.processing import correct_peaks, xqrs_detect

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

    sample_ratio = fs_target // record.fs
    record.d_signal = resample(record.d_signal, record.sig_len * sample_ratio, axis=0)

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


def get_qrs_locs(
    record: wfdb.Record, channel: int = 0, with_correction: bool = False, access_option: str = "infer"
) -> np.array:
    """Wrapping `wfdb.processing.xqrs_detect`."""
    if access_option == "infer":
        qrs_locs = xqrs_detect(sig=record.d_signal[:, channel], fs=record.fs, verbose=False, learn=False)
        if with_correction:
            try:
                qrs_locs = correct_peaks(record.d_signal[:, 0], qrs_locs, search_radius=10, smooth_window_size=3)
            except IndexError:
                logging.info(f"Cannot optimize peaks for Record {record.record_name} due to `IndexError`")

    elif access_option == "read":
        qrs_locs = _get_comment_line(record, indicator="r_peaks:")
        qrs_locs = np.array([int(s) for s in qrs_locs.split(",")])
    else:
        raise ValueError(
            f"`get_qrs_locs` expected `access_option` to be one of 'infer', 'read', but got {access_option}"
        )

    return qrs_locs


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
    record = wfdb.rdrecord("./test_data/I0049", physical=False)
    record.fs = 1234
    record.comments.append("jiiiha")
    write_record(record, "wfdb_utils/out_data")

    print("Done")
