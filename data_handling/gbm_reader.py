import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import wfdb
from tqdm import tqdm

from data_handling.data_reader import DataPoint, DataReader

LEADS_DICT = {
    12: ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    6: ["I", "II", "III", "aVR", "aVL", "aVF"],
    4: ["I", "II", "III", "V2"],
    3: ["I", "II", "V2"],
    2: ["I", "II"],
}

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


def load_header(header_file: str) -> str:
    with open(header_file, "r") as f:
        header = f.read()
    return header


def get_labels(header: str) -> List[str]:
    labels = list()
    for line in header.split("\n"):
        if line.startswith("#Dx"):
            try:
                entries = line.split(": ")[1].split(",")
                for entry in entries:
                    labels.append(entry.strip())
            except IndexError:
                pass
    return labels


# Get frequency from header.
def get_frequency(header: str) -> Optional[float]:
    frequency = None
    for i, line in enumerate(header.split("\n")):
        if i == 0:
            try:
                frequency = float(line.split(" ")[2])
            except IndexError:
                pass
        else:
            break
    return frequency


class GBMReader(DataReader):
    def __init__(self, path: Union[str, Path], num_leads: int = 12, resolution: int = 16):
        super().__init__(path=path)

        self._file_names = os.listdir(self._path)
        self._file_names = list(set([name.split(".")[0] for name in self._file_names]))

        print("Getting info:")
        self.info: Dict[str, Any] = dict()
        for name in tqdm(self._file_names, total=len(self._file_names)):
            self.info[name] = self.get_info_from_name(name)
        self.ds_dict: Dict[str, DataPoint] = dict()

        self._resolution = resolution
        if num_leads in LEADS_DICT.keys():
            self._list_of_leads = LEADS_DICT[num_leads]
        else:
            raise KeyError(f"Expected `num_leads` to be one of 2, 3, 4, 6, 12; got {num_leads}")

        self.label_to_int: Dict[str, int] = dict()

    def get_dataset(self, max_per_group: int = 10000, accepted_labels: Optional[List[str]] = None) -> List[DataPoint]:

        num_read_per_label: Dict[str, int] = dict()

        skipped_signals = 0

        print("\n Reading data: ")
        for name in tqdm(self.info.keys(), total=len(self.info.keys())):

            labels = self.info[name]["labels"]
            if labels is None or len(labels) == 0:
                skipped_signals += 1
                continue

            if accepted_labels is not None:
                matched_labels = [label for label in labels if label in accepted_labels]
                if len(matched_labels) == 0:
                    skipped_signals += 1
                    continue
                elif len(matched_labels) > 1:
                    logging.warning(f"Sample {name} got multiple macthing labels: {', '.join(matched_labels)}")

            else:
                matched_labels = labels

            ########
            if len(matched_labels) > 1 and "atrial fibrillation" in matched_labels:
                label = "atrial fibrillation"
            else:
                label = matched_labels[0]

            if label in num_read_per_label.keys():
                num_read_per_label[label] += 1
            else:
                num_read_per_label[label] = 1
                self.label_to_int[label] = len(self.label_to_int)

            if num_read_per_label[label] <= max_per_group:
                label_as_int = self.label_to_int[label]
                data = self.get_data_from_name(name)
                self.ds_dict[name] = DataPoint(data, label_as_int)

            if accepted_labels is not None:
                if len(self.ds_dict.keys()) >= max_per_group * len(accepted_labels):
                    break

        print(f"\n Signals skipped due to wrong labels: {skipped_signals}")
        return list(self.ds_dict.values())

    @property
    def int_to_label(self) -> Dict[int, str]:
        return dict(zip(self.label_to_int.values(), self.label_to_int.keys()))

    def get_info_from_name(self, name: str) -> Optional[Dict[str, Any]]:
        try:
            header = load_header(os.path.join(self._path, f"{name}.hea"))
            labels: List[str] = []

            for label in get_labels(header):
                if label in SNOMED_CODE_2_LABEL.keys():
                    # print("LABEL FOUND")
                    labels.append(SNOMED_CODE_2_LABEL[label])
                # else:
                #  logging.warning(f"Sample {name}: Got unknown label {label} (type {type(label)})")

            # if len(labels) == 0:
            # logging.warning(f"Sample {name}: No known label found")

            freq = get_frequency(header)

            return {"freq": freq, "labels": labels}

        except FileNotFoundError:
            logging.warning(f"File {name} not found")
            return None

    def get_data_from_name(self, name: str) -> np.ndarray:

        record = wfdb.rdrecord(
            os.path.join(self._path, name),
            physical=True,
            channel_names=self._list_of_leads,
            return_res=self._resolution,
        )
        return record.p_signal
