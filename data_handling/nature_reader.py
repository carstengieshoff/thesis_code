from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_handling.data_reader import DataPoint, DataReader

DataType = Literal["noisy", "clean"]


class NatureReader(DataReader):
    def __init__(self, path: Union[str, Path], use: DataType = "noisy", with_sb: bool = False):
        super().__init__(path=path)
        self._use = use
        self._info, self.label_to_int, self.int_to_label = self._read_info()
        self._with_sb = with_sb

    def get_dataset(self) -> List[DataPoint]:
        ds: List[DataPoint] = []

        for ref in tqdm(self._info.keys(), total=len(self._info)):
            data = self.get_data_from_reference(ref)
            if self.get_label_from_reference(ref) == "SB" and not self._with_sb:
                continue
            label = self.label_to_int[self.get_label_from_reference(ref)]
            ds.append(DataPoint(data, label))

        return ds

    def get_data_from_reference(self, ref: str) -> np.ndarray:
        self._check_ref_exists(ref=ref)
        extension = "ECGData"
        if self._use == "clean":
            extension += "Denoised"

        df = pd.read_csv(self._path / extension / f"{ref}.csv")
        data = df.values
        return data

    def get_label_from_reference(self, ref: str) -> "str":
        self._check_ref_exists(ref=ref)
        label: str = self._info[ref]["Rhythm"]
        return label

    def _read_info(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int], Dict[int, str]]:
        info_df = pd.read_excel(self._path / "Diagnostics.xlsx")
        info_df.set_index("FileName", inplace=True)
        info: Dict[str, Dict[str, Any]] = info_df.T.to_dict()

        unique_labels = set(info[ref]["Rhythm"] for ref in info.keys())
        label_to_int = dict(zip(unique_labels, list(range(len(unique_labels)))))
        int_to_label = dict(zip(label_to_int.values(), label_to_int.keys()))
        return info, label_to_int, int_to_label

    def _check_ref_exists(self, ref: str) -> None:
        if ref not in self._info.keys():
            raise KeyError(f"Got unknown reference: `{ref}`")


class NatureReaderRestricted(DataReader):
    def __init__(self, path: Union[str, Path], use: DataType = "noisy"):
        super().__init__(path=path)
        self._use = use
        self._info, self.label_to_int, self.int_to_label = self._read_info()

    def get_dataset(self) -> List[DataPoint]:
        ds: List[DataPoint] = []

        for ref in tqdm(self._info.keys(), total=len(self._info)):
            data = self.get_data_from_reference(ref)
            if self.get_label_from_reference(ref) == "SB":
                continue
            label = self.label_to_int[self.get_label_from_reference(ref)]
            ds.append(DataPoint(data, label))
            if len(ds) > 500:
                break

        return ds

    def get_data_from_reference(self, ref: str) -> np.ndarray:
        self._check_ref_exists(ref=ref)
        extension = "ECGData"
        if self._use == "clean":
            extension += "Denoised"

        df = pd.read_csv(self._path / extension / f"{ref}.csv")
        data = df.values
        return data

    def get_label_from_reference(self, ref: str) -> "str":
        self._check_ref_exists(ref=ref)
        label: str = self._info[ref]["Rhythm"]
        return label

    def _read_info(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int], Dict[int, str]]:
        info_df = pd.read_excel(self._path / "Diagnostics.xlsx")
        info_df.set_index("FileName", inplace=True)
        info: Dict[str, Dict[str, Any]] = info_df.T.to_dict()

        unique_labels = set(info[ref]["Rhythm"] for ref in info.keys())
        label_to_int = dict(zip(unique_labels, list(range(len(unique_labels)))))
        int_to_label = dict(zip(label_to_int.values(), label_to_int.keys()))
        return info, label_to_int, int_to_label

    def _check_ref_exists(self, ref: str) -> None:
        if ref not in self._info.keys():
            raise KeyError(f"Got unknown reference: `{ref}`")
