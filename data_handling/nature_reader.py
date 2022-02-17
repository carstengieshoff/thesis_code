from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_handling.data_reader import DataPoint, DataReader

DataType = Literal["noisy", "clean"]


class NatureReader(DataReader):
    def __init__(self, path: Union[str, Path], use: DataType = "noisy"):
        super().__init__(path=path)
        self._use = use
        self._info = self._read_info()

    def get_dataset(self) -> List[DataPoint]:
        ds: List[DataPoint] = []

        for ref in tqdm(self._info.keys(), total=len(self._info)):
            ds.append(DataPoint(self.get_data_from_reference(ref), self.get_label_from_reference(ref)))

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

    def _read_info(self) -> Dict[str, Dict[str, Any]]:
        info_df = pd.read_excel(self._path / "Diagnostics.xlsx")
        info_df.set_index("FileName", inplace=True)
        info: Dict[str, Dict[str, Any]] = info_df.T.to_dict()
        return info

    def _check_ref_exists(self, ref: str) -> None:
        if ref not in self._info.keys():
            raise KeyError(f"Got unknown reference: `{ref}`")
