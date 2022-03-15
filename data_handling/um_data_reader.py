import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from tqdm import tqdm

from data_handling.data_reader import DataPoint, DataReader


class UMDataReader(DataReader):
    def __init__(self, path: Union[str, Path]):
        super().__init__(path=path)

        info_path = os.path.join(self._path, "info", "BSPM_ECVOutcome.csv")
        self._label_info = pd.read_csv(info_path).fillna(0)

        self._label_info = self._label_info.astype(
            dtype={"CRFnumber": str, "CVSuccess": int, "Recurrence46weeks": int, "TypeAF": str}
        )

        self._file_names = os.listdir(os.path.join(self._path, "data"))
        self._file_names = [name for name in self._file_names if name.startswith("ECV")]

    def get_dataset(
        self,
        reduced_only: bool = False,
        size: Optional[int] = None,
        load_cancelled_data: bool = True,
        valid_leads_only: bool = False,
    ) -> List[DataPoint]:
        ds: List[DataPoint] = []

        n = len(self._file_names) if size is None else min(size, len(self._file_names))

        for name in tqdm(self._file_names, total=n):
            label = self.read_label(name)
            if reduced_only and label == 2:
                continue

            x = self.read_single_file(
                file_name=name, load_cancelled_data=load_cancelled_data, valid_leads_only=valid_leads_only
            )
            ds.append(DataPoint(x, label))

            if size and len(ds) >= n:
                break

        return ds

    def get_r_peaks(
        self,
        reduced_only: bool = False,
        size: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        peak_indicators: List[pd.DataFrame] = []
        n = len(self._file_names) if size is None else min(size, len(self._file_names))
        for name in tqdm(self._file_names, total=n):
            label = self.read_label(file_name=name)
            if reduced_only and label == 2:
                continue
            r_peaks = pd.read_csv(os.path.join(self._path, "data", name, "rWaveIndices.csv"))
            peak_indicators.append(r_peaks)

            if size and len(ds) >= n:
                break

        return peak_indicators

    def read_label(self, file_name: str) -> int:
        file_name = file_name.split(" ")[2].split("_")[0]
        file_idx = int(file_name)

        label_info = self._label_info.loc[lambda x: x["CRFnumber"] == str(file_idx)]
        if ((label_info["CVSuccess"] == 1) * (label_info["TypeAF"] == "persAF")).item:
            label = label_info["Recurrence46weeks"].item()
        else:
            label = 2

        return int(label)

    def read_single_file(
        self, file_name: str, load_cancelled_data: bool = True, valid_leads_only: bool = False
    ) -> pd.DataFrame:
        file_path = os.path.join(self._path, "data", file_name)

        if load_cancelled_data:
            x = pd.read_csv(os.path.join(file_path, "cancelledEcgData.csv"))
        else:
            x = pd.read_csv(os.path.join(file_path, "ecgData.csv"))

        if valid_leads_only:
            try:
                invalid_electrodes = pd.read_csv(os.path.join(file_path, "invalidelectrodes.txt"), header=None)
                invalid_electrodes = invalid_electrodes[0].values.tolist()
                x.drop(columns=invalid_electrodes, inplace=True)
            except FileNotFoundError:
                pass
        return x


if __name__ == "__main__":

    path = "../matlab_src/"
    reader = UMDataReader(path=path)

    ds = reader.get_dataset(reduced_only=False, load_cancelled_data=True, valid_leads_only=True, size=1)

    peaks = reader.get_r_peaks(reduced_only=False, size=1)

    print("Done!")
