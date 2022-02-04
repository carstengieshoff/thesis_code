import csv
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from scipy.io import loadmat

from data_handling.data_reader import DataPoint, DataReader


class PhysioNetReader(DataReader):
    def __init__(self, path: Union[str, Path]):
        super().__init__(path=path)
        self._references = self._read_references()

    def get_dataset(self) -> List[DataPoint]:
        return [DataPoint(self.get_data_from_reference(ref), label) for ref, label in self._references.items()]

    def get_data_from_reference(self, ref: str) -> np.ndarray:
        self._check_ref_exists(ref=ref)

        data = loadmat(self._path / f"{ref}.mat")
        data = data["val"]
        return data

    def get_label_from_reference(self, ref: str) -> "str":
        self._check_ref_exists(ref=ref)
        return self._references[ref]

    def _read_references(self) -> Dict[str, str]:
        with open(self._path / "REFERENCE.csv", "r") as csvfile:
            reference_reader = csv.reader(csvfile)
            references = {ref[0]: ref[1] for ref in reference_reader}
        return references

    def _check_ref_exists(self, ref: str) -> None:
        if ref not in self._references.keys():
            raise KeyError(f"Got unknown reference: `{ref}`")
