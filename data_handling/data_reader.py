from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import List, Union

DataPoint = namedtuple("DataPoint", "x y")


class DataReader(ABC):
    """Abstract class for reading data from different sources to the same format

    Args:
        path: Location of data to read.
    """

    def __init__(self, path: Union[str, Path]):
        self._path = Path(path)

    @abstractmethod
    def get_dataset(self) -> List[DataPoint]:
        pass
