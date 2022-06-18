from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import List, Union

import pyrqa

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
        """Return a list of `Datapoints`.

        Return a list of `Datapoints` i.e. of tuples (x,y) where x a `np.array` is of shape (n_samples, n_dims) and y
        is a str.
        """
        pass
