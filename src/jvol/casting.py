from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .types import DType


def to_min_scalar_type(data: Sequence[Any] | npt.NDArray[DType]) -> npt.NDArray[DType]:
    dtype = np.min_scalar_type(data)
    return np.array(data, dtype=dtype)
