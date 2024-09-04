from typing import TypeVar

import numpy as np
import numpy.typing as npt
from numpy import generic

DType = TypeVar("DType", bound=generic)
block_shape_dtype = np.uint8
TypeBlockIndices = npt.NDArray[np.uint8]
TypeShapeBlockNumpy = npt.NDArray[block_shape_dtype]
TypeRleValues = npt.NDArray[np.int32]
TypeRleCounts = npt.NDArray[np.uint32]
TypeShapeArray = npt.NDArray[np.uint16]
TypeShapeTuple = tuple[int, int, int]
