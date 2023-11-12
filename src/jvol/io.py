import enum
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt

from .decoding import decode_array
from .encoding import encode_array
from .encoding import get_quantization_table


class FormatKeys(str, enum.Enum):
    IJK_TO_RAS = "ijk_to_ras"
    QUANTIZATION_BLOCK = "quantization_block"
    RLE_VALUES = "rle_values"
    RLE_COUNTS = "rle_counts"
    DTYPE = "dtype"
    INTERCEPT = "intercept"
    SLOPE = "slope"
    SHAPE = "shape"


def save_jvol(
    array: np.ndarray,
    ijk_to_ras: np.ndarray,
    path: Path,
    block_size: int = 4,
    quality: int = 60,
) -> None:
    block_shape = block_size, block_size, block_size
    quantization_table = get_quantization_table(block_shape, quality)
    dtype = array.dtype
    intercept = array.min()
    slope = array.max() - intercept
    rle_values, rle_counts = encode_array(array, quantization_table)

    save_dict = {
        FormatKeys.IJK_TO_RAS.value: ijk_to_ras[:3],
        FormatKeys.QUANTIZATION_BLOCK.value: quantization_table,
        FormatKeys.RLE_VALUES.value: rle_values.astype(np.min_scalar_type(rle_values)),
        FormatKeys.RLE_COUNTS.value: rle_counts.astype(np.min_scalar_type(rle_counts)),
        FormatKeys.DTYPE.value: np.empty((), dtype=dtype),
        FormatKeys.INTERCEPT.value: intercept,
        FormatKeys.SLOPE.value: slope,
        FormatKeys.SHAPE.value: np.array(array.shape, dtype=np.uint16),
    }

    with open(path, "wb") as f:
        np.savez_compressed(f, **save_dict)


def open_jvol(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    loaded = np.load(path)
    ijk_to_ras = fill_ijk_to_ras(loaded[FormatKeys.IJK_TO_RAS.value])
    quantization_block = loaded[FormatKeys.QUANTIZATION_BLOCK.value]
    array = decode_array(
        rle_values=loaded[FormatKeys.RLE_VALUES],
        rle_counts=loaded[FormatKeys.RLE_COUNTS],
        quantization_block=quantization_block,
        target_shape=loaded[FormatKeys.SHAPE],
        intercept=loaded[FormatKeys.INTERCEPT],
        slope=loaded[FormatKeys.SLOPE],
        dtype=loaded[FormatKeys.DTYPE].dtype,
    )
    return array, ijk_to_ras


def fill_ijk_to_ras(ijk_to_ras: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    last_row = [0, 0, 0, 1]
    return np.vstack((ijk_to_ras, last_row))
