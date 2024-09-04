import enum
from pathlib import Path

import itk
import numpy as np
import numpy.typing as npt

from .decoding import decode_array
from .encoding import encode_array
from .encoding import get_quantization_table
from .transforms import create_ijk_to_ras_from_itk_image
from .transforms import get_itk_metadata_from_ijk_to_ras


class FormatKeys(str, enum.Enum):
    IJK_TO_RAS = "ijk_to_ras"
    QUANTIZATION_BLOCK = "quantization_block"
    DC_RLE_VALUES = "dc_rle_values"
    DC_RLE_COUNTS = "dc_rle_counts"
    AC_RLE_VALUES = "ac_rle_values"
    AC_RLE_COUNTS = "ac_rle_counts"
    DTYPE = "dtype"
    INTERCEPT = "intercept"
    SLOPE = "slope"
    SHAPE = "shape"


def open_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    _open = open_jvol if path.suffix == ".jvol" else open_itk_image
    return _open(path)


def save_image(
    array: np.ndarray,
    ijk_to_ras: np.ndarray,
    path: Path,
    **kwargs: int,
) -> None:
    if path.suffix == ".jvol":
        save_jvol(array, ijk_to_ras, path, **kwargs)
    else:
        save_itk_image(array, ijk_to_ras, path)


def open_itk_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = itk.imread(path)
    array = itk.array_view_from_image(image).T
    ijk_to_ras = create_ijk_to_ras_from_itk_image(image)
    return array, ijk_to_ras


def save_itk_image(array: np.ndarray, ijk_to_ras: np.ndarray, path: Path) -> None:
    image = itk.image_view_from_array(array.T.copy())
    origin, rotation, spacing = get_itk_metadata_from_ijk_to_ras(ijk_to_ras)
    image.SetOrigin(origin)
    image.SetDirection(rotation)
    image.SetSpacing(spacing)
    itk.imwrite(image, path)


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
    dc_rle_values, dc_rle_counts, ac_rle_values, ac_rle_counts = encode_array(
        array,
        quantization_table,
    )

    dc_rle_values = dc_rle_values.astype(np.min_scalar_type(dc_rle_values))
    dc_rle_counts = dc_rle_counts.astype(np.min_scalar_type(dc_rle_counts))
    ac_rle_values = ac_rle_values.astype(np.min_scalar_type(ac_rle_values))
    ac_rle_counts = ac_rle_counts.astype(np.min_scalar_type(ac_rle_counts))

    save_dict = {
        FormatKeys.IJK_TO_RAS.value: ijk_to_ras[:3],
        FormatKeys.QUANTIZATION_BLOCK.value: quantization_table,
        FormatKeys.DC_RLE_VALUES.value: dc_rle_values,
        FormatKeys.DC_RLE_COUNTS.value: dc_rle_counts,
        FormatKeys.AC_RLE_VALUES.value: ac_rle_values,
        FormatKeys.AC_RLE_COUNTS.value: ac_rle_counts,
        FormatKeys.DTYPE.value: np.empty((), dtype=dtype),
        FormatKeys.INTERCEPT.value: intercept,
        FormatKeys.SLOPE.value: slope,
        FormatKeys.SHAPE.value: np.array(array.shape, dtype=np.uint16),
    }

    with open(path, "wb") as f:
        np.savez_compressed(f, **save_dict)


def open_jvol(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = np.load(path)
    ijk_to_ras = fill_ijk_to_ras(loaded[FormatKeys.IJK_TO_RAS.value])
    quantization_block = loaded[FormatKeys.QUANTIZATION_BLOCK.value]
    array = decode_array(
        dc_rle_values=loaded[FormatKeys.DC_RLE_VALUES],
        dc_rle_counts=loaded[FormatKeys.DC_RLE_COUNTS],
        ac_rle_values=loaded[FormatKeys.AC_RLE_VALUES],
        ac_rle_counts=loaded[FormatKeys.AC_RLE_COUNTS],
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
