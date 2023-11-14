from typing import Any
from typing import TypeVar

import numpy as np
from numpy import generic
import numpy.typing as npt
from einops import rearrange
from loguru import logger
from scipy.fft import idctn

from .encoding import get_scan_indices_block


DType = TypeVar("DType", bound=generic)
TypeShapeArray = npt.NDArray[np.uint16]
TypeShapeBlockNumpy = npt.NDArray[np.uint8]
TypeShapeTuple = tuple[int, int, int]
TypeRleValues = npt.NDArray[np.int32]
TypeRleCounts = npt.NDArray[np.uint32]


def decode_array(
    rle_values: npt.NDArray[np.int32],
    rle_counts: npt.NDArray[np.uint32],
    quantization_block: npt.NDArray[np.uint16],
    target_shape: TypeShapeArray,
    intercept: float,
    slope: float,
    dtype: npt.DTypeLike,
) -> npt.NDArray[Any]:
    logger.debug(f"Decoding {len(rle_values)} RLE values...")
    scanned_sequence = np.repeat(rle_values, rle_counts)

    logger.debug(f"Reconstructing blocks from {len(scanned_sequence)} components...")
    block_shape_array = np.array(quantization_block.shape, np.uint8)
    scanning_indices = get_scan_indices_block(block_shape_array)
    dct_blocks = sequence_to_blocks(scanned_sequence, scanning_indices)

    logger.debug(f"Computing inverse cosine transform of {len(dct_blocks)} blocks...")
    array_raw = inverse_cosine_transform(
        dct_blocks,
        quantization_block,
        target_shape,
    )
    logger.debug("Rescaling array...")
    array_rescaled = rescale_array_for_decoding(array_raw, intercept, slope)

    logger.debug("Cropping array if necessary...")
    array_cropped = crop_array(array_rescaled, target_shape)

    iinfo = np.iinfo(dtype)
    if array_cropped.min() < iinfo.min or array_cropped.max() > iinfo.max:
        logger.warning(
            f'Array is outside of bounds of data type "{dtype}"'
            f" ([{iinfo.min}, {iinfo.max}])):"
            f" [{array_cropped.min()}, {array_cropped.max()}]. Clipping..."
        )
        array_cropped = np.clip(array_cropped, iinfo.min, iinfo.max)

    logger.debug(f"Casting array to {repr(dtype)} if needed...")
    array_cast = array_cropped.astype(dtype)

    logger.success("Decoded array")
    return array_cast


def rescale_array_for_decoding(
    array: npt.NDArray[np.float32],
    intercept: float,
    slope: float,
) -> npt.NDArray[np.float32]:
    array_rescaled = array.astype(np.float32)
    array_rescaled += 128
    array_rescaled /= 255
    array_rescaled *= slope
    array_rescaled += intercept
    return array_rescaled


def crop_array(
    array: npt.NDArray[DType],
    target_shape: TypeShapeArray,
) -> npt.NDArray[DType]:
    i, j, k = target_shape
    return array[:i, :j, :k]


def inverse_cosine_transform(
    dct_blocks: npt.NDArray[np.int32],
    quantization_block: npt.NDArray[np.uint16],
    target_shape: TypeShapeArray,
) -> npt.NDArray[np.float32]:
    dct_blocks = dct_blocks.astype(float)
    quantization_block = quantization_block.astype(float)
    dct_blocks_rescaled = dct_blocks * quantization_block
    blocks = idctn(dct_blocks_rescaled, axes=(-3, -2, -1))
    blocks_array = np.array(blocks, dtype=np.float32)
    block_shape = np.array(quantization_block.shape, np.uint8)
    padded_target_shape = pad_image_shape(target_shape, block_shape)
    num_blocks_ijk = (padded_target_shape / block_shape).astype(int)
    assert len(blocks) == np.prod(num_blocks_ijk)
    array_reconstructed = rearrange(
        blocks_array,
        "(i1 j1 k1) i2 j2 k2 -> (i1 i2) (j1 j2) (k1 k2)",
        i1=num_blocks_ijk[0],
        j1=num_blocks_ijk[1],
        k1=num_blocks_ijk[2],
    )
    return array_reconstructed


def pad_image_shape(
    image_shape: TypeShapeArray,
    block_shape: TypeShapeBlockNumpy,
) -> TypeShapeArray:
    padding = block_shape - image_shape % block_shape
    return image_shape + padding


def sequence_to_blocks(
    sequence: npt.NDArray[DType], indices_block: npt.NDArray[np.uint8]
) -> npt.NDArray[DType]:
    num_elements_block = len(indices_block)

    block_size = np.cbrt(num_elements_block)
    assert block_size.is_integer()
    block_size = int(block_size)

    num_blocks = len(sequence) / num_elements_block
    assert num_blocks.is_integer()
    num_blocks = int(num_blocks)

    block_shape = block_size, block_size, block_size
    blocks_shape = num_blocks, *block_shape
    blocks = np.zeros(blocks_shape, dtype=sequence.dtype)

    seq_index = 0
    for index in indices_block:
        values_from_sequence = sequence[seq_index : seq_index + num_blocks]
        blocks[:, index[0], index[1], index[2]] = values_from_sequence
        seq_index += num_blocks

    return blocks
