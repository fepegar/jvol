from typing import Any

import numpy as np
import numpy.typing as npt
from einops import rearrange
from loguru import logger
from scipy.fft import idctn

from .encoding import get_scan_indices_block
from .huffman import HuffmanCoding
from .timer import timed
from .types import DType
from .types import TypeRleCounts
from .types import TypeRleValues
from .types import TypeShapeArray
from .types import TypeShapeBlockNumpy
from .types import block_shape_dtype


@timed()  # pyright: ignore
def decode_array(
    dc_rle_values: npt.NDArray[np.int32],
    dc_rle_counts: npt.NDArray[np.uint32],
    # ac_rle_values: npt.NDArray[np.int32],
    # ac_rle_counts: npt.NDArray[np.uint32],
    quantization_block: npt.NDArray[np.uint16],
    target_shape: TypeShapeArray,
    intercept: float,
    slope: float,
    dtype: npt.DTypeLike,
    huffman_coding: HuffmanCoding,
) -> npt.NDArray[Any]:
    logger.info(f"Decoding {len(dc_rle_values):,} DC RLE values...")
    dc_scanned_sequence = run_length_decode(dc_rle_values, dc_rle_counts)
    # logger.info(f"Decoding {len(ac_rle_values):,} AC RLE values...")
    ac_rle_values, ac_rle_counts = huffman_coding.decode()
    ac_scanned_sequence = run_length_decode(ac_rle_values, ac_rle_counts)

    logger.info(
        f"Reconstructing blocks from {len(dc_scanned_sequence):,} DC components"
        f" and {len(ac_scanned_sequence):,} AC components..."
    )
    block_shape_array = np.array(quantization_block.shape, block_shape_dtype)
    scanning_indices = get_scan_indices_block(block_shape_array)

    dct_blocks = sequence_to_blocks(
        dc_scanned_sequence,
        ac_scanned_sequence,
        scanning_indices,
    )

    logger.info(f"Computing inverse cosine transform of {len(dct_blocks):,} blocks...")
    array_raw = inverse_cosine_transform(
        dct_blocks,
        quantization_block,
        target_shape,
    )
    logger.info("Rescaling array intensities...")
    array_rescaled = rescale_array_for_decoding(array_raw, intercept, slope)

    array_cropped = crop_array(array_rescaled, target_shape)
    array_cropped = clip_to_dtype(array_cropped, dtype)

    if array_cropped.dtype != dtype:
        logger.info(f'Casting array to data type "{repr(dtype)}"...')
        array_cropped = array_cropped.astype(dtype)

    logger.success("Array decoded successfully")
    return array_cropped


def clip_to_dtype(
    array: npt.NDArray[DType], dtype: npt.DTypeLike
) -> npt.NDArray[DType]:
    try:
        iinfo = np.iinfo(dtype)
        if array.min() < iinfo.min or array.max() > iinfo.max:
            logger.warning(
                f"Array intensities [{array.min():.1f}, {array.max():.1f}]"
                f' are outside of bounds of data type "{dtype}"'
                f" [{iinfo.min}, {iinfo.max}] after inverse cosine transform."
                f" Clipping..."
            )
            array = np.clip(array, iinfo.min, iinfo.max)
    except ValueError:
        pass
    return array


@timed()  # pyright: ignore
def run_length_decode(
    values: TypeRleValues,
    counts: TypeRleCounts,
) -> npt.NDArray[np.int32]:
    return np.repeat(values, counts)


@timed()  # pyright: ignore
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
    if tuple(target_shape) == array.shape:
        return array
    logger.info(f"Cropping array from {array.shape} to {target_shape}...")
    i, j, k = target_shape
    return array[:i, :j, :k]


@timed()  # pyright: ignore
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
    block_shape = np.array(quantization_block.shape, block_shape_dtype)
    padded_target_shape = pad_image_shape(target_shape, block_shape)
    num_blocks_ijk = (padded_target_shape / block_shape).astype(int)
    if len(blocks) != np.prod(num_blocks_ijk):
        raise RuntimeError(
            f"Number of blocks ({len(blocks)}) does not match"
            f" number of blocks ({num_blocks_ijk}) implied by target shape"
            f" ({num_blocks_ijk}, i.e., {np.prod(num_blocks_ijk)} blocks)"
        )
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
    padding[padding == block_shape] = 0
    return image_shape + padding


@timed()  # pyright: ignore
def sequence_to_blocks(
    dc_sequence: npt.NDArray[DType],
    ac_sequence: npt.NDArray[DType],
    indices_block: npt.NDArray[np.uint8],
) -> npt.NDArray[DType]:
    num_elements_block = len(indices_block)

    block_size = np.cbrt(num_elements_block)
    assert block_size.is_integer()
    block_size = int(block_size)

    num_blocks = len(dc_sequence)

    block_shape = block_size, block_size, block_size
    blocks_shape = num_blocks, *block_shape
    blocks = np.zeros(blocks_shape, dtype=dc_sequence.dtype)

    blocks[:, 0, 0, 0] = dc_sequence

    seq_index = 0
    for index in indices_block[1:]:
        values_from_sequence = ac_sequence[seq_index : seq_index + num_blocks]
        blocks[:, index[0], index[1], index[2]] = values_from_sequence
        seq_index += num_blocks

    return blocks
