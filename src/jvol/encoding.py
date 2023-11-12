from typing import TypeVar

import numpy as np
from numpy import generic
import numpy.typing as npt
from einops import rearrange
from scipy.fft import dctn


DType = TypeVar("DType", bound=generic)
TypeBlockIndices = npt.NDArray[np.uint8]
TypeShapeBlockNumpy = npt.NDArray[np.uint8]
TypeRleValues = npt.NDArray[np.int32]
TypeRleCounts = npt.NDArray[np.uint32]


def encode_array(
    array: npt.NDArray[DType],
    quantization_table: npt.NDArray[np.float32],
) -> tuple[TypeRleValues, TypeRleCounts]:
    block_shape_tuple = quantization_table.shape
    block_shape_array = np.array(block_shape_tuple, np.uint8)
    padded_array = pad_array(array, block_shape_array)
    blocks = split_into_blocks(padded_array, block_shape_array)
    dct_blocks = cosine_transform(blocks)
    dct_blocks_quantized = quantize(dct_blocks, quantization_table)
    scan_indices = get_scan_indices_block(block_shape_array)
    sequence = blocks_to_sequence(dct_blocks_quantized, scan_indices)
    values, counts = run_length_encode(sequence)
    return values, counts


def get_distance_to_origin(indices: TypeBlockIndices) -> npt.NDArray[np.float64]:
    return np.linalg.norm(indices, axis=1)


def get_scan_indices_block(block_shape: TypeShapeBlockNumpy) -> TypeBlockIndices:
    block_shape_tuple = tuple(int(b) for b in block_shape)
    indices_list = list(np.ndindex(block_shape_tuple))
    indices_array = np.array(indices_list)
    distances_to_origin = get_distance_to_origin(indices_array)
    scan_indices = indices_array[np.argsort(distances_to_origin)]
    return scan_indices.astype(np.uint8)


def split_into_blocks(
    array: npt.NDArray[DType],
    block_shape: TypeShapeBlockNumpy,
) -> npt.NDArray[DType]:
    # This assumes the array has been padded so the shape is divisible by block size
    blocks = rearrange(
        array,
        "(i1 i2) (j1 j2) (k1 k2) -> (i1 j1 k1) i2 j2 k2",
        i2=block_shape[0],
        j2=block_shape[1],
        k2=block_shape[2],
    )
    return blocks


def cosine_transform(blocks: npt.ArrayLike) -> npt.NDArray[np.float64]:
    blocks_cast = np.array(blocks, np.float64).copy()
    blocks_cast -= blocks_cast.min()
    blocks_cast /= blocks_cast.max()
    blocks_cast *= 255
    blocks_cast -= 128
    dct_blocks = dctn(blocks_cast, axes=(-3, -2, -1))
    return np.array(dct_blocks)


def quantize(
    dct_blocks: npt.NDArray[np.float64],
    quantization_table: npt.NDArray[np.float32],
) -> npt.NDArray[np.int32]:
    dct_blocks_quantized = np.round(dct_blocks / quantization_table)
    return dct_blocks_quantized.astype(np.int32)


def blocks_to_sequence(
    dct_blocks: npt.NDArray[np.int32],
    scan_indices: npt.NDArray[np.uint8],
) -> npt.NDArray[np.int32]:
    sequence = []
    for index in scan_indices:
        sequence.extend(dct_blocks[:, index[0], index[1], index[2]])
    sequence = np.array(sequence)
    assert len(sequence) == dct_blocks.size
    return sequence


def run_length_encode(
    sequence: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.uint32]]:
    # Adapted from https://stackoverflow.com/a/32681075/3956024
    sequence = np.asarray(sequence)  # force numpy
    num_symbols = len(sequence)
    y = sequence[1:] != sequence[:-1]  # pairwise unequal (string safe)
    indices = np.append(
        np.where(y), num_symbols - 1
    )  # must include last element position
    counts = np.diff(np.append(-1, indices))  # run lengths
    # p = np.cumsum(np.append(0, z))[:-1]  # positions
    values = sequence[indices]
    counts = np.array(counts, dtype=np.uint32)
    return values, counts


def pad_array(
    array: npt.NDArray[DType], block_shape: TypeShapeBlockNumpy
) -> npt.NDArray[DType]:
    padding = block_shape - np.array(array.shape) % block_shape
    padding = [(0, int(p)) for p in padding]
    padded = np.pad(array, padding, mode="reflect")
    return padded


def get_multiplier(
    quality: int,
    max_multiplier: float = 100,
    spread: float = 15,
) -> float:
    qualities = np.arange(1, 101)
    # Exponential decay
    multipliers = 1 + (max_multiplier - 1) * np.exp(-(qualities - 1) / spread)
    # Rescale to [1, max_multiplier]
    multipliers -= multipliers.min()
    multipliers /= multipliers.max()
    multipliers *= max_multiplier - 1
    multipliers += 1
    return multipliers[quality - 1]


def get_quantization_table(shape, quality: int) -> npt.NDArray[np.float32]:
    indices = np.array(list(np.ndindex(*shape)))
    norms = np.linalg.norm(indices, axis=1)
    norms_block = norms.reshape(shape)
    qtm_unit = norms_block / norms_block.max()
    qtm_min_50, qtm_max_50 = 10, 120  # min and max for quality 50 in 2D JPEG
    qtm_range_50 = qtm_max_50 - qtm_min_50
    qtm_50 = qtm_unit * qtm_range_50
    qtm_50 += qtm_min_50
    qtm = 2 * qtm_50
    return qtm * get_multiplier(quality)
