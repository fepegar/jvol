from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from typing import Optional
from typing import TypeAlias
from typing import Union

import numpy as np
import numpy.typing as npt

from .io import open_jvol
from .io import save_jvol

TypePath: TypeAlias = Union[str, os.PathLike]


class JpegVolume:
    """Base class for saving and loading JPEG-encoded volumes.

    Args:
        array: 3D NumPy array.
        ijk_to_ras: 4×4 affine transformation matrix containing the mapping
            from voxel indices to RAS+ (left → right, posterior → anterior,
            inferior → superior) coordinates. If not specified, the identity
            matrix is used.

    Tip:
        To learn more about coordinates systems, check the following resources:

            * [Neuroimaging in Python](https://nipy.org/nibabel/coordinate_systems.html)
            * [Slicer Coordinate Systems](https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html)
            * [FSL (see "Background information on NIfTI Orientation")](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Orientation%20Explained)
    """  # noqa: E501

    def __init__(
        self,
        array: npt.ArrayLike,
        ijk_to_ras: Optional[npt.ArrayLike] = None,
    ):
        self.array = np.array(array)
        if ijk_to_ras is None:
            ijk_to_ras = np.eye(4)
        self.ijk_to_ras = np.array(ijk_to_ras, dtype=np.float64)
        if self.array.ndim != 3:
            raise ValueError(
                f"Array must have 3 dimensions, got shape {self.array.shape}"
            )
        if self.ijk_to_ras.shape != (4, 4):
            raise ValueError(
                f"ijk_to_ras must have shape (4, 4), got {self.ijk_to_ras.shape}"
            )
        assert self.ijk_to_ras.shape == (4, 4)

    @classmethod
    def open(cls, path: TypePath) -> JpegVolume:
        """Open a JVol file.

        Args:
            path: Path to a file with `'.jvol'` extension.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'File not found: "{path}"')
        if path.suffix != ".jvol":
            raise ValueError(f'File must have .jvol extension, got "{path}"')

        return cls(*open_jvol(path))

    def save(self, path: TypePath, block_size: int = 8, quality: int = 60) -> None:
        """Save the image using a lossy encoding algorithm for compression.

        Args:
            path: Output path with `.jvol` extension.
            block_size: Size of the blocks to use for encoding. Higher
                values result in better quality, but larger file sizes.
            quality: Quality of the JPEG encoding, between 1 and 100. Higher
                values result in better quality, but larger file sizes.

        Raises:
            ValueError: If the quality is not an integer between 1 and 100.
            ValueError: If the block size is not a positive integer.
        """
        if quality != int(quality):
            raise ValueError(f"Quality must be an integer, got {quality}")
        if not 1 <= quality <= 100:
            raise ValueError(f"Quality must be between 1 and 100, got {quality}")

        if block_size != int(block_size):
            raise ValueError(f"Block size must be an integer, got {block_size}")
        if block_size <= 0:
            raise ValueError(f"Block size must be positive, got {block_size}")

        path = Path(path)

        save_jvol(
            self.array,
            self.ijk_to_ras,
            path,
            block_size=block_size,
            quality=quality,
        )

    def __getattribute__(self, name: str) -> Any:
        # Get attribute from the JpegVolume or the underlying array
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            try:
                return getattr(self.array, name)
            except AttributeError:
                raise e
