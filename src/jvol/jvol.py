from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .io import open_jvol
from .io import save_jvol


class JpegVolume:
    """Base class for saving and loading JPEG-encoded volumes.

    Args:
        array: 3D NumPy array.
        ijk_to_ras: 4x4 affine transformation matrix containing the mapping
            from voxel indices to RAS (left → right, posterior → anterior,
            inferior → superior) coordinates.
    """

    def __init__(self, array: npt.ArrayLike, ijk_to_ras: npt.ArrayLike):
        self.array = np.array(array)
        self.ijk_to_ras = np.array(ijk_to_ras, dtype=np.float64)
        assert self.array.ndim == 3
        assert self.ijk_to_ras.shape == (4, 4)

    @classmethod
    def open(cls, path: Path) -> JpegVolume:
        """Open a JpegVolume from a file.

        Args:
            path: Path to the file to open, typically with a `'.jvol'` extension.
        """
        return cls(*open_jvol(path))

    def save(self, path: Path, block_size: int = 8, quality: int = 60) -> None:
        """Save the JpegVolume to a file.

        Args:
            path: Path to save the file to, typicall with a `'.jvol'` extension.
            block_size: Size of the blocks to use for encoding.
            quality: Quality of the JPEG encoding, between 1 and 100.
        """
        if not 1 <= quality <= 100:
            raise ValueError(f"Quality must be between 1 and 100, got {quality}.")
        save_jvol(
            self.array,
            self.ijk_to_ras,
            path,
            block_size=block_size,
            quality=quality,
        )

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            try:
                return getattr(self.array, name)
            except AttributeError:
                raise e
