from loguru import logger

from .io import open_jvol
from .io import save_jvol
from .jvol import JpegVolume

__version__ = "0.1.0"

__all__ = [
    "JpegVolume",
    "open_jvol",
    "save_jvol",
]

logger.disable("jvol")
