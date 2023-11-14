import sys
from pathlib import Path
from typing import Tuple

import itk
import numpy as np
import numpy.typing as npt
import transforms3d
import typer
from humanize import naturalsize
from loguru import logger
from typing_extensions import Annotated

from jvol import open_jvol
from jvol import save_jvol


app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
    ),
    quality: int = typer.Option(
        60,
        "--quality",
        "-q",
        min=1,
        max=100,
    ),
    block_size: int = typer.Option(
        8,
        "--block-size",
        "-b",
        min=2,
    ),
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0,
) -> None:
    setup_logging(verbose)

    logger.info(f'Opening "{input_path}"...')
    array, ijk_to_ras = open_image(input_path)
    logger.debug(f"Array shape: {array.shape}")
    logger.debug("ijk_to_ras:")
    for line in str(ijk_to_ras).splitlines():
        logger.debug(f"  {line}")
    logger.success(f'Opened "{input_path}" ({naturalsize(input_path.stat().st_size)})')

    logger.info(f'Saving "{output_path}"...')
    save_image(array, ijk_to_ras, output_path, quality=quality, block_size=block_size)
    logger.success(f'Saved "{output_path}" ({naturalsize(output_path.stat().st_size)})')


def open_image(path: Path) -> Tuple[np.ndarray, np.ndarray]:
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


def open_itk_image(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    image = itk.imread(path)
    array = itk.array_view_from_image(image).T
    ijk_to_ras = _create_ijk_to_ras_from_itk_image(image)
    return array, ijk_to_ras


def save_itk_image(array: np.ndarray, ijk_to_ras: np.ndarray, path: Path) -> None:
    image = itk.image_view_from_array(array.T)
    origin, rotation, spacing = _get_itk_metadata_from_ijk_to_ras(ijk_to_ras)
    image.SetOrigin(origin)
    image.SetDirection(rotation)
    image.SetSpacing(spacing)
    itk.imwrite(image, path)


def _create_ijk_to_ras_from_itk_image(image) -> npt.NDArray[np.float64]:
    ijk_to_lps = transforms3d.affines.compose(
        tuple(image.GetOrigin()),
        np.array(image.GetDirection()).reshape(3, 3),
        tuple(image.GetSpacing()),
    )
    lps_to_ras = np.diag((-1, -1, 1, 1))
    ijk_to_ras = lps_to_ras @ ijk_to_lps
    return ijk_to_ras


def _get_itk_metadata_from_ijk_to_ras(ijk_to_ras: npt.NDArray[np.float64]):
    ras_to_lps = np.diag((-1, -1, 1, 1))
    ijk_to_lps = ras_to_lps @ ijk_to_ras
    origin, rotation, spacing, _ = transforms3d.affines.decompose(ijk_to_lps)
    return origin, rotation, spacing


def setup_logging(verbosity: int):
    logger.remove()
    match verbosity:
        case 0:
            level = "WARNING"
        case 1:
            level = "INFO"
        case 2:
            level = "DEBUG"
        case _:
            level = "TRACE"

    logger.add(sys.stderr, level=level)
    logger.enable("jvol")
