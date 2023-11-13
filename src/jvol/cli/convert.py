from pathlib import Path
from typing import Tuple

import itk
import numpy as np
import typer
from loguru import logger

from jvol import open_jvol
from jvol import save_jvol


FLIPXY = np.diag([-1, -1, 1])

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
) -> None:
    logger.info(f'Opening "{input_path}"...')
    array, ijk_to_ras = open_image(input_path)
    logger.success(f'Opened "{input_path}"')

    logger.info(f'Saving "{output_path}"...')
    save_image(array, ijk_to_ras, output_path, quality=quality, block_size=block_size)
    logger.success(f'Saved "{output_path}"')


def open_image(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".jvol":
        return open_jvol(path)
    else:
        return open_itk_image(path)


def open_itk_image(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    image = itk.imread(path)
    array = itk.array_view_from_image(image).T
    import transforms3d

    kji_to_spl = transforms3d.affines.compose(  # is this a good name?
        image["origin"],
        image["direction"],
        image["spacing"],
    )
    result = np.eye(4)
    result[:3, :3] = _rotation_from_itk_convention(kji_to_spl[:3, :3])
    result[:3, 3] = kji_to_spl[:3, 3][::-1] * np.array((-1, -1, 1))
    return array, result


def save_itk_image(array: np.ndarray, ijk_to_ras: np.ndarray, path: Path) -> None:
    import transforms3d

    kji_to_spl = np.eye(4)
    kji_to_spl[:3, 3] = ijk_to_ras[:3, 3][::-1] * np.array((-1, -1, 1))
    kji_to_spl[:3, :3] = _rotation_to_itk_convention(ijk_to_ras[:3, :3])
    origin, direction, spacing, _ = transforms3d.affines.decompose(kji_to_spl)
    image = itk.image_from_array(array.T)
    image["origin"] = origin
    image["direction"] = direction
    image["spacing"] = spacing
    itk.imwrite(image, path)


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


def _rotation_to_itk_convention(matrix):
    matrix = np.dot(FLIPXY, matrix)
    matrix = np.dot(matrix, FLIPXY)
    matrix = np.linalg.inv(matrix)
    return matrix


def _rotation_from_itk_convention(matrix):
    matrix = np.dot(matrix, FLIPXY)
    matrix = np.dot(FLIPXY, matrix)
    matrix = np.linalg.inv(matrix)
    return matrix
