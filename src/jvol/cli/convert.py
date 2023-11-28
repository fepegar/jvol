from __future__ import annotations  # for mkdocstrings

import sys
from pathlib import Path

import typer
from humanize import naturalsize
from loguru import logger
from typing_extensions import Annotated

from jvol.io import open_image
from jvol.io import save_image

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
        help="Quality of the JPEG encoding, between 1 and 100.",
    ),
    block_size: int = typer.Option(
        8,
        "--block-size",
        "-b",
        min=2,
        help=(
            "Size of the blocks to use for encoding."
            " Quality is higher with larger blocks, but so is the file size."
        ),
    ),
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Enable verbose logging. Use -vv for debug logging.",
        ),
    ] = 0,
) -> None:
    """Tool for converting medical images to and from JPEG-encoded volumes."""
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
