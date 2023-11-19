# Frequently asked questions

## Why JVol?

> Lossy compression is typically used when a file can afford to lose some data, and/or if storage space needs to be drastically "freed up".

- [Adobe.com](https://www.adobe.com/uk/creativecloud/photography/discover/lossy-vs-lossless.html)

## How do I get started?

The best way to get started is to read the [Getting started](getting-started.ipynb) guide.

## How do I get help?

If you have a question, please [open a discussion](https://github.com/fepegar/jvol/discussions).
If you would like to request a feature or report a bug, please [submit an issue](https://github.com/fepegar/jvol/issues/new).
If you are reporting a bug, please make sure to mention what you got and what you expected.

## What frameworks does JVol use under the hood?

JVol uses [NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/) for for numerical computations.
The command line is implemented using [Typer](https://typer.tiangolo.com/), and it uses [ITK](https://itk.org/) to read and write medical images in standard formats.

Other libraries that are used:

- [`humanize`](https://python-humanize.readthedocs.io/) for human-readable messages.
- [`einops`](https://einops.rocks/) for human-readable code.
- [`transforms3d`](https://matthew-brett.github.io/transforms3d/) for human-readable transforms manipulation.
