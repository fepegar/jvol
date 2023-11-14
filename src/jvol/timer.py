import datetime
import time
from contextlib import ContextDecorator

import humanize
from loguru import logger


# Adapted from https://stackoverflow.com/a/70059951/3956024
class timed(ContextDecorator):
    def __init__(self, minimum_unit: str = "milliseconds"):
        self._label = None
        self._minimum_unit = minimum_unit

    def __call__(self, func):
        self._label = func.__name__
        return super().__call__(func)

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        seconds = time.perf_counter() - self.start_time
        delta = datetime.timedelta(seconds=seconds)
        delta_string = humanize.precisedelta(delta, minimum_unit=self._minimum_unit)
        logger.debug(f"{self._label} took {delta_string}")
        return False
