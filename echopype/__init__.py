from __future__ import absolute_import, division, print_function

from _echopype_version import version as __version__  # noqa

from . import calibrate, clean, commongrid, consolidate, mask, utils
from .convert.api import open_raw, open_raw_multi
from .convert.from_ping_data import PingAccumulator, from_ping_data
from .echodata.api import open_converted
from .echodata.combine import combine_echodata
from .utils.io import init_ep_dir
from .utils.log import verbose

# Turn off verbosity for echopype
verbose(override=True)

init_ep_dir()

__all__ = [
    "calibrate",
    "clean",
    "combine_echodata",
    "commongrid",
    "consolidate",
    "from_ping_data",
    "mask",
    "metrics",
    "open_converted",
    "open_raw",
    "open_raw_multi",
    "PingAccumulator",
    "utils",
    "verbose",
]
