"""compute.py

Module containing various helper functions
for performing computations within echopype.

When CuPy is available the GPU-accelerated array module is used transparently.
"""

from typing import Union

import dask.array
import numpy as np

from .gpu import has_cuda

if has_cuda():
    import cupy as cp


def _log2lin(data: Union[dask.array.Array, np.ndarray]) -> Union[dask.array.Array, np.ndarray]:
    """Perform log to linear transform on data

    Parameters
    ----------
    data : dask.array.Array or np.ndarray
         The data to be transformed

    Returns
    -------
    dask.array.Array or np.ndarray
        The transformed data
    """
    if has_cuda() and isinstance(data, cp.ndarray):
        return cp.power(10, data / 10)
    return 10 ** (data / 10)


def _lin2log(data: Union[dask.array.Array, np.ndarray]) -> Union[dask.array.Array, np.ndarray]:
    """Perform linear to log transform on data

    Parameters
    ----------
    data : dask.array.Array or np.ndarray
         The data to be transformed

    Returns
    -------
    dask.array.Array or np.ndarray
        The transformed data
    """
    if has_cuda() and isinstance(data, cp.ndarray):
        return 10 * cp.log10(data)
    return 10 * np.log10(data)
