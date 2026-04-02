from typing import TYPE_CHECKING, Dict, Optional

import logging

import xarray as xr

if TYPE_CHECKING:
    from ..core import PathHint

from .echodata import EchoData

logger = logging.getLogger(__name__)


def open_converted(
    converted_raw_path: "PathHint",
    storage_options: Dict[str, str] = None,
    **kwargs
    # kwargs: Dict[str, Any] = {'chunks': 'auto'} # TODO: do we need this?
):
    """Create an EchoData object from a single converted netcdf or zarr file.

    Parameters
    ----------
    converted_raw_path : str
        path to converted data file
    storage_options : dict
        options for cloud storage
    kwargs : dict
        optional keyword arguments to be passed
        into xr.open_dataset

    Returns
    -------
    EchoData object
    """
    # TODO: combine multiple files when opening
    return EchoData.from_file(
        converted_raw_path=converted_raw_path,
        storage_options=storage_options,
        open_kwargs=kwargs,
    )


def append_to_zarr(
    echodata: EchoData,
    zarr_path: str,
    storage_options: Optional[Dict[str, str]] = None,
) -> None:
    """Append EchoData to an existing Zarr store along ``ping_time``.

    Each SONAR-netCDF4 group in the EchoData's DataTree is appended
    independently. Groups with a ``ping_time`` dimension are appended
    along that dimension; other groups are skipped (they contain static
    metadata that does not change across pings).

    The target Zarr store must already exist and have matching channel
    dimensions. This function is designed for incremental writes from
    real-time data acquisition (e.g. ``PingAccumulator`` buffers).

    Parameters
    ----------
    echodata : EchoData
        New data to append.
    zarr_path : str
        Path to the existing Zarr store (local or cloud).
    storage_options : dict, optional
        fsspec storage options for cloud stores.

    Raises
    ------
    FileNotFoundError
        If ``zarr_path`` does not exist.
    """
    import zarr as zarr_mod
    from pathlib import Path

    # Validate target exists
    if storage_options:
        import fsspec
        fs, _ = fsspec.core.url_to_fs(zarr_path, **storage_options)
        if not fs.exists(zarr_path):
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")
    else:
        if not Path(zarr_path).exists():
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    tree = echodata._tree
    if tree is None:
        raise ValueError("EchoData has no DataTree to append.")

    # Groups with ping_time that should be appended
    appendable_groups = {"Sonar/Beam_group1", "Sonar/Beam_group2", "Platform", "Platform/NMEA"}

    for group_path in tree.groups:
        node = tree[group_path] if group_path != "/" else tree
        ds = node.to_dataset(inherit=False)
        if len(ds.variables) == 0:
            continue

        # Only append groups that have ping_time and are in the appendable set
        # Strip leading "/" for matching
        clean_path = group_path.lstrip("/")

        if "ping_time" in ds.dims and clean_path in appendable_groups:
            # Clear encodings to avoid conflicts
            for var in ds.data_vars:
                ds[var].encoding.clear()
            for coord in ds.coords:
                ds[coord].encoding.clear()

            target = f"{zarr_path}/{clean_path}" if clean_path else zarr_path
            logger.info("Appending %d pings to %s", ds.sizes.get("ping_time", 0), target)

            ds.to_zarr(
                target,
                mode="a",
                append_dim="ping_time",
            )
        elif "time1" in ds.dims and clean_path in appendable_groups:
            # Platform NMEA data uses time1
            for var in ds.data_vars:
                ds[var].encoding.clear()
            for coord in ds.coords:
                ds[coord].encoding.clear()

            target = f"{zarr_path}/{clean_path}" if clean_path else zarr_path
            logger.info("Appending %d time1 records to %s", ds.sizes.get("time1", 0), target)

            ds.to_zarr(
                target,
                mode="a",
                append_dim="time1",
            )
