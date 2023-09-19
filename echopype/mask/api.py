import datetime
import operator as op
import pathlib
from typing import List, Optional, Union

import numpy as np
import xarray as xr

from echopype.mask.seabed import (
    _get_seabed_mask_ariza,
    _get_seabed_mask_blackwell,
    _get_seabed_mask_blackwell_mod,
    _get_seabed_mask_deltaSv,
    _get_seabed_mask_experimental,
    _get_seabed_mask_maxSv,
)

from ..utils.io import validate_source_ds_da
from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from .freq_diff import _check_freq_diff_source_Sv, _parse_freq_diff_eq
from .shoal import _weill as shoal_weill

# lookup table with key string operator and value as corresponding Python operator
str2ops = {
    ">": op.gt,
    "<": op.lt,
    "<=": op.le,
    ">=": op.ge,
    "==": op.eq,
}


def _validate_source_ds(source_ds, storage_options_ds):
    """
    Validate the input ``source_ds`` and the associated ``storage_options_mask``.
    """
    # Validate the source_ds type or path (if it is provided)
    source_ds, file_type = validate_source_ds_da(source_ds, storage_options_ds)

    if isinstance(source_ds, str):
        # open up Dataset using source_ds path
        source_ds = xr.open_dataset(source_ds, engine=file_type, chunks={}, **storage_options_ds)

    # Check source_ds coordinates
    if "ping_time" not in source_ds or "range_sample" not in source_ds:
        raise ValueError("'source_ds' must have coordinates 'ping_time' and 'range_sample'!")

    return source_ds


def _validate_and_collect_mask_input(
    mask: Union[
        Union[xr.DataArray, str, pathlib.Path], List[Union[xr.DataArray, str, pathlib.Path]]
    ],
    storage_options_mask: Union[dict, List[dict]],
) -> Union[xr.DataArray, List[xr.DataArray]]:
    """
    Validate that the input ``mask`` and associated ``storage_options_mask`` are correctly
    provided to ``apply_mask``. Additionally, form the mask input that should be used
    in the core routine of ``apply_mask``.

    Parameters
    ----------
    mask: xr.DataArray, str, pathlib.Path, or a list of these datatypes
        The mask(s) to be applied. Can be a single input or list that corresponds to a
        DataArray or a path. If a path is provided this should point to a zarr or netcdf
        file with only one data variable in it.
    storage_options_mask: dict or list of dict, default={}
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``mask``. If ``mask`` is a list, then this input should either
        be a list of dictionaries or a single dictionary with storage options that
        correspond to all elements in ``mask`` that are paths.

    Returns
    -------
    xr.DataArray or list of xr.DataArray
        If the ``mask`` input is a single value, then the corresponding DataArray will be
        returned, else a list of DataArrays corresponding to the input masks will be returned

    Raises
    ------
    ValueError
        If ``mask`` is a single-element and ``storage_options_mask`` is not a single dict
    TypeError
        If ``storage_options_mask`` is not a list of dict or a dict
    """

    if isinstance(mask, list):
        # if storage_options_mask is not a list create a list of
        # length len(mask) with elements storage_options_mask
        if not isinstance(storage_options_mask, list):
            if not isinstance(storage_options_mask, dict):
                raise TypeError("storage_options_mask must be a list of dict or a dict!")

            storage_options_mask = [storage_options_mask] * len(mask)
        else:
            # ensure all element of storage_options_mask are a dict
            if not all([isinstance(elem, dict) for elem in storage_options_mask]):
                raise TypeError("storage_options_mask must be a list of dict or a dict!")

        for mask_ind in range(len(mask)):
            # validate the mask type or path (if it is provided)
            mask_val, file_type = validate_source_ds_da(
                mask[mask_ind], storage_options_mask[mask_ind]
            )

            # replace mask element path with its corresponding DataArray
            if isinstance(mask_val, (str, pathlib.Path)):
                # open up DataArray using mask path
                mask[mask_ind] = xr.open_dataarray(
                    mask_val, engine=file_type, chunks={}, **storage_options_mask[mask_ind]
                )

            # check mask coordinates
            # the coordinate sequence matters, so fix the tuple form
            allowed_dims = [
                ("ping_time", "range_sample"),
                ("channel", "ping_time", "range_sample"),
            ]
            if mask[mask_ind].dims not in allowed_dims:
                raise ValueError("All masks must have dimensions ('ping_time', 'range_sample')!")

    else:
        if not isinstance(storage_options_mask, dict):
            raise ValueError(
                "The provided input storage_options_mask should be a single "
                "dict because mask is a single value!"
            )

        # validate the mask type or path (if it is provided)
        mask, file_type = validate_source_ds_da(mask, storage_options_mask)

        if isinstance(mask, (str, pathlib.Path)):
            # open up DataArray using mask path
            mask = xr.open_dataarray(mask, engine=file_type, chunks={}, **storage_options_mask)

    return mask


def _check_var_name_fill_value(
    source_ds: xr.Dataset, var_name: str, fill_value: Union[int, float, np.ndarray, xr.DataArray]
) -> Union[int, float, np.ndarray, xr.DataArray]:
    """
    Ensures that the inputs ``var_name`` and ``fill_value`` for the function
    ``apply_mask`` were appropriately provided.

    Parameters
    ----------
    source_ds: xr.Dataset
        A Dataset that contains the variable ``var_name``
    var_name: str
        The variable name in ``source_ds`` that the mask should be applied to
    fill_value: int or float or np.ndarray or xr.DataArray
        Specifies the value(s) at false indices

    Returns
    -------
    fill_value: int or float or np.ndarray or xr.DataArray
        fill_value with sanitized dimensions

    Raises
    ------
    TypeError
        If ``var_name`` or ``fill_value`` are not an accepted type
    ValueError
        If the Dataset ``source_ds`` does not contain ``var_name``
    ValueError
        If ``fill_value`` is an array and not the same shape as ``var_name``
    """

    # check the type of var_name
    if not isinstance(var_name, str):
        raise TypeError("The input var_name must be a string!")

    # ensure var_name is in source_ds
    if var_name not in source_ds.variables:
        raise ValueError("The Dataset source_ds does not contain the variable var_name!")

    # check the type of fill_value
    if not isinstance(fill_value, (int, float, np.ndarray, xr.DataArray)):
        raise TypeError(
            "The input fill_value must be of type int or " "float or np.ndarray or xr.DataArray!"
        )

    # make sure that fill_values is the same shape as var_name
    if isinstance(fill_value, (np.ndarray, xr.DataArray)):
        if isinstance(fill_value, xr.DataArray):
            fill_value = fill_value.data.squeeze()  # squeeze out length=1 channel dimension
        elif isinstance(fill_value, np.ndarray):
            fill_value = fill_value.squeeze()  # squeeze out length=1 channel dimension

        source_ds_shape = (
            source_ds[var_name].isel(channel=0).shape
            if "channel" in source_ds[var_name].coords
            else source_ds[var_name].shape
        )

        if fill_value.shape != source_ds_shape:
            raise ValueError(
                f"If fill_value is an array it must be of the same shape as {var_name}!"
            )

    return fill_value


def _variable_prov_attrs(
    masked_da: xr.DataArray, source_mask: Union[xr.DataArray, List[xr.DataArray]]
) -> dict:
    """
    Extract and compose masked Sv provenance attributes from the masked Sv and the
    masks used to generate it.

    Parameters
    ----------
    masked_da: xr.DataArray
        Masked Sv
    source_mask: Union[xr.DataArray, List[xr.DataArray]]
        Individual mask or list of masks used to create the masked Sv

    Returns
    -------
    dict
        Dictionary of provenance attributes (attribute name and value) for the intended variable.
    """
    # Modify core variable attributes
    attrs = {
        "long_name": "Volume backscattering strength, masked (Sv re 1 m-1)",
        "actual_range": [
            round(float(masked_da.min().values), 2),
            round(float(masked_da.max().values), 2),
        ],
    }
    # Add history attribute
    history_attr = f"{datetime.datetime.utcnow()} +00:00. " "Created masked Sv dataarray."  # noqa
    attrs = {**attrs, **{"history": history_attr}}

    # Add attributes from the mask DataArray, if present
    # Handle only a single mask. If not passed to apply_mask as a single DataArray,
    # will use the first mask of the list passed to  apply_mask
    # TODO: Expand it to handle attributes from multiple masks
    if isinstance(source_mask, xr.DataArray) or (
        isinstance(source_mask, list) and isinstance(source_mask[0], xr.DataArray)
    ):
        use_mask = source_mask[0] if isinstance(source_mask, list) else source_mask
        if len(use_mask.attrs) > 0:
            mask_attrs = use_mask.attrs.copy()
            if "history" in mask_attrs:
                # concatenate the history string as new line
                attrs["history"] += f"\n{mask_attrs['history']}"
                mask_attrs.pop("history")
            attrs = {**attrs, **mask_attrs}

    return attrs


@add_processing_level("L3*")
def apply_mask(
    source_ds: Union[xr.Dataset, str, pathlib.Path],
    mask: Union[xr.DataArray, str, pathlib.Path, List[Union[xr.DataArray, str, pathlib.Path]]],
    var_name: str = "Sv",
    fill_value: Union[int, float, np.ndarray, xr.DataArray] = np.nan,
    storage_options_ds: dict = {},
    storage_options_mask: Union[dict, List[dict]] = {},
) -> xr.Dataset:
    """
    Applies the provided mask(s) to the Sv variable ``var_name``
    in the provided Dataset ``source_ds``.

    Parameters
    ----------
    source_ds: xr.Dataset, str, or pathlib.Path
        Points to a Dataset that contains the variable the mask should be applied to
    mask: xr.DataArray, str, pathlib.Path, or a list of these datatypes
        The mask(s) to be applied.
        Can be a single input or list that corresponds to a DataArray or a path.
        Each entry in the list must have dimensions ``('ping_time', 'range_sample')``.
        Multi-channel masks are not currently supported.
        If a path is provided this should point to a zarr or netcdf file with only
        one data variable in it.
        If the input ``mask`` is a list, a logical AND will be used to produce the final
        mask that will be applied to ``var_name``.
    var_name: str, default="Sv"
        The Sv variable name in ``source_ds`` that the mask should be applied to.
        This variable needs to have coordinates ``ping_time`` and ``range_sample``,
        and can optionally also have coordinate ``channel``.
        In the case of a multi-channel Sv data variable, the ``mask`` will be broadcast
        to all channels.
    fill_value: int, float, np.ndarray, or xr.DataArray, default=np.nan
        Value(s) at masked indices.
        If ``fill_value`` is of type ``np.ndarray`` or ``xr.DataArray``,
        it must have the same shape as each entry of ``mask``.
    storage_options_ds: dict, default={}
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``source_ds``
    storage_options_mask: dict or list of dict, default={}
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``mask``. If ``mask`` is a list, then this input should either
        be a list of dictionaries or a single dictionary with storage options that
        correspond to all elements in ``mask`` that are paths.

    Returns
    -------
    xr.Dataset
        A Dataset with the same format of ``source_ds`` with the mask(s) applied to ``var_name``
    """

    # Validate the source_ds
    source_ds = _validate_source_ds(source_ds, storage_options_ds)

    # Validate and form the mask input to be used downstream
    mask = _validate_and_collect_mask_input(mask, storage_options_mask)

    # Check var_name and sanitize fill_value dimensions if an array
    fill_value = _check_var_name_fill_value(source_ds, var_name, fill_value)

    # Obtain final mask to be applied to var_name
    if isinstance(mask, list):
        # perform a logical AND element-wise operation across the masks
        final_mask = np.logical_and.reduce(mask)

        # xr.where has issues with attrs when final_mask is an array, so we make it a DataArray
        final_mask = xr.DataArray(final_mask, coords=mask[0].coords)
    else:
        final_mask = mask

    # Sanity check: final_mask should be of the same shape as source_ds[var_name]
    #               along the ping_time and range_sample dimensions
    def get_ch_shape(da):
        return da.isel(channel=0).shape if "channel" in da.dims else da.shape

    # Below operate on the actual data array to be masked
    source_da = source_ds[var_name]

    source_da_shape = get_ch_shape(source_da)
    final_mask_shape = get_ch_shape(final_mask)

    if final_mask_shape != source_da_shape:
        raise ValueError(
            f"The final constructed mask is not of the same shape as source_ds[{var_name}] "
            "along the ping_time and range_sample dimensions!"
        )

    # final_mask is always an xr.DataArray with at most length=1 channel dimension
    if "channel" in final_mask.dims:
        final_mask = final_mask.isel(channel=0)

    # Make sure fill_value and final_mask are expanded in dimensions
    if "channel" in source_da.dims:
        if isinstance(fill_value, np.ndarray):
            fill_value = np.array([fill_value] * source_da["channel"].size)
        final_mask = np.array([final_mask.data] * source_da["channel"].size)

    # Apply the mask to var_name
    # Somehow keep_attrs=True errors out here, so will attach later
    var_name_masked = xr.where(final_mask, x=source_da, y=fill_value)

    # Obtain a shallow copy of source_ds
    output_ds = source_ds.copy(deep=False)

    # Replace var_name with var_name_masked
    output_ds[var_name] = var_name_masked
    output_ds[var_name] = output_ds[var_name].assign_attrs(source_da.attrs)

    # Add or modify variable and global (dataset) provenance attributes
    output_ds[var_name] = output_ds[var_name].assign_attrs(
        _variable_prov_attrs(output_ds[var_name], mask)
    )

    process_type = "mask"
    prov_dict = echopype_prov_attrs(process_type=process_type)
    prov_dict[f"{process_type}_function"] = "mask.apply_mask"

    output_ds = output_ds.assign_attrs(prov_dict)

    output_ds = insert_input_processing_level(output_ds, input_ds=source_ds)

    return output_ds


def frequency_differencing(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    storage_options: Optional[dict] = {},
    freqABEq: Optional[str] = None,
    chanABEq: Optional[str] = None,
) -> xr.DataArray:
    """
    Create a mask based on the differences of Sv values using a pair of
    frequencies. This method is often referred to as the "frequency-differencing"
    or "dB-differencing" method.

    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a mask for,
        else it specifies the path to a zarr or netcdf file containing
        a Dataset. This input must correspond to a Dataset that has the
        coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
    storage_options: dict, optional
        Any additional parameters for the storage backend, corresponding to the
        path provided for ``source_Sv``
    freqABEq: string, optional
        The frequency differencing criteria.
        Only one of ``freqAB`` and ``chanAB`` should be provided, and not both.
    chanAB: string, optional
        The frequency differencing criteria in terms of channel names where channel names
        in the criteria are enclosed in double quotes. Only one of ``freqAB`` and ``chanAB``
        should be provided, and not both.

    Returns
    -------
    xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.

    Raises
    ------
    ValueError
        If neither ``freqABEq`` or ``chanABEq`` are given
    ValueError
        If both ``freqABEq`` and ``chanABEq`` are given
    TypeError
        If any input is not of the correct type
    ValueError
        If either ``freqABEq`` or ``chanABEq`` are provided and the extracted
        ``freqAB`` or ``chanAB`` does not contain 2 distinct elements
    ValueError
        If ``freqABEq`` contains values that are not contained in ``frequency_nominal``
    ValueError
        If ``chanABEq`` contains values that not contained in ``channel``
    ValueError
        If ``operator`` is not one of the following: ``">", "<", "<=", ">=", "=="``
    ValueError
        If the path provided for ``source_Sv`` is not a valid path
    ValueError
        If ``freqABEq`` or ``chanABEq`` is provided and the Dataset produced by ``source_Sv``
        does not contain the coordinate ``channel`` and variable ``frequency_nominal``

    Notes
    -----
    This function computes the frequency differencing as follows:
    ``Sv_freqA - Sv_freqB operator diff``. Thus, if ``operator = "<"``
    and ``diff = "5"`` the following would be calculated:
    ``Sv_freqA - Sv_freqB < 5``.

    Examples
    --------
    Compute frequency-differencing mask using a mock Dataset and channel selection:

    >>> n = 5 # set the number of ping times and range samples
    ...
    >>> # create mock Sv data
    >>> Sv_da = xr.DataArray(data=np.stack([np.arange(n**2).reshape(n,n), np.identity(n)]),
    ...                      coords={"channel": ['chan1', 'chan2'],
    ...                              "ping_time": np.arange(n), "range_sample":np.arange(n)})
    ...
    >>> # obtain mock frequency_nominal data
    >>> freq_nom = xr.DataArray(data=np.array([1.0, 2.0]),
    ...                         coords={"channel": ['chan1', 'chan2']})
    ...
    >>> # construct mock Sv Dataset
    >>> Sv_ds = xr.Dataset(data_vars={"Sv": Sv_da, "frequency_nominal": freq_nom})
    ...
    >>> # compute frequency-differencing mask using channel names
    >>> echopype.mask.frequency_differencing(source_Sv=mock_Sv_ds, storage_options={},
    ...                                      freqABEq=None, chanABEq = '"chan1" - "chan2">=10.0')
    <xarray.DataArray 'mask' (ping_time: 5, range_sample: 5)>
    array([[False, False, False, False, False],
           [False, False, False, False, False],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True]])
    Coordinates:
      * ping_time     (ping_time) int64 0 1 2 3 4
      * range_sample  (range_sample) int64 0 1 2 3 4
    """

    # check that non-data related inputs were correctly provided
    # _check_freq_diff_non_data_inputs(freqAB, chanAB, operator, diff)
    freqAB, chanAB, operator, diff = _parse_freq_diff_eq(freqABEq, chanABEq)

    # validate the source_Sv type or path (if it is provided)
    source_Sv, file_type = validate_source_ds_da(source_Sv, storage_options)

    if isinstance(source_Sv, str):
        # open up Dataset using source_Sv path
        source_Sv = xr.open_dataset(source_Sv, engine=file_type, chunks={}, **storage_options)

    # check the source_Sv with respect to channel and frequency_nominal
    _check_freq_diff_source_Sv(source_Sv, freqAB, chanAB)

    # determine chanA and chanB
    if freqAB is not None:
        # obtain position of frequency provided in frequency_nominal
        freqA_pos = np.argwhere(source_Sv.frequency_nominal.values == freqAB[0]).flatten()[0]
        freqB_pos = np.argwhere(source_Sv.frequency_nominal.values == freqAB[1]).flatten()[0]

        # get channels corresponding to frequencies provided
        chanA = str(source_Sv.channel.isel(channel=freqA_pos).values)
        chanB = str(source_Sv.channel.isel(channel=freqB_pos).values)

    else:
        # get individual channels
        chanA = chanAB[0]
        chanB = chanAB[1]

    # get the left-hand side of condition
    lhs = source_Sv["Sv"].sel(channel=chanA) - source_Sv["Sv"].sel(channel=chanB)

    # create mask using operator lookup table
    da = xr.where(str2ops[operator](lhs, diff), True, False)

    # assign a name to DataArray
    da.name = "mask"

    # assign provenance attributes
    mask_attrs = {"mask_type": "frequency differencing"}
    history_attr = (
        f"{datetime.datetime.utcnow()} +00:00. "
        "Mask created by mask.frequency_differencing. "
        f"Operation: Sv['{chanA}'] - Sv['{chanB}'] {operator} {diff}"
    )

    da = da.assign_attrs({**mask_attrs, **{"history": history_attr}})
    return da


def get_shoal_mask(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    desired_channel: str,
    mask_type: str = "will",
    **kwargs,
):
    """
    Wrapper function for (future) multiple shoal masking algorithms
    (currently, only MOVIES-B (Will) is implemented)

    Args:
        source_Sv: xr.Dataset or str or pathlib.Path
                    If a Dataset this value contains the Sv data to create a mask for,
                    else it specifies the path to a zarr or netcdf file containing
                    a Dataset. This input must correspond to a Dataset that has the
                    coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
        desired_channel: str specifying the channel to generate the mask on
        mask_type: string specifying the algorithm to use
                    currently, 'weill' is the only one implemented

    Returns
    -------
    mask: xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.
    mask_: xr.DataArray
        A DataArray containing the mask for areas in which shoals were searched.
        Edge regions are filled with 'False', whereas the portion
        in which shoals could be detected is 'True'


    Raises
    ------
    ValueError
        If 'weill' is not given
    """
    assert mask_type in ["will"]
    if mask_type == "will":
        # Define a list of the keyword arguments your function can handle
        valid_args = {"thr", "maxvgap", "maxhgap", "minvlen", "minhlen"}
        # Filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask, mask_ = shoal_weill(source_Sv, desired_channel, **filtered_kwargs)
    else:
        raise ValueError("The provided mask type must be Will")
    return_mask = xr.DataArray(
        mask,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": source_Sv.ping_time, "range_sample": source_Sv.range_sample},
    )
    return_mask_ = xr.DataArray(
        mask_,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": source_Sv.ping_time, "range_sample": source_Sv.range_sample},
    )
    return return_mask, return_mask_


def get_seabed_mask(
    source_Sv: Union[xr.Dataset, str, pathlib.Path],
    desired_channel: str,
    mask_type: str = "ariza",
    **kwargs,
) -> xr.DataArray:
    """
    Create a mask based on the identified signal attenuations of Sv values.
    Parameters
    ----------
    source_Sv: xr.Dataset or str or pathlib.Path
        If a Dataset this value contains the Sv data to create a mask for,
        else it specifies the path to a zarr or netcdf file containing
        a Dataset. This input must correspond to a Dataset that has the
        coordinate ``channel`` and variables ``frequency_nominal`` and ``Sv``.
    desired_channel: str - channel to generate the mask for
    mask_type: str with either "ariza", "experimental", "blackwell_mod",
                                "blackwell", "deltaSv", "maxSv"
                                based on the preferred method for signal attenuation mask generation
    Returns
    -------
    xr.DataArray
        A DataArray containing the mask for the Sv data. Regions satisfying the thresholding
        criteria are filled with ``True``, else the regions are filled with ``False``.

    Raises
    ------
    ValueError
        If neither "ariza", "experimental", "blackwell_mod",
        "blackwell", "deltaSv", "maxSv" are given

    Notes
    -----


    Examples
    --------

    """
    assert mask_type in [
        "ariza",
        "experimental",
        "blackwell_mod",
        "blackwell",
        "deltaSv",
        "maxSv",
    ], "mask_type must be either 'ariza', 'experimental', 'blackwell', 'maxSv', 'deltaSv'"

    channel_Sv = source_Sv.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"].values.T
    r = source_Sv["echo_range"].values[0, 0]

    if mask_type == "ariza":
        # Define a list of the keyword arguments your function can handle
        valid_args = {"r0", "r1", "roff", "thr", "ec", "ek", "dc", "dk"}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_ariza(Sv, r, **filtered_kwargs)
    elif mask_type == "experimental":
        # Define a list of the keyword arguments your function can handle
        valid_args = {"r0", "r1", "roff", "thr", "ns", "n_dil"}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_experimental(Sv, r, **filtered_kwargs)
    elif mask_type == "blackwell":
        # Define a list of the keyword arguments your function can handle
        # valid_args = {"theta", "phi", "r0", "r1", "tSv", "ttheta", "tphi", "wtheta", "wphi"}
        valid_args = {"r0", "r1", "tSv", "ttheta", "tphi", "wtheta", "wphi"}
        theta = channel_Sv["angle_alongship"].values.T
        phi = channel_Sv["angle_athwartship"].values.T
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_blackwell(Sv, r, theta=theta, phi=phi, **filtered_kwargs)
    elif mask_type == "blackwell_mod":
        # Define a list of the keyword arguments your function can handle
        valid_args = {
            # "theta",
            # "phi",
            "r0",
            "r1",
            "tSv",
            "ttheta",
            "tphi",
            "wtheta",
            "wphi",
            "rlog",
            "tpi",
            "freq",
            "rank",
        }
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        theta = channel_Sv["angle_alongship"].values.T
        phi = channel_Sv["angle_athwartship"].values.T
        mask = _get_seabed_mask_blackwell_mod(Sv, r, theta=theta, phi=phi, **filtered_kwargs)
    elif mask_type == "deltaSv":
        # Define a list of the keyword arguments your function can handle
        valid_args = {"r0", "r1", "roff", "thr"}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_deltaSv(Sv, r, **filtered_kwargs)
    elif mask_type == "maxSv":
        # Define a list of the keyword arguments your function can handle
        valid_args = {"r0", "r1", "roff", "thr"}
        # Use dictionary comprehension to filter out any kwargs not in your list
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        mask = _get_seabed_mask_maxSv(Sv, r, **filtered_kwargs)
    else:
        raise ValueError(
            "The provided mask_type must be 'ariza', "
            + "'experimental', 'blackwell', 'maxSv' or 'deltaSv'!"
        )

    mask = np.logical_not(mask.T)
    return_mask = xr.DataArray(
        mask,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": source_Sv.ping_time, "range_sample": source_Sv.range_sample},
    )
    return return_mask
