"""
Algorithms for masking seabed.
  These methods are based on:

    Copyright (c) 2020 Echopy

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    __authors__ = ['Alejandro Ariza'   # wrote maxSv(), deltaSv(), blackwell(),
                                       # blackwell_mod(), aliased2seabed(),
                                       # seabed2aliased(), ariza(), experimental()
  __authors__ = ['Mihai Boldeanu'
                  # adapted the mask seabed algorithms from the Echopy library
                  and implemented them for use with the Echopype library.
                ]
"""

import warnings

import dask.array as da
import numpy as np
import xarray as xr
from dask_image.ndmorph import binary_dilation, binary_erosion

MAX_SV_DEFAULT_PARAMS = {"r0": 10, "r1": 1000, "roff": 0, "thr": (-40, -60)}
DELTA_SV_DEFAULT_PARAMS = {"r0": 10, "r1": 1000, "roff": 0, "thr": 20}
BLACKWELL_DEFAULT_PARAMS = {
    "theta": None,
    "phi": None,
    "r0": 10,
    "r1": 1000,
    "tSv": -75,
    "ttheta": 702,
    "tphi": 282,
    "wtheta": 28,
    "wphi": 52,
}
BLACKWELL_MOD_DEFAULT_PARAMS = {
    "theta": None,
    "phi": None,
    "r0": 10,
    "r1": 1000,
    "tSv": -75,
    "ttheta": 702,
    "tphi": 282,
    "wtheta": 28,
    "wphi": 52,
    "rlog": None,
    "tpi": None,
    "freq": None,
    "rank": 50,
}
EXPERIMENTAL_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "roff": 0,
    "thr": (-30, -70),
    "ns": 150,
    "n_dil": 3,
}
ARIZA_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "roff": 0,
    "thr": -40,
    "ec": 1,
    "ek": (3, 3),
    "dc": 3,
    "dk": (3, 3),
}

ARIZA_ITERATIVE_DEFAULT_PARAMS = {
    "r0": 10,
    "r1": 1000,
    "roff": 0,
    "thr": -40,
    "ec": 1,
    "ek": (3, 3),
    "dc": 3,
    "dk": (3, 3),
    "s": 5,
    "eac": 3,
    "dac": 3,
    "maximum_spike": 500,
}


def _get_seabed_range(mask: xr.DataArray):
    """
    Given a seabed mask, returns the range_sample depth of the seabed

    Args:
        mask (xr.DataArray): seabed mask

    Returns:
        xr.DataArray: a ping_time-sized array containing the range_sample seabed depth,
        or max range_sample if no seabed is detected
    """
    seabed_depth = mask.argmax(dim="range_sample").compute()
    seabed_depth[seabed_depth == 0] = mask.range_sample.max().item()
    return seabed_depth


def _morpho(mask: xr.DataArray, operation: str, c: int, k: int):
    """
    Given a preexisting 1/0 mask, run erosion or dilation cycles on it to remove noise

    Args:
        mask (xr.DataArray): xr.DataArray with 1 and 0 data
        operation(str): dilation, erosion
        c (int): number of cycles.
        k (int): 2-elements tuple with vertical and horizontal dimensions
                      of the kernel.

    Returns:
        xr.DataArray: A DataArray containing the denoised mask.
            Regions satisfying the criteria are 1, others are 0
    """
    function_dict = {"dilation": binary_dilation, "erosion": binary_erosion}

    if c > 0:
        dask_mask = da.asarray(mask, allow_unknown_chunksizes=False)
        dask_mask.compute_chunk_sizes()
        dask_mask = function_dict[operation](
            dask_mask,
            structure=da.ones(shape=k, dtype=bool),
            iterations=c,
        ).compute()
        dask_mask = da.asarray(dask_mask, allow_unknown_chunksizes=False)
        dask_mask.compute()
        mask.values = dask_mask.compute()
    return mask


def _erode_dilate(mask: xr.DataArray, ec: int, ek: int, dc: int, dk: int):
    """
    Given a preexisting 1/0 mask, run erosion and dilation cycles on it to remove noise

    Args:
        mask (xr.DataArray): xr.DataArray with 1 and 0 data
        ec (int): number of erosion cycles.
        ek (int): 2-elements tuple with vertical and horizontal dimensions
                      of the erosion kernel.
        dc (int): number of dilation cycles.
        dk (int): 2-elements tuple with vertical and horizontal dimensions
                      of the dilation kernel.

    Returns:
        xr.DataArray: A DataArray containing the denoised mask.
            Regions satisfying the criteria are 1, others are 0
    """
    mask = _morpho(mask, "erosion", ec, ek)
    mask = _morpho(mask, "dilation", dc, dk)
    return mask


def _erode_dilate_iterative(
    mask: xr.DataArray,
    ec: int,
    ek: int,
    dc: int,
    dk: int,
    s: int,
    eac: int,
    dac: int,
    maximum_spike: int,
):
    """
    Given a preexisting 1/0 mask, run erosion and dilation cycles on it to remove noise
    until no seabed spikes larger than a certain value can be found

    Args:
        mask (xr.DataArray): xr.DataArray with 1 and 0 data
        ec (int): number of erosion cycles.
        ek (int): 2-elements tuple with vertical and horizontal dimensions
                      of the erosion kernel.

        dc (int): number of dilation cycles.
        dk (int): 2-elements tuple with vertical and horizontal dimensions
                      of the dilation kernel.
        s (int): additional iterations of the process
        eac (int): number of erosion cycles to apply in the additional iteration
        dac (int): number of erosion cycles to apply in the additional iteration
        maximum_spike (int): maximum spike permitted

    Returns:
        xr.DataArray: A DataArray containing the denoised mask.
            Regions satisfying the criteria are 1, others are 0
    """
    mask = _erode_dilate(mask, ec, ek, dc, dk)

    for i in range(s):
        seabed = _get_seabed_range(mask)
        spike = -seabed.diff(dim="ping_time", n=1).min().item()
        print(spike)
        if spike < maximum_spike:
            break
        mask = _erode_dilate(mask, eac, ek, dac, dk)

    return mask


def _create_range_mask(Sv_ds: xr.DataArray, desired_channel: str, thr: int, r0: int, r1: int):
    """
    Return a raw threshold/range mask for a certain dataset and desired channel

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        r0 (int): minimum range below which the search will be performed (m).
        r1 (int): maximum range above which the search will be performed (m).
        thr (int): Sv threshold above which seabed might occur (dB).

    Returns:
        dict: a dict containing the mask and whether or not further processing is necessary
            mask (xr.DataArray): a basic range/threshold mask.
                                Regions satisfying the criteria are 1, others are 0
            ok (bool): should the mask be further processed  or is there no data to be found?

    """
    channel_Sv = Sv_ds.sel(channel=desired_channel)
    Sv = channel_Sv["Sv"]
    r = channel_Sv["echo_range"][0]

    # return empty mask if searching range is outside the echosounder range
    if (r0 > r[-1]) or (r1 < r[0]):
        # Raise a warning to inform the user
        warnings.warn(
            "The searching range is outside the echosounder range. "
            "A default mask with all True values is returned, "
            "which won't mask any data points in the dataset."
        )
        mask = xr.DataArray(
            np.ones_like(Sv, dtype=bool),
            dims=("ping_time", "range_sample"),
            coords={"ping_time": Sv.ping_time, "range_sample": Sv.range_sample},
        )
        return {"mask": mask, "ok": False, "Sv": Sv, "range": r}

    # get upper and lower range indexes
    up = abs(r - r0).argmin(dim="range_sample").item()
    lw = abs(r - r1).argmin(dim="range_sample").item()

    # get threshold mask with shallow and deep waters masked
    mask = xr.where(Sv > thr, 1, 0).drop("channel")
    mask.fillna(0)
    range_filter = (mask["range_sample"] >= up) & (mask["range_sample"] <= lw)
    mask = mask.where(range_filter, other=0)

    # give empty mask if there is nothing above threshold
    if mask.sum() == 0:
        warnings.warn(
            "Nothing found above the threshold. " "A default mask with all True values is returned."
        )
        mask = xr.DataArray(
            np.ones_like(Sv, dtype=bool),
            dims=("ping_time", "range_sample"),
            coords={"ping_time": Sv.ping_time, "range_sample": Sv.range_sample},
        )
        return {"mask": mask, "ok": False, "Sv": Sv, "range": r}
    return {"mask": mask, "ok": True, "Sv": Sv, "range": r}


def _mask_down(mask: xr.DataArray):
    """
    Given a seabed mask, masks all signal under the detected seabed,
    """
    seabed_depth = _get_seabed_range(mask)
    mask = (mask["range_sample"] <= seabed_depth).transpose()
    return mask


def _experimental_correction(mask: xr.DataArray, Sv: xr.DataArray, thr: int):
    """
    Given an existing seabed mask, the single-channel dataset it was created on
    and a secondary, lower threshold, it builds the mask up until the Sv falls below the threshold

    """
    secondary_mask = xr.where(Sv < thr, 1, 0).drop("channel")
    secondary_mask.fillna(1)
    fill_mask = secondary_mask & mask
    return fill_mask


def _ariza(Sv_ds: xr.DataArray, desired_channel: str, parameters: dict = ARIZA_DEFAULT_PARAMS):
    """
    Mask Sv above a threshold to get potential seabed features. These features
    are eroded first to get rid of fake seabeds (spikes, schools, etc.) and
    dilated afterwards to fill in seabed breaches. Seabed detection is coarser
    than other methods (it removes water nearby the seabed) but the seabed line
    never drops when a breach occurs. Suitable for pelagic assessments and
    reconmended for non-supervised processing.

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        parameters: parameter dict, should contain:
            r0 (int): minimum range below which the search will be performed (m).
            r1 (int): maximum range above which the search will be performed (m).
            roff (int): seabed range offset (m).
            thr (int): Sv threshold above which seabed might occur (dB).
            ec (int): number of erosion cycles.
            ek (int): 2-elements tuple with vertical and horizontal dimensions
                      of the erosion kernel.
            dc (int): number of dilation cycles.
            dk (int): 2-elements tuple with vertical and horizontal dimensions
                      of the dilation kernel.


    Returns:
        xr.DataArray: A DataArray containing the mask for the Sv data.
            Regions satisfying the thresholding criteria are True, others are False
    """
    parameter_names = ["r0", "r1", "roff", "thr", "ec", "ek", "dc", "dk"]
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be: "
            + str(parameter_names)
            + ", are: "
            + str(parameters.keys())
        )
    r0 = parameters["r0"]
    r1 = parameters["r1"]
    thr = parameters["thr"]
    ec = parameters["ec"]
    ek = parameters["ek"]
    dc = parameters["dc"]
    dk = parameters["dk"]

    # create raw range and threshold mask, if no seabed is detected return empty
    raw = _create_range_mask(Sv_ds, desired_channel=desired_channel, thr=thr, r0=r0, r1=r1)
    mask = raw["mask"]
    if raw["ok"] is False:
        return mask

    # run erosion and dilation denoising cycles
    mask = _erode_dilate(mask, ec, ek, dc, dk)

    # mask areas under the detected seabed
    mask = _mask_down(mask)

    return mask


def _ariza_iterative(
    Sv_ds: xr.DataArray, desired_channel: str, parameters: dict = ARIZA_ITERATIVE_DEFAULT_PARAMS
):
    """
    Mask Sv above a threshold to get potential seabed features. These features
    are eroded first to get rid of fake seabeds (spikes, schools, etc.) and
    dilated afterwards to fill in seabed breaches. Seabed detection is coarser
    than other methods (it removes water nearby the seabed) but the seabed line
    never drops when a breach occurs. Suitable for pelagic assessments and
    reconmended for non-supervised processing.

    Args:
        Sv_ds (xr.DataArray): xr.DataArray with Sv data for multiple channels (dB)
        desired_channel(str): Name of the desired frequency channel
        parameters: parameter dict, should contain:
            r0 (int): minimum range below which the search will be performed (m).
            r1 (int): maximum range above which the search will be performed (m).
            roff (int): seabed range offset (m).
            thr (int): Sv threshold above which seabed might occur (dB).
            ec (int): number of erosion cycles.
            ek (int): 2-elements tuple with vertical and horizontal dimensions
                          of the erosion kernel.

            dc (int): number of dilation cycles.
            dk (int): 2-elements tuple with vertical and horizontal dimensions
                          of the dilation kernel.
            s (int): additional iterations of the process
            eac (int): number of erosion cycles to apply in the additional iteration
            dac (int): number of dilation cycles to apply in the additional iteration
            maximum_spike (int): maximum spike permitted

    Returns:
        xr.DataArray: A DataArray containing the mask for the Sv data.
            Regions satisfying the thresholding criteria are True, others are False
    """
    parameter_names = [
        "r0",
        "r1",
        "roff",
        "thr",
        "ec",
        "ek",
        "dc",
        "dk",
        "s",
        "eac",
        "dac",
        "maximum_spike",
    ]
    if not all(name in parameters.keys() for name in parameter_names):
        raise ValueError(
            "Missing parameters - should be: "
            + str(parameter_names)
            + ", are: "
            + str(parameters.keys())
        )
    r0 = parameters["r0"]
    r1 = parameters["r1"]
    thr = parameters["thr"]
    ec = parameters["ec"]
    ek = parameters["ek"]
    dc = parameters["dc"]
    dk = parameters["dk"]
    s = parameters["s"]
    eac = parameters["eac"]
    dac = parameters["dac"]
    maximum_spike = parameters["maximum_spike"]

    # create raw range and threshold mask, if no seabed is detected return empty
    raw = _create_range_mask(Sv_ds, desired_channel=desired_channel, thr=thr, r0=r0, r1=r1)
    mask = raw["mask"]
    if raw["ok"] is False:
        return mask

    # run erosion and dilation denoising cycles
    mask = _erode_dilate_iterative(
        mask, ec=ec, ek=ek, dc=dc, dk=dk, eac=eac, dac=dac, s=s, maximum_spike=maximum_spike
    )

    # mask areas under the detected seabed
    mask = _mask_down(mask)

    return mask
