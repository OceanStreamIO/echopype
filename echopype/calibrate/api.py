import numpy as np
import xarray as xr

from ..echodata import EchoData
from ..echodata.simrad import check_input_args_combination
from ..utils.log import _init_logger
from ..utils.prov import echopype_prov_attrs, source_files_vars
from .calibrate_azfp import CalibrateAZFP
from .calibrate_ek import CalibrateEK60, CalibrateEK80

CALIBRATOR = {
    "EK60": CalibrateEK60,
    "EK80": CalibrateEK80,
    "AZFP": CalibrateAZFP,
    "ES70": CalibrateEK60,
    "ES80": CalibrateEK80,
    "EA640": CalibrateEK80,
}

logger = _init_logger(__name__)


def _to_vec(da, ch_idx, n_pings):
    """Extract a 1-D ``(n_pings,)`` float64 vector from an xarray DataArray.

    Handles DataArrays with or without a ``channel`` dimension and varying
    dim order (e.g. ``transmit_power`` is ``(channel, ping_time)``).
    """
    if "channel" in da.dims:
        v = da.isel(channel=ch_idx)
    else:
        v = da
    val = np.atleast_1d(v.values.ravel()).astype(np.float64)
    if val.size == 1:
        val = np.full(n_pings, val[0])
    return val


def _compute_sv_gpu_ek80(echodata, cal_obj):
    """GPU-accelerated Sv for EK80 CW complex mode.

    Uses the already-initialised *cal_obj* (``CalibrateEK80``) to read
    calibration parameters, then computes Sv per channel on GPU via
    :func:`~echopype.calibrate.gpu_cal.compute_sv_from_complex_gpu`.

    Returns an ``xr.Dataset`` with the same structure as the standard path.
    """
    from .ek80_complex import get_filter_coeff, get_tau_effective, get_transmit_signal
    from .gpu_cal import compute_sv_from_complex_gpu
    from .range import range_mod_TVG_EK

    beam = echodata[cal_obj.ed_beam_group].sel(channel=cal_obj.chan_sel)
    vend = echodata["Vendor_specific"].sel(channel=cal_obj.chan_sel)

    # Calibration scalars
    z_et = float(cal_obj.cal_params["impedance_transducer"].values.flat[0])
    z_er = float(cal_obj.cal_params["impedance_transceiver"].values.flat[0])
    n_beams = int(beam["beam"].size)

    sound_speed_da = cal_obj.env_params["sound_speed"]
    absorption_da = cal_obj.env_params["sound_absorption"]
    range_meter = cal_obj.range_meter
    tvg_mod = range_mod_TVG_EK(echodata, cal_obj.ed_beam_group, range_meter, sound_speed_da)

    # Effective pulse length
    tx_coeff = get_filter_coeff(vend)
    fs = cal_obj.cal_params["receiver_sampling_frequency"]
    tx, tx_time = get_transmit_signal(beam, tx_coeff, "CW", fs)
    tau_eff = get_tau_effective(
        ytx_dict=tx,
        fs_deci_dict={k: 1 / np.diff(v[:2]) for (k, v) in tx_time.items()},
        waveform_mode="CW",
        channel=cal_obj.chan_sel,
        ping_time=beam["ping_time"],
    )
    ch_GPT = (vend["transceiver_type"] == "GPT").compute()
    tau_eff[ch_GPT] = beam["transmit_duration_nominal"][ch_GPT].isel(ping_time=0)

    # Per-channel GPU Sv
    n_channels = beam.sizes["channel"]
    n_pings = beam.sizes["ping_time"]
    n_range = beam.sizes["range_sample"]
    sv_all = np.empty((n_channels, n_pings, n_range), dtype=np.float64)

    for ch in range(n_channels):
        sv_all[ch] = compute_sv_from_complex_gpu(
            beam["backscatter_r"].isel(channel=ch).values,
            beam["backscatter_i"].isel(channel=ch).values,
            tvg_mod.isel(channel=ch).values,
            _to_vec(sound_speed_da, ch, n_pings),
            _to_vec(absorption_da, ch, n_pings),
            _to_vec(cal_obj.freq_center, ch, n_pings),
            _to_vec(beam["transmit_power"], ch, n_pings),
            _to_vec(cal_obj.cal_params["gain_correction"], ch, n_pings),
            _to_vec(cal_obj.cal_params["equivalent_beam_angle"], ch, n_pings),
            _to_vec(cal_obj.cal_params["sa_correction"], ch, n_pings),
            _to_vec(tau_eff, ch, n_pings),
            z_et, z_er, n_beams,
        )

    # Assemble output Dataset matching the standard calibration format
    sv_da = xr.DataArray(
        sv_all,
        dims=["channel", "ping_time", "range_sample"],
        coords={
            "channel": beam["channel"],
            "ping_time": beam["ping_time"],
            "range_sample": beam["range_sample"],
        },
        name="Sv",
    )
    ds = sv_da.to_dataset()
    ds = ds.merge(range_meter)
    ds["frequency_nominal"] = beam["frequency_nominal"]
    ds = cal_obj._add_params_to_output(ds)

    return ds


def _compute_cal(
    cal_type,
    echodata: EchoData,
    env_params=None,
    cal_params=None,
    ecs_file=None,
    waveform_mode=None,
    encode_mode=None,
    use_gpu="auto",
):
    # Make waveform_mode "FM" equivalent to "BB"
    waveform_mode = "BB" if waveform_mode == "FM" else waveform_mode

    # Check on waveform_mode and encode_mode inputs
    if echodata.sonar_model == "EK80":
        if waveform_mode is None or encode_mode is None:
            raise ValueError("waveform_mode and encode_mode must be specified for EK80 calibration")
        check_input_args_combination(waveform_mode=waveform_mode, encode_mode=encode_mode)
    elif echodata.sonar_model in ("EK60", "AZFP"):
        if waveform_mode is not None and waveform_mode != "CW":
            logger.warning(
                "This sonar model transmits only narrowband signals (waveform_mode='CW'). "
                "Calibration will be in CW mode",
            )
        if encode_mode is not None and encode_mode != "power":
            logger.warning(
                "This sonar model only record data as power or power/angle samples "
                "(encode_mode='power'). Calibration will be done on the power samples.",
            )

    # Set up calibration object
    cal_obj = CALIBRATOR[echodata.sonar_model](
        echodata,
        env_params=env_params,
        cal_params=cal_params,
        ecs_file=ecs_file,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
    )

    # Check Echodata Backscatter Size
    cal_obj._check_echodata_backscatter_size()

    # Perform calibration — optionally via GPU
    from ..utils.gpu import resolve_use_gpu

    _do_gpu = resolve_use_gpu(use_gpu)
    if (
        _do_gpu
        and cal_type == "Sv"
        and echodata.sonar_model in ("EK80", "ES80", "EA640")
        and waveform_mode == "CW"
        and encode_mode == "complex"
    ):
        cal_ds = _compute_sv_gpu_ek80(echodata, cal_obj)
        logger.info("compute_Sv: used GPU path (EK80 CW complex)")
    elif cal_type == "Sv":
        cal_ds = cal_obj.compute_Sv()
    elif cal_type == "TS":
        cal_ds = cal_obj.compute_TS()
    else:
        raise ValueError("cal_type must be Sv or TS")

    # Add attributes
    def add_attrs(cal_type, ds):
        """Add attributes to backscattering strength dataset.
        cal_type: Sv or TS
        """
        ds["range_sample"].attrs = {"long_name": "Along-range sample number, base 0"}
        ds["echo_range"].attrs = {"long_name": "Range distance", "units": "m"}
        ds[cal_type].attrs = {
            "long_name": {
                "Sv": "Volume backscattering strength (Sv re 1 m-1)",
                "TS": "Target strength (TS re 1 m^2)",
            }[cal_type],
            "units": "dB",
        }
        if echodata.sonar_model == "EK80":
            ds[cal_type] = ds[cal_type].assign_attrs(
                {
                    "waveform_mode": waveform_mode,
                    "encode_mode": encode_mode,
                }
            )

    add_attrs(cal_type, cal_ds)

    # Add provinance
    # Provenance source files may originate from raw files (echodata.source_files)
    # or converted files (echodata.converted_raw_path)
    if echodata.source_file is not None:
        source_file = echodata.source_file
    elif echodata.converted_raw_path is not None:
        source_file = echodata.converted_raw_path
    else:
        source_file = "SOURCE FILE NOT IDENTIFIED"

    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = f"calibrate.compute_{cal_type}"
    files_vars = source_files_vars(source_file)
    cal_ds = (
        cal_ds.assign(**files_vars["source_files_var"])
        .assign_coords(**files_vars["source_files_coord"])
        .assign_attrs(prov_dict)
    )

    # Add water_level to the created xr.Dataset
    if "water_level" in echodata["Platform"].data_vars.keys():
        cal_ds["water_level"] = echodata["Platform"].water_level

    return cal_ds


def compute_Sv(echodata: EchoData, use_gpu="auto", **kwargs) -> xr.Dataset:
    """
    Compute volume backscattering strength (Sv) from raw data.

    The calibration routine varies depending on the sonar type.
    Currently this operation is supported for the following ``sonar_model``:
    EK60, AZFP, EK80 (see Notes below for detail).

    Parameters
    ----------
    echodata : EchoData
        An `EchoData` object created by using `open_raw` or `open_converted`

    use_gpu : bool or {"auto"}, default "auto"
        GPU acceleration strategy:

        * ``"auto"`` — use GPU when CuPy + CUDA are available (transparent).
        * ``True``   — require GPU; raises ``RuntimeError`` if unavailable.
        * ``False``  — force CPU-only computation.

        Currently GPU-accelerated for EK80/ES80/EA640 with
        ``waveform_mode="CW"`` and ``encode_mode="complex"``.

    env_params : dict, optional
        Environmental parameters needed for calibration.
        Users can supply `"sound speed"` and `"absorption"` directly,
        or specify other variables that can be used to compute them,
        including `"temperature"`, `"salinity"`, and `"pressure"`.

        For EK60 and EK80 echosounders, by default echopype uses
        environmental variables stored in the data files.
        For AZFP echosounder, all environmental parameters need to be supplied.
        AZFP echosounders typically are equipped with an internal temperature
        sensor, and some are equipped with a pressure sensor, but automatically
        using these pressure data is not currently supported.

    cal_params : dict, optional
        Intrument-dependent calibration parameters.

        For EK60, EK80, and AZFP echosounders, by default echopype uses
        environmental variables stored in the data files.
        Users can optionally pass in custom values shown below.

        - for EK60 echosounder, allowed parameters include:
          `"sa_correction"`, `"gain_correction"`, `"equivalent_beam_angle"`
        - for AZFP echosounder, allowed parameters include:
          `"EL"`, `"DS"`, `"TVR"`, `"VTX0"`, `"equivalent_beam_angle"`, `"Sv_offset"`

        Passing in calibration parameters for other echosounders
        are not currently supported.

    waveform_mode : {"CW", "BB", "FM"}, optional
        Type of transmit waveform.
        Required only for data from the EK80 echosounder
        and not used with any other echosounder.

        - `"CW"` for narrowband transmission,
          returned echoes recorded either as complex or power/angle samples
        - `"BB"` or `"FM"` for broadband transmission,
          returned echoes recorded as complex samples

    encode_mode : {"complex", "power"}, optional
        Type of encoded return echo data.
        Required only for data from the EK80 echosounder
        and not used with any other echosounder.

        - `"complex"` for complex samples
        - `"power"` for power/angle samples, only allowed when
          the echosounder is configured for narrowband transmission

    Returns
    -------
    xr.Dataset
        The calibrated Sv dataset, including calibration parameters
        and environmental variables used in the calibration operations.

    Notes
    -----
    The EK80 echosounder can be configured to transmit
    either broadband/frequency modulated (``waveform_mode="BB"`` or ``waveform_mode="FM"``)
    or narrowband (``waveform_mode="CW"``) signals.
    When transmitting in broadband mode, the returned echoes are
    encoded as complex samples (``encode_mode="complex"``).
    When transmitting in narrowband mode, the returned echoes can be encoded
    either as complex samples (``encode_mode="complex"``)
    or as power/angle combinations (``encode_mode="power"``) in a format
    similar to those recorded by EK60 echosounders.

    The current calibration implemented for EK80 broadband complex data
    uses band-integrated Sv with the gain computed at the center frequency
    of the transmit signal.

    The returned xr.Dataset will contain the variable `water_level` from the
    EchoData object provided, if it exists. If `water_level` is not returned,
    it must be set using `EchoData.update_platform()`.
    """
    return _compute_cal(cal_type="Sv", echodata=echodata, use_gpu=use_gpu, **kwargs)


def compute_TS(echodata: EchoData, **kwargs):
    """
    Compute target strength (TS) from raw data.

    The calibration routine varies depending on the sonar type.
    Currently this operation is supported for the following ``sonar_model``:
    EK60, AZFP, EK80 (see Notes below for detail).

    Parameters
    ----------
    echodata : EchoData
        An `EchoData` object created by using `open_raw` or `open_converted`

    env_params : dict, optional
        Environmental parameters needed for calibration.
        Users can supply `"sound speed"` and `"absorption"` directly,
        or specify other variables that can be used to compute them,
        including `"temperature"`, `"salinity"`, and `"pressure"`.

        For EK60 and EK80 echosounders, by default echopype uses
        environmental variables stored in the data files.
        For AZFP echosounder, all environmental parameters need to be supplied.
        AZFP echosounders typically are equipped with an internal temperature
        sensor, and some are equipped with a pressure sensor, but automatically
        using these pressure data is not currently supported.

    cal_params : dict, optional
        Intrument-dependent calibration parameters.

        For EK60, EK80, and AZFP echosounders, by default echopype uses
        environmental variables stored in the data files.
        Users can optionally pass in custom values shown below.

        - for EK60 echosounder, allowed parameters include:
          `"sa_correction"`, `"gain_correction"`, `"equivalent_beam_angle"`
        - for AZFP echosounder, allowed parameters include:
          `"EL"`, `"DS"`, `"TVR"`, `"VTX0"`, `"equivalent_beam_angle"`, `"Sv_offset"`

        Passing in calibration parameters for other echosounders
        are not currently supported.

    waveform_mode : {"CW", "BB", "FM"}, optional
        Type of transmit waveform.
        Required only for data from the EK80 echosounder
        and not used with any other echosounder.

        - `"CW"` for narrowband transmission,
          returned echoes recorded either as complex or power/angle samples
        - `"BB"` or `"FM"` for broadband transmission,
          returned echoes recorded as complex samples

    encode_mode : {"complex", "power"}, optional
        Type of encoded return echo data.
        Required only for data from the EK80 echosounder
        and not used with any other echosounder.

        - `"complex"` for complex samples
        - `"power"` for power/angle samples, only allowed when
          the echosounder is configured for narrowband transmission

    Returns
    -------
    xr.Dataset
        The calibrated TS dataset, including calibration parameters
        and environmental variables used in the calibration operations.

    Notes
    -----
    The EK80 echosounder can be configured to transmit
    either broadband/frequency modulated (``waveform_mode="BB"`` or ``waveform_mode="FM"``)
    or narrowband (``waveform_mode="CW"``) signals.
    When transmitting in broadband mode, the returned echoes are
    encoded as complex samples (``encode_mode="complex"``).
    When transmitting in narrowband mode, the returned echoes can be encoded
    either as complex samples (``encode_mode="complex"``)
    or as power/angle combinations (``encode_mode="power"``) in a format
    similar to those recorded by EK60 echosounders.

    The current calibration implemented for EK80 broadband complex data
    uses band-integrated TS with the gain computed at the center frequency
    of the transmit signal.

    Note that in the fisheries acoustics context, it is customary to
    associate TS to a single scatterer.
    TS is defined as: TS = 10 * np.log10 (sigma_bs), where sigma_bs
    is the backscattering cross-section.

    For details, see:
    MacLennan et al. 2002. A consistent approach to definitions and
    symbols in fisheries acoustics. ICES J. Mar. Sci. 59: 365-369.
    https://doi.org/10.1006/jmsc.2001.1158
    """
    return _compute_cal(cal_type="TS", echodata=echodata, **kwargs)
