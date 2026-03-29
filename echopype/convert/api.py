from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence, Tuple, Union

import dask.array
import fsspec
import numpy as np
import xarray as xr
from xarray import DataTree

# fmt: off
# black and isort have conflicting ideas about how this should be formatted
from ..core import SONAR_MODELS

if TYPE_CHECKING:
    from ..core import EngineHint, PathHint, SonarModelsHint
# fmt: on
from ..echodata.echodata import XARRAY_ENGINE_MAP, EchoData
from ..utils import io
from ..utils.coding import COMPRESSION_SETTINGS, sanitize_dtypes, set_storage_encodings
from ..utils.log import _init_logger
from ..utils.prov import add_processing_level, echopype_prov_attrs, source_files_vars

BEAM_SUBGROUP_DEFAULT = "Beam_group1"

# Logging setup
logger = _init_logger(__name__)


def to_file(
    echodata: EchoData,
    engine: "EngineHint",
    save_path: Optional["PathHint"] = None,
    compress: bool = True,
    overwrite: bool = False,
    parallel: bool = False,
    output_storage_options: Dict[str, str] = {},
    **kwargs,
):
    """Save content of EchoData to netCDF or zarr.

    Parameters
    ----------
    engine : str {'netcdf4', 'zarr'}
        type of converted file
    save_path : str
        path that converted .nc file will be saved
    compress : bool
        whether or not to perform compression on data variables
        Defaults to ``True``
    overwrite : bool
        whether or not to overwrite existing files
        Defaults to ``False``
    parallel : bool
        whether or not to use parallel processing. (Not yet implemented)
    output_storage_options : dict
        Additional keywords to pass to the filesystem class.
    **kwargs : dict, optional
        Extra arguments to either `xr.Dataset.to_netcdf`
        or `xr.Dataset.to_zarr`: refer to each method documentation
        for a list of all possible arguments.

    """
    if parallel:
        raise NotImplementedError("Parallel conversion is not yet implemented.")
    if engine not in XARRAY_ENGINE_MAP.values():
        raise ValueError("Unknown type to convert file to!")

    # Assemble output file names and path
    output_file = io.validate_output_path(
        source_file=echodata.source_file,
        engine=engine,
        save_path=save_path,
        output_storage_options=output_storage_options,
    )

    # Get all existing files
    fs = fsspec.get_mapper(output_file, **output_storage_options).fs  # get file system
    exists = True if fs.exists(output_file) else False

    # Sequential or parallel conversion
    if exists and not overwrite:
        logger.info(
            f"{echodata.source_file} has already been converted to {engine}. "  # noqa
            f"File saving not executed."
        )
    else:
        if exists:
            logger.info(f"overwriting {output_file}")
        else:
            logger.info(f"saving {output_file}")
        _save_groups_to_file(
            echodata,
            output_path=io.sanitize_file_path(
                file_path=output_file, storage_options=output_storage_options
            ),
            engine=engine,
            compress=compress,
            **kwargs,
        )

    # Link path to saved file with attribute as if from open_converted
    echodata.converted_raw_path = output_file


def has_zero_length_dim(dataset):
    return any(size == 0 for size in dataset.sizes.values())


def remove_zero_length_vars(dataset):
    # Clear chunking encoding for variables with zero-length dimensions
    for var_name, var in dataset.variables.items():
        if any(dataset.sizes[dim] == 0 for dim in var.dims):
            var.encoding.pop('chunks', None)
            var.encoding.pop('chunksizes', None)
    return dataset


def _save_groups_to_file(echodata, output_path, engine, compress=True, **kwargs):
    """Serialize all groups to file using DataTree-native I/O.

    Uses ``DataTree.to_netcdf`` or ``DataTree.to_zarr`` to write the entire
    tree in a single call, replacing the previous per-group save loop.
    """
    tree = echodata._tree
    if tree is None:
        raise ValueError("EchoData has no DataTree to save.")

    compression_settings = COMPRESSION_SETTINGS[engine] if compress else None

    # Sanitize dtypes in each node (e.g. convert object → str) and build
    # the nested encoding dict keyed by group path (e.g. "/", "/Environment").
    # Use inherit=False so each group only produces encodings for its own
    # variables, not coordinates inherited from parent nodes.
    encoding = {}
    for group_path in tree.groups:
        node = tree[group_path] if group_path != "/" else tree
        if not (node.has_data or node.has_attrs):
            continue
        ds = node.to_dataset(inherit=False)
        if len(ds.variables) == 0:
            continue
        # Handle zero-length dimensions before encoding
        if has_zero_length_dim(ds):
            ds = remove_zero_length_vars(ds)
        ds = sanitize_dtypes(ds)
        group_encoding = set_storage_encodings(ds, compression_settings, engine)

        # For zarr: align dask chunks with encoding chunks to avoid
        # "overlapping chunks" errors during parallel writes.
        if engine == "zarr":
            for var, enc in group_encoding.items():
                if var in ds and isinstance(ds[var].data, dask.array.Array):
                    enc_chunks = enc.get("chunks")
                    if enc_chunks is not None:
                        ds[var] = ds[var].chunk(
                            dict(zip(ds[var].dims, enc_chunks))
                        )

        node.dataset = ds
        encoding[group_path] = group_encoding

    if engine == "netcdf4":
        if isinstance(output_path, fsspec.FSMap):
            # DataTree.to_netcdf requires a file path, not an FSMap
            file_path = output_path.root
        else:
            file_path = str(output_path)
        tree.to_netcdf(
            file_path,
            mode="w",
            engine="netcdf4",
            encoding=encoding,
            write_inherited_coords=True,
            **kwargs,
        )
    elif engine == "zarr":
        if isinstance(output_path, fsspec.FSMap):
            store = output_path.root
        else:
            store = str(output_path)
        tree.to_zarr(
            store,
            mode="w",
            encoding=encoding,
            write_inherited_coords=True,
            **kwargs,
        )
    else:
        raise ValueError(f"{engine} is not a supported save format")


def _set_convert_params(param_dict: Dict[str, str]) -> Dict[str, str]:
    """Set parameters (metadata) that may not exist in the raw files.

    The default set of parameters include:
    - Platform group: ``platform_name``, ``platform_type``, ``platform_code_ICES``, ``water_level``
    - Top-level group: ``survey_name``

    Other parameters will be saved to the top level.

    # TODO: revise docstring, give examples.
    Examples
    --------
    # set parameters that may not already be in source files
    echodata.set_param({
        'platform_name': 'OOI',
        'platform_type': 'mooring'
    })
    """
    out_params = dict()

    # Parameters for the Platform group
    out_params["platform_name"] = param_dict.get("platform_name", "")
    out_params["platform_code_ICES"] = param_dict.get("platform_code_ICES", "")
    out_params["platform_type"] = param_dict.get("platform_type", "")
    out_params["water_level"] = param_dict.get("water_level", None)

    # Parameters for the Top-level group
    out_params["survey_name"] = param_dict.get("survey_name", "")
    for k, v in param_dict.items():
        if k not in out_params:
            out_params[k] = v

    return out_params


def _check_file(
    raw_file,
    sonar_model: "SonarModelsHint",
    xml_path: Optional["PathHint"] = None,
    include_bot: bool = False,
    include_idx: bool = False,
    storage_options: Dict[str, str] = {},
) -> Tuple[str, str, str, str]:
    """Checks whether the file and/or xml file exists and
    whether they have the correct extensions.

    Parameters
    ----------
    raw_file : str
        path to raw data file
    sonar_model : str
        model of the sonar instrument
    xml_path : str
        path to XML config file used by AZFP
    include_bot : bool, default `False`
        Include bottom depth file in parsing. Only used by EK60/EK80.
    include_index : bool, default `False`
        Include index file in parsing. Only used by EK60/EK80.
    storage_options : dict
        options for cloud storage

    Returns
    -------
    raw_file : str
        path to existing raw data file
    xml : str
        path to existing xml file
        empty string if no xml file is required for the specified model
    bot_file : str
        Path to existing bot file.
        Empty string if `.bot` file is not requested to be parsed and/or
        `.bot` parsing is not allowed by the specified model.
    idx_file : str
        Path to existing idx file.
        Empty string if `.idx` file is not requested to be parsed and/or
        `.idx` parsing is not allowed by the specified model.
    """
    if SONAR_MODELS[sonar_model]["xml"]:  # if this sonar model expects an XML file
        if not xml_path:
            raise ValueError(f"XML file is required for {sonar_model} raw data")
        else:
            if ".XML" not in Path(xml_path).suffix.upper():
                raise ValueError(f"{Path(xml_path).name} is not an XML file")

        xmlmap = fsspec.get_mapper(str(xml_path), **storage_options)
        if not xmlmap.fs.exists(xmlmap.root):
            raise FileNotFoundError(f"There is no file named {Path(xml_path).name}")

        xml = xml_path
    else:
        xml = ""

    # Check .bot file
    if SONAR_MODELS[sonar_model]["accepts_bot"] and include_bot:
        bot_file = str(Path(raw_file).with_suffix(".bot"))
        bot_fsmap = fsspec.get_mapper(bot_file, **storage_options)
        if not bot_fsmap.fs.exists(bot_fsmap.root):
            raise FileNotFoundError(
                f"There is no file named {bot_file}. The .BOT file must be contained in "
                + " the same directory as that of the input 'raw' file."
            )
    else:
        bot_file = ""

    # Check .idx file
    if SONAR_MODELS[sonar_model]["accepts_idx"] and include_idx:
        idx_file = str(Path(raw_file).with_suffix(".idx"))
        idx_fsmap = fsspec.get_mapper(idx_file, **storage_options)
        if not idx_fsmap.fs.exists(idx_fsmap.root):
            raise FileNotFoundError(
                f"There is no file named {idx_file}. The .IDX file must be contained in "
                + " the same directory as that of the input 'raw' file."
            )
    else:
        idx_file = ""

    # TODO: https://github.com/OSOceanAcoustics/echopype/issues/229
    #  to add compatibility for pathlib.Path objects for local paths
    fsmap = fsspec.get_mapper(raw_file, **storage_options)
    validate_ext = SONAR_MODELS[sonar_model]["validate_ext"]
    if not fsmap.fs.exists(fsmap.root):
        raise FileNotFoundError(f"There is no file named {Path(raw_file).name}")
    validate_ext(Path(raw_file).suffix.upper())

    return str(raw_file), str(xml), bot_file, idx_file


@add_processing_level("L1A", is_echodata=True)
def open_raw(
    raw_file: "PathHint",
    sonar_model: "SonarModelsHint",
    xml_path: Optional["PathHint"] = None,
    include_bot: bool = False,
    include_idx: bool = False,
    convert_params: Optional[Dict[str, str]] = None,
    storage_options: Optional[Dict[str, str]] = None,
    use_swap: Union[bool, Literal["auto"]] = False,
    max_chunk_size: str = "100MB",
) -> EchoData:
    """Create an EchoData object containing parsed data from a single raw data file.

    The EchoData object can be used for adding metadata and ancillary data
    as well as to serialize the parsed data to zarr or netcdf.

    Parameters
    ----------
    raw_file : str
        path to raw data file
    sonar_model : str
        model of the sonar instrument

        - ``EK60``: Kongsberg Simrad EK60 echosounder
        - ``ES70``: Kongsberg Simrad ES70 echosounder
        - ``EK80``: Kongsberg Simrad EK80 echosounder
        - ``EA640``: Kongsberg EA640 echosounder
        - ``AZFP``: ASL Environmental Sciences AZFP echosounder
        - ``AZFP6``: ASL Environmental Sciences AZFP echosounder (ULS6)
        - ``AD2CP``: Nortek Signature series ADCP
          (tested with Signature 500 and Signature 1000)

    xml_path : str
        path to XML config file used by AZFP
    include_bot : bool, default `False`
        Include bottom depth file in parsing. Only used by EK60/EK80.
    include_index : bool, default `False`
        Include index file in parsing. Only used by EK60/EK80.
    convert_params : dict
        parameters (metadata) that may not exist in the raw file
        and need to be added to the converted file
    storage_options : dict, optional
        options for cloud storage
    use_swap: bool or "auto", default False
        Flag to use disk swap in case of a large memory footprint.
        When set to ``True`` (or when set to "auto" and large memory footprint is needed,
        this function will create a temporary zarr store at the operating system's
        temporary directory.
    max_mb : int
        The maximum data chunk size in Megabytes (MB), when offloading
        variables with a large memory footprint to a temporary zarr store


    Returns
    -------
    EchoData object

    Raises
    ------
    ValueError
        If ``sonar_model`` is ``None`` or ``sonar_model``
        given is unsupported.
    FileNotFoundError
        If ``raw_file`` is ``None``.
    TypeError
        If ``raw_file`` input is neither ``str`` or
        ``pathlib.Path`` type.

    Notes
    -----
    In case of a large memory footprint, the program will determine if using
    a temporary swap space is needed. If so, it will use that space
    during conversion to prevent out of memory errors.

    Users can override this behaviour by either passing
    ``use_swap=True`` or ``use_swap=False``. If a keyword "auto" is
    used for the ``use_swap`` parameter, echopype will determine the usage of
    swap space automatically.

    This feature is only available for the following
    echosounders: EK60, ES70, EK80, ES80, EA640.
    """
    if raw_file is None:
        raise FileNotFoundError("The path to the raw data file must be specified.")

    # Check for path type
    if isinstance(raw_file, Path):
        raw_file = str(raw_file)
    if not isinstance(raw_file, str):
        raise TypeError("File path must be a string or Path")

    if sonar_model is None:
        raise ValueError("Sonar model must be specified.")

    # Check inputs
    if convert_params is None:
        convert_params = {}
    storage_options = storage_options if storage_options is not None else {}

    # Uppercased model in case people use lowercase
    sonar_model = sonar_model.upper()  # type: ignore

    # Check models
    if sonar_model not in SONAR_MODELS:
        raise ValueError(
            f"Unsupported echosounder model: {sonar_model}\nMust be one of: {list(SONAR_MODELS)}"  # noqa
        )

    # Check file extension and existence
    file_chk, xml_chk, bot_chk, idx_chk = _check_file(
        raw_file, sonar_model, xml_path, include_bot, include_idx, storage_options
    )

    # Parse raw file and organize data into groups
    parser = SONAR_MODELS[sonar_model]["parser"](
        file_chk,
        # Currently used only for AZFP XML File
        file_meta=xml_chk,
        # `bot_file` and `idx_file` used only for EK60/EK80 parsing
        bot_file=bot_chk,
        idx_file=idx_chk,
        storage_options=storage_options,
        sonar_model=sonar_model,
    )
    # Actually parse the raw datagrams from source file
    parser.parse_raw()

    # Direct offload to zarr and rectangularization only available for some sonar models
    # No rectangularization for other sonar models not listed below
    if sonar_model in ["EK60", "ES70", "EK80", "ES80", "EA640"]:
        # Perform rectangularization and offload to zarr
        # if the data expansion is too large to fit in memory
        parser.rectangularize_data(
            use_swap=use_swap,
            max_chunk_size=max_chunk_size,
        )

    setgrouper = SONAR_MODELS[sonar_model]["set_groups"](
        parser,
        input_file=file_chk,
        xml_path=xml_chk,
        output_path=None,
        sonar_model=sonar_model,
        params=_set_convert_params(convert_params),
    )

    # Setup tree dictionary
    tree_dict = {}

    # Top-level date_created varies depending on sonar model
    # Top-level is called "root" within tree
    if sonar_model in ["EK60", "ES70", "EK80", "ES80", "EA640"]:
        tree_dict["/"] = setgrouper.set_toplevel(
            sonar_model=sonar_model,
            date_created=parser.config_datagram["timestamp"],
        )
    else:
        tree_dict["/"] = setgrouper.set_toplevel(
            sonar_model=sonar_model, date_created=parser.ping_time[0]
        )
    tree_dict["Environment"] = setgrouper.set_env()
    tree_dict["Platform"] = setgrouper.set_platform()
    if sonar_model in ["EK60", "ES70", "EK80", "ES80", "EA640"]:
        tree_dict["Platform/NMEA"] = setgrouper.set_nmea()
    tree_dict["Provenance"] = setgrouper.set_provenance()
    # Allocate a tree_dict entry for Sonar? Otherwise, a DataTree error occurs
    tree_dict["Sonar"] = None

    # Set multi beam groups
    beam_groups = setgrouper.set_beam()

    beam_group_type = []
    for idx, beam_group in enumerate(beam_groups, start=1):
        if beam_group is not None:
            # fill in beam_group_type (only necessary for EK80, ES80, EA640)
            if idx == 1:
                # choose the appropriate description key for Beam_group1
                beam_group_type.append("complex" if "backscatter_i" in beam_group else "power")
            else:
                # provide None for all other beam groups (since the description does not have a key)
                beam_group_type.append(None)

            tree_dict[f"Sonar/Beam_group{idx}"] = beam_group

    if sonar_model in ["EK80", "ES80", "EA640"]:
        tree_dict["Sonar"] = setgrouper.set_sonar(beam_group_type=beam_group_type)
    else:
        tree_dict["Sonar"] = setgrouper.set_sonar()

    tree_dict["Vendor_specific"] = setgrouper.set_vendor()

    # Create tree and echodata
    # TODO: make the creation of tree dynamically generated from yaml
    tree = DataTree.from_dict(tree_dict, name="root")
    echodata = EchoData(source_file=file_chk, xml_path=xml_chk, sonar_model=sonar_model)
    echodata._set_tree(tree)
    echodata._load_tree()

    return echodata


# ---------------------------------------------------------------------------
# open_raw_multi — batch loader for multiple raw files
# ---------------------------------------------------------------------------

def _accumulate_parser_data(parser, accum, sorted_ch_all, sorted_ch_pc):
    """Extract per-file parsed data into accumulator lists.

    Parameters
    ----------
    parser : ParseEK
        Parser after parse_raw() + rectangularize_data().
    accum : dict
        Accumulator with keys for each data category.
    sorted_ch_all : list
        Sorted list of all channel IDs (set on first file).
    sorted_ch_pc : list
        Sorted list of power/complex channel IDs.
    """
    pdd = parser.ping_data_dict

    for ch in sorted_ch_pc:
        # Backscatter data (complex or power)
        has_data = False
        if ch in (parser.ch_ids.get("complex", [])):
            data = pdd.get("complex", {}).get(ch)
            if isinstance(data, dict):
                accum["complex_real"][ch].append(data["real"])
                accum["complex_imag"][ch].append(data["imag"])
                has_data = True
        elif ch in (parser.ch_ids.get("power", [])):
            power_data = pdd.get("power", {}).get(ch)
            if power_data is not None:
                accum["power"][ch].append(power_data)
                angle_data = pdd.get("angle", {}).get(ch)
                if angle_data is not None and ch in parser.ch_ids.get("angle", []):
                    accum["angle"][ch].append(angle_data)
                has_data = True

        if not has_data:
            continue

        # Per-ping beam metadata (only if we have backscatter data for this channel)
        accum["ping_time"][ch].append(parser.ping_time[ch])
        for key in [
            "sample_interval", "transmit_power", "slope",
            "channel_mode", "pulse_form", "offset",
        ]:
            if key in pdd and ch in pdd[key]:
                accum["ping_meta"][key][ch].append(np.asarray(pdd[key][ch]))

        # Pulse length/duration
        for key in ["pulse_length", "pulse_duration"]:
            if key in pdd and ch in pdd[key]:
                accum["ping_meta"]["pulse_duration"][ch].append(
                    np.asarray(pdd[key][ch], dtype="float32")
                )
                break

        # Frequency start/end for BB data
        if "frequency_start" in pdd and ch in pdd["frequency_start"]:
            accum["ping_meta"]["frequency_start"][ch].append(
                np.asarray(pdd["frequency_start"][ch])
            )
            accum["ping_meta"]["frequency_end"][ch].append(
                np.asarray(pdd["frequency_end"][ch])
            )

    # Transmit pulse (RAW4) data
    if hasattr(parser, "ping_data_dict_tx") and "complex" in parser.ping_data_dict_tx:
        for ch in sorted_ch_pc:
            if ch in parser.ping_data_dict_tx["complex"]:
                tx_data = parser.ping_data_dict_tx["complex"][ch]
                if isinstance(tx_data, dict):
                    accum["tx_real"][ch].append(tx_data["real"])
                    accum["tx_imag"][ch].append(tx_data["imag"])

    # NMEA
    accum["nmea_strings"].extend(parser.nmea.get("nmea_string", []))
    accum["nmea_timestamps"].extend(parser.nmea.get("timestamp", []))

    # MRU0
    for key in ["timestamp", "pitch", "roll", "heave", "heading"]:
        accum["mru0"][key].extend(parser.mru0.get(key, []))

    # MRU1
    for key in ["timestamp", "latitude", "longitude"]:
        accum["mru1"][key].extend(parser.mru1.get(key, []))


def _concat_pad(arrays, axis=0):
    """Concatenate arrays, padding along axis=1 if shapes differ."""
    if not arrays:
        return None
    shapes_1 = [a.shape[1] for a in arrays]
    max_s1 = max(shapes_1)
    if all(s == max_s1 for s in shapes_1):
        return np.concatenate(arrays, axis=axis)

    padded = []
    for a in arrays:
        if a.shape[1] < max_s1:
            pad_width = [(0, 0)] * a.ndim
            pad_width[1] = (0, max_s1 - a.shape[1])
            a = np.pad(a, pad_width, constant_values=np.nan)
        padded.append(a)
    return np.concatenate(padded, axis=axis)


def _build_combined_parser(accum, first_parser, sorted_ch_all, sorted_ch_pc):
    """Build a synthetic parser object with combined data for set_groups.

    Rather than calling set_groups per file, we concatenate the parsed numpy
    data across all files and stuff it back into a parser-like object so
    SetGroupsEK80 can build the xr.Datasets once on the combined data.
    """
    # Shallow copy to preserve config_datagram, environment, fil_coeffs, etc.
    import copy
    combined = copy.copy(first_parser)

    # Combined ping_time
    combined.ping_time = {}
    for ch in sorted_ch_pc:
        if accum["ping_time"][ch]:
            combined.ping_time[ch] = np.concatenate(accum["ping_time"][ch])
        else:
            combined.ping_time[ch] = np.array([], dtype="datetime64[ns]")

    # Combined ping_data_dict
    combined.ping_data_dict = defaultdict(lambda: defaultdict(list))

    # Complex backscatter
    for ch in sorted_ch_pc:
        if accum["complex_real"][ch]:
            combined.ping_data_dict["complex"][ch] = {
                "real": _concat_pad(accum["complex_real"][ch]),
                "imag": _concat_pad(accum["complex_imag"][ch]),
            }
        elif accum["power"][ch]:
            combined.ping_data_dict["power"][ch] = _concat_pad(accum["power"][ch])
            if accum["angle"][ch]:
                combined.ping_data_dict["angle"][ch] = _concat_pad(accum["angle"][ch])

    # Per-ping metadata
    for key in [
        "sample_interval", "transmit_power", "slope",
        "channel_mode", "pulse_form", "offset",
    ]:
        for ch in sorted_ch_pc:
            if accum["ping_meta"][key][ch]:
                combined.ping_data_dict[key][ch] = np.concatenate(
                    accum["ping_meta"][key][ch]
                )

    # Pulse duration
    for ch in sorted_ch_pc:
        if accum["ping_meta"]["pulse_duration"][ch]:
            combined.ping_data_dict["pulse_duration"][ch] = np.concatenate(
                accum["ping_meta"]["pulse_duration"][ch]
            )
            combined.ping_data_dict["pulse_length"][ch] = (
                combined.ping_data_dict["pulse_duration"][ch]
            )

    # Frequency start/end
    for ch in sorted_ch_pc:
        if accum["ping_meta"]["frequency_start"][ch]:
            combined.ping_data_dict["frequency_start"][ch] = np.concatenate(
                accum["ping_meta"]["frequency_start"][ch]
            )
            combined.ping_data_dict["frequency_end"][ch] = np.concatenate(
                accum["ping_meta"]["frequency_end"][ch]
            )

    # Transmit pulse (RAW4)
    combined.ping_data_dict_tx = defaultdict(lambda: defaultdict(list))
    for ch in sorted_ch_pc:
        if accum["tx_real"][ch]:
            combined.ping_data_dict_tx["complex"][ch] = {
                "real": _concat_pad(accum["tx_real"][ch]),
                "imag": _concat_pad(accum["tx_imag"][ch]),
            }

    # ch_ids — must be a defaultdict for SetGroupsEK80 compatibility
    combined.ch_ids = defaultdict(list, first_parser.ch_ids)

    # NMEA
    combined.nmea = {
        "nmea_string": accum["nmea_strings"],
        "timestamp": accum["nmea_timestamps"],
    }

    # MRU0
    combined.mru0 = {k: v for k, v in accum["mru0"].items()}

    # MRU1
    combined.mru1 = {k: v for k, v in accum["mru1"].items()}

    # bot/idx — not supported in multi mode
    combined.bot = defaultdict(list)
    combined.idx = defaultdict(list)
    combined.bot_file = ""
    combined.idx_file = ""

    return combined


@add_processing_level("L1A", is_echodata=True)
def open_raw_multi(
    raw_files: Sequence["PathHint"],
    sonar_model: "SonarModelsHint",
    xml_path: Optional["PathHint"] = None,
    convert_params: Optional[Dict[str, str]] = None,
    storage_options: Optional[Dict[str, str]] = None,
) -> EchoData:
    """Create an EchoData object by batch-parsing multiple raw files.

    Parses all files into numpy arrays first, concatenates at the array level,
    then builds the xarray/DataTree structure once. This is significantly
    faster than calling ``open_raw`` per file and then ``combine_echodata``,
    because it avoids per-file xarray Dataset construction and the subsequent
    re-concatenation.

    Currently supports EK80, ES80, and EA640 sonar models (EK60/ES70 planned).

    Parameters
    ----------
    raw_files : sequence of str or Path
        Paths to raw data files. Files are processed in the order given.
    sonar_model : str
        Sonar model (``'EK80'``, ``'ES80'``, ``'EA640'``).
    xml_path : str, optional
        Path to XML config file (only for AZFP).
    convert_params : dict, optional
        Additional metadata parameters.
    storage_options : dict, optional
        Options for cloud storage.

    Returns
    -------
    EchoData
        Combined EchoData object with all files' data.

    Raises
    ------
    ValueError
        If no valid files are found or sonar_model is unsupported.
    """
    if sonar_model is None:
        raise ValueError("Sonar model must be specified.")
    sonar_model = sonar_model.upper()
    if sonar_model not in SONAR_MODELS:
        raise ValueError(
            f"Unsupported echosounder model: {sonar_model}\n"
            f"Must be one of: {list(SONAR_MODELS)}"
        )
    if sonar_model not in ("EK80", "ES80", "EA640", "EK60", "ES70"):
        raise ValueError(
            f"open_raw_multi currently supports EK60/ES70/EK80/ES80/EA640, "
            f"got: {sonar_model}"
        )
    if convert_params is None:
        convert_params = {}
    storage_options = storage_options if storage_options is not None else {}

    parser_class = SONAR_MODELS[sonar_model]["parser"]

    # Accumulators
    accum = {
        "complex_real": defaultdict(list),
        "complex_imag": defaultdict(list),
        "power": defaultdict(list),
        "angle": defaultdict(list),
        "ping_time": defaultdict(list),
        "ping_meta": defaultdict(lambda: defaultdict(list)),
        "tx_real": defaultdict(list),
        "tx_imag": defaultdict(list),
        "nmea_strings": [],
        "nmea_timestamps": [],
        "mru0": defaultdict(list),
        "mru1": defaultdict(list),
    }

    first_parser = None
    sorted_ch_all = None
    sorted_ch_pc = None
    valid_files = []
    skipped = 0

    for raw_file in raw_files:
        raw_file = str(raw_file)
        try:
            # Validate file
            file_chk, xml_chk, _, _ = _check_file(
                raw_file, sonar_model, xml_path,
                include_bot=False, include_idx=False,
                storage_options=storage_options,
            )
            parser = parser_class(
                file_chk, file_meta=xml_chk, bot_file="", idx_file="",
                storage_options=storage_options, sonar_model=sonar_model,
            )
            parser.parse_raw()
            parser.rectangularize_data(use_swap=False)

            # ch_ids is populated during rectangularize_data
            if first_parser is None:
                first_parser = parser
                all_ch = list(parser.config_datagram["configuration"].keys())
                sorted_ch_all = sorted(all_ch)
                sorted_ch_pc = sorted(
                    parser.ch_ids.get("power", []) + parser.ch_ids.get("complex", [])
                )
                # Deduplicate: power channels may overlap with complex channels
                seen = set()
                sorted_ch_pc = [
                    x for x in sorted_ch_pc if not (x in seen or seen.add(x))
                ]
            else:
                # Skip files with different channel configuration
                this_ch = sorted(
                    parser.ch_ids.get("power", []) + parser.ch_ids.get("complex", [])
                )
                this_seen = set()
                this_ch = [x for x in this_ch if not (x in this_seen or this_seen.add(x))]
                if this_ch != sorted_ch_pc:
                    logger.warning(
                        f"Skipping {raw_file}: channel mismatch "
                        f"(expected {sorted_ch_pc}, got {this_ch})"
                    )
                    skipped += 1
                    continue

            _accumulate_parser_data(
                parser, accum, sorted_ch_all, sorted_ch_pc
            )
            valid_files.append(file_chk)

        except Exception as e:
            logger.warning(f"Skipping {raw_file}: {e}")
            skipped += 1

    if first_parser is None or not valid_files:
        raise ValueError(
            f"No valid files found among {len(raw_files)} input files "
            f"({skipped} skipped)."
        )

    logger.info(
        f"Parsed {len(valid_files)} files ({skipped} skipped), "
        f"building combined EchoData..."
    )

    # Build a combined parser-like object
    combined_parser = _build_combined_parser(
        accum, first_parser, sorted_ch_all, sorted_ch_pc
    )

    # Use SetGroups to build xr.Datasets once on combined data
    setgrouper_class = SONAR_MODELS[sonar_model]["set_groups"]
    setgrouper = setgrouper_class(
        combined_parser,
        input_file=valid_files[0],
        xml_path=xml_path or "",
        output_path=None,
        sonar_model=sonar_model,
        params=_set_convert_params(convert_params),
    )

    # Build tree_dict (same structure as open_raw)
    tree_dict = {}
    tree_dict["/"] = setgrouper.set_toplevel(
        sonar_model=sonar_model,
        date_created=first_parser.config_datagram["timestamp"],
    )
    tree_dict["Environment"] = setgrouper.set_env()
    tree_dict["Platform"] = setgrouper.set_platform()
    tree_dict["Platform/NMEA"] = setgrouper.set_nmea()

    # Provenance: list all source files
    prov_dict = echopype_prov_attrs(process_type="conversion")
    files_vars = source_files_vars(valid_files)
    if files_vars["meta_source_files_var"] is None:
        source_vars = files_vars["source_files_var"]
    else:
        source_vars = {
            **files_vars["source_files_var"],
            **files_vars["meta_source_files_var"],
        }
    tree_dict["Provenance"] = xr.Dataset(
        data_vars=source_vars,
        coords=files_vars["source_files_coord"],
        attrs=prov_dict,
    )

    tree_dict["Sonar"] = None
    beam_groups = setgrouper.set_beam()
    beam_group_type = []
    for idx, beam_group in enumerate(beam_groups, start=1):
        if beam_group is not None:
            if idx == 1:
                beam_group_type.append(
                    "complex" if "backscatter_i" in beam_group else "power"
                )
            else:
                beam_group_type.append(None)
            tree_dict[f"Sonar/Beam_group{idx}"] = beam_group

    if sonar_model in ("EK80", "ES80", "EA640"):
        tree_dict["Sonar"] = setgrouper.set_sonar(beam_group_type=beam_group_type)
    else:
        tree_dict["Sonar"] = setgrouper.set_sonar()

    tree_dict["Vendor_specific"] = setgrouper.set_vendor()

    # Create tree and EchoData
    tree = DataTree.from_dict(tree_dict, name="root")
    echodata = EchoData(
        source_file=valid_files[0],
        xml_path=xml_path or "",
        sonar_model=sonar_model,
    )
    echodata._set_tree(tree)
    echodata._load_tree()

    return echodata
