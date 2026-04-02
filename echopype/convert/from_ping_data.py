"""Construct EchoData from in-memory ping data arrays.

Provides :class:`PingAccumulator` for buffering decoded SampleData pings
from a real-time UDP stream (e.g. ``ek80_udp_client``) and
:func:`from_ping_data` for one-shot EchoData construction.

This module bridges the gap between echopype's file-based conversion
pipeline and real-time data acquisition, enabling GPU-accelerated
processing on IoT Edge devices without intermediate file I/O.

Example
-------
>>> from echopype.convert.from_ping_data import PingAccumulator
>>> acc = PingAccumulator()
>>> for ping in live_pings:
...     acc.add_ping(
...         timestamp=ping.time,
...         channel_id="WBT 1-1 ES200-7C",
...         frequency=200_000.0,
...         power_samples=ping.power,
...         sample_interval=ping.sample_interval,
...         transmit_power=ping.transmit_power,
...     )
>>> ed = acc.to_echodata(sonar_model="EK80")
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..utils.log import _init_logger

if TYPE_CHECKING:
    from ..echodata.echodata import EchoData

logger = _init_logger(__name__)


@dataclass
class ChannelConfig:
    """Static configuration for one transducer channel.

    Mirrors the per-channel ``config_datagram["configuration"][ch_id]``
    dict that :class:`~echopype.convert.parse_base.ParseEK` produces
    when reading a ``.raw`` file.
    """

    channel_id: str
    frequency: float  # Hz
    transducer_name: str = ""
    transducer_serial_number: str = ""
    serial_number: str = ""  # transceiver serial number
    transducer_frequency: float = 0.0  # alias (set from frequency)
    application_name: str = "EK80"
    application_version: str = ""
    channel_id_short: str = ""
    transducer_beam_type: int = 1  # 0=single, 1=split-beam
    beam_width_alongship: float = 7.0  # degrees
    beam_width_athwartship: float = 7.0
    equivalent_beam_angle: float = -20.7  # dB re 1 sr
    gain: float = 25.0  # dB (table or scalar)
    sa_correction: float = 0.0
    angle_offset_alongship: float = 0.0
    angle_offset_athwartship: float = 0.0
    angle_sensitivity_alongship: float = 23.0
    angle_sensitivity_athwartship: float = 23.0
    transducer_offset_x: float = 0.0
    transducer_offset_y: float = 0.0
    transducer_offset_z: float = 0.0
    transducer_alpha_x: float = 0.0
    transducer_alpha_y: float = 0.0
    transducer_alpha_z: float = 0.0
    beam_direction_x: float = 0.0
    beam_direction_y: float = 0.0
    beam_direction_z: float = 0.0
    impedance: float = 75.0  # ohms
    rx_bandwidth: float = 0.0
    rx_sample_frequency: float = 1500000.0  # receiver sampling frequency (Hz)
    transceiver_type: str = "WBT"  # transceiver type string
    pulse_duration: float = 0.001024  # s
    pulse_form: int = 0  # 0=CW, 1=LFM
    sample_interval: float = 0.0  # s (compute from data if 0)

    def __post_init__(self):
        if self.transducer_frequency == 0.0:
            self.transducer_frequency = self.frequency
        if not self.channel_id_short:
            self.channel_id_short = self.channel_id.split()[-1] if self.channel_id else ""

    def to_config_dict(self) -> Dict[str, Any]:
        """Convert to the dict format expected by ``SetGroupsEK80``."""
        return {
            "channel_id": self.channel_id,
            "transducer_frequency": self.transducer_frequency,
            "transducer_name": self.transducer_name,
            "transducer_serial_number": self.transducer_serial_number,
            "serial_number": self.serial_number,
            "application_name": self.application_name,
            "application_version": self.application_version,
            "channel_id_short": self.channel_id_short,
            "transducer_beam_type": self.transducer_beam_type,
            "beam_width_alongship": self.beam_width_alongship,
            "beam_width_athwartship": self.beam_width_athwartship,
            "equivalent_beam_angle": self.equivalent_beam_angle,
            "gain": [self.gain],
            "sa_correction": [self.sa_correction],
            "gain_table": [self.gain],
            "angle_offset_alongship": self.angle_offset_alongship,
            "angle_offset_athwartship": self.angle_offset_athwartship,
            "angle_sensitivity_alongship": self.angle_sensitivity_alongship,
            "angle_sensitivity_athwartship": self.angle_sensitivity_athwartship,
            "transducer_offset_x": self.transducer_offset_x,
            "transducer_offset_y": self.transducer_offset_y,
            "transducer_offset_z": self.transducer_offset_z,
            "transducer_alpha_x": self.transducer_alpha_x,
            "transducer_alpha_y": self.transducer_alpha_y,
            "transducer_alpha_z": self.transducer_alpha_z,
            "beam_direction_x": self.beam_direction_x,
            "beam_direction_y": self.beam_direction_y,
            "beam_direction_z": self.beam_direction_z,
            "impedance": self.impedance,
            "rx_bandwidth": self.rx_bandwidth,
            "rx_sample_frequency": self.rx_sample_frequency,
            "transceiver_type": self.transceiver_type,
            "pulse_duration": [self.pulse_duration],
            "pulse_form": self.pulse_form,
            "sample_interval": self.sample_interval,
        }


@dataclass
class PingRecord:
    """One ping of data from a single channel."""

    timestamp: np.datetime64
    channel_id: str
    power_samples: NDArray[np.int16]  # raw int16 samples
    angle_samples: Optional[NDArray[np.int16]] = None  # interleaved athwart/along
    transmit_power: float = 0.0
    pulse_duration: float = 0.001024
    sample_interval: float = 0.0
    frequency: float = 0.0
    sound_speed: float = 1500.0
    absorption: float = 0.0


class _MockParserEK80:
    """Lightweight stand-in for ``ParseEK80`` that satisfies ``SetGroupsEK80``.

    Holds only the attributes accessed by ``SetGroupsEK80.set_*()`` methods.
    This avoids running the full file-based parse pipeline.
    """

    def __init__(self):
        self.sonar_model = "EK80"
        self.source_file = "<ping_data>"
        self.config_datagram: Dict[str, Any] = {
            "timestamp": np.datetime64("now"),
            "configuration": {},
            "xml": "",
        }
        self.ch_ids: Dict[str, list] = defaultdict(list)
        self.ping_data_dict: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(list))
        self.ping_data_dict_tx: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.ping_time: Dict[str, list] = defaultdict(list)
        self.environment: Dict[str, Any] = {}
        self.nmea: Dict[str, list] = defaultdict(list)
        self.mru0: Dict[str, list] = defaultdict(list)
        self.mru1: Dict[str, list] = defaultdict(list)
        self.fil_coeffs: Dict = defaultdict(dict)
        self.fil_df: Dict = defaultdict(dict)
        self.bot: Dict[str, list] = defaultdict(list)
        self.bot_file: str = ""
        self.idx: Dict[str, list] = defaultdict(list)
        self.idx_file: str = ""
        self.num_range_sample_groups: Optional[int] = None
        self.data_types = ["power", "angle", "complex"]


class PingAccumulator:
    """Buffer decoded pings and convert to ``EchoData`` for processing.

    Designed for real-time UDP data acquisition where pings arrive one at a
    time from ``ek80_udp_client``.

    Parameters
    ----------
    sonar_model : str
        Echosounder model, e.g. ``"EK80"`` or ``"EK60"``.

    Attributes
    ----------
    channels : dict[str, ChannelConfig]
        Per-channel static configuration.  Register channels with
        :meth:`register_channel` before adding pings.
    """

    def __init__(self, sonar_model: str = "EK80"):
        self.sonar_model = sonar_model.upper()
        self.channels: Dict[str, ChannelConfig] = {}
        self._pings: Dict[str, List[PingRecord]] = defaultdict(list)
        self._nav_timestamps: List[np.datetime64] = []
        self._nav_lat: List[float] = []
        self._nav_lon: List[float] = []
        self._nav_heading: List[float] = []
        self._nav_speed: List[float] = []
        self._environment: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Channel registration
    # ------------------------------------------------------------------

    def register_channel(self, config: ChannelConfig) -> None:
        """Register a transducer channel with its static configuration.

        Must be called for each channel *before* :meth:`add_ping`.
        """
        self.channels[config.channel_id] = config

    def register_channel_simple(
        self,
        channel_id: str,
        frequency: float,
        **kwargs,
    ) -> None:
        """Convenience wrapper to register a channel from scalar parameters."""
        self.register_channel(
            ChannelConfig(channel_id=channel_id, frequency=frequency, **kwargs)
        )

    # ------------------------------------------------------------------
    # Data accumulation
    # ------------------------------------------------------------------

    def add_ping(
        self,
        timestamp: datetime.datetime | np.datetime64,
        channel_id: str,
        power_samples: NDArray[np.int16] | Sequence[int],
        *,
        angle_samples: Optional[NDArray[np.int16] | Sequence[int]] = None,
        transmit_power: float = 0.0,
        pulse_duration: float = 0.001024,
        sample_interval: float = 0.0,
        frequency: float = 0.0,
        sound_speed: float = 1500.0,
        absorption: float = 0.0,
    ) -> None:
        """Add one ping of sample data from a single channel.

        Parameters
        ----------
        timestamp
            Ping timestamp (UTC).
        channel_id
            Must match a previously registered channel.
        power_samples
            Raw int16 power samples (value / 100.0 = dB for EK80 UDP).
        angle_samples
            Optional interleaved athwart/along int16 angle samples.
        transmit_power
            Transmit power in watts.
        pulse_duration
            Actual pulse duration in seconds.
        sample_interval
            Sample interval in seconds.
        frequency
            Transmit frequency in Hz (overrides channel config if nonzero).
        sound_speed
            Sound speed in m/s.
        absorption
            Absorption coefficient in dB/m.
        """
        if channel_id not in self.channels:
            raise ValueError(
                f"Channel {channel_id!r} not registered. "
                f"Call register_channel() first. Known: {list(self.channels)}"
            )

        if isinstance(timestamp, datetime.datetime):
            timestamp = np.datetime64(timestamp, "ns")
        elif not isinstance(timestamp, np.datetime64):
            timestamp = np.datetime64(timestamp, "ns")

        power = np.asarray(power_samples, dtype=np.int16)
        angles = np.asarray(angle_samples, dtype=np.int16) if angle_samples is not None else None

        self._pings[channel_id].append(
            PingRecord(
                timestamp=timestamp,
                channel_id=channel_id,
                power_samples=power,
                angle_samples=angles,
                transmit_power=transmit_power,
                pulse_duration=pulse_duration,
                sample_interval=sample_interval or self.channels[channel_id].sample_interval,
                frequency=frequency or self.channels[channel_id].frequency,
                sound_speed=sound_speed,
                absorption=absorption,
            )
        )

    def add_navigation(
        self,
        timestamp: datetime.datetime | np.datetime64,
        latitude: float,
        longitude: float,
        heading: float = np.nan,
        speed: float = np.nan,
    ) -> None:
        """Add a navigation fix (GPS position)."""
        if isinstance(timestamp, datetime.datetime):
            timestamp = np.datetime64(timestamp, "ns")
        self._nav_timestamps.append(timestamp)
        self._nav_lat.append(latitude)
        self._nav_lon.append(longitude)
        self._nav_heading.append(heading)
        self._nav_speed.append(speed)

    def set_environment(
        self,
        sound_speed: float = 1500.0,
        temperature: float = 10.0,
        salinity: float = 35.0,
        depth: float = 0.0,
        **kwargs,
    ) -> None:
        """Set environment parameters for the buffer."""
        self._environment = {
            "sound_speed": sound_speed,
            "temperature": temperature,
            "salinity": salinity,
            "depth": depth,
            **kwargs,
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of pings across all channels."""
        return sum(len(pings) for pings in self._pings.values())

    @property
    def duration_seconds(self) -> float:
        """Time span of accumulated data in seconds."""
        all_times = []
        for pings in self._pings.values():
            for p in pings:
                all_times.append(p.timestamp)
        if len(all_times) < 2:
            return 0.0
        arr = np.array(all_times, dtype="datetime64[ns]")
        return float((arr.max() - arr.min()) / np.timedelta64(1, "s"))

    @property
    def channel_ids(self) -> List[str]:
        """Sorted list of channel IDs with data."""
        return sorted(self._pings.keys())

    # ------------------------------------------------------------------
    # Build EchoData
    # ------------------------------------------------------------------

    def to_echodata(self) -> "EchoData":
        """Convert accumulated pings into an ``EchoData`` object.

        Constructs a mock parser object that mirrors the internal data
        structures of ``ParseEK80``, then delegates to ``SetGroupsEK80``
        to build the SONAR-netCDF4 group datasets.

        Returns
        -------
        EchoData
            Fully formed EchoData ready for ``compute_Sv()``.

        Raises
        ------
        ValueError
            If no pings have been accumulated.
        """
        if len(self) == 0:
            raise ValueError("No pings accumulated — call add_ping() first.")

        from xarray import DataTree

        from ..core import SONAR_MODELS
        from ..echodata.echodata import EchoData

        parser = self._build_mock_parser()

        # Rectangularize power data (pad to max range_sample across pings)
        self._rectangularize(parser)

        # Use the standard SetGroups class for this sonar model
        setgrouper_cls = SONAR_MODELS[self.sonar_model]["set_groups"]
        setgrouper = setgrouper_cls(
            parser,
            input_file="<ping_data>",
            xml_path="",
            output_path=None,
            sonar_model=self.sonar_model,
            params={},
        )

        # Build tree dict (same sequence as open_raw)
        tree_dict: Dict[str, Any] = {}
        tree_dict["/"] = setgrouper.set_toplevel(
            sonar_model=self.sonar_model,
            date_created=parser.config_datagram["timestamp"],
        )
        tree_dict["Environment"] = setgrouper.set_env()
        tree_dict["Platform"] = setgrouper.set_platform()
        tree_dict["Platform/NMEA"] = setgrouper.set_nmea()
        tree_dict["Provenance"] = setgrouper.set_provenance()
        tree_dict["Sonar"] = None

        beam_groups = setgrouper.set_beam()
        beam_group_type: list = []
        for idx, beam_group in enumerate(beam_groups, start=1):
            if beam_group is not None:
                if idx == 1:
                    beam_group_type.append(
                        "complex" if "backscatter_i" in beam_group else "power"
                    )
                else:
                    beam_group_type.append(None)
                tree_dict[f"Sonar/Beam_group{idx}"] = beam_group

        tree_dict["Sonar"] = setgrouper.set_sonar(beam_group_type=beam_group_type)
        tree_dict["Vendor_specific"] = setgrouper.set_vendor()

        tree = DataTree.from_dict(tree_dict, name="root")
        echodata = EchoData(
            source_file="<ping_data>",
            xml_path="",
            sonar_model=self.sonar_model,
        )
        echodata._set_tree(tree)
        echodata._load_tree()

        return echodata

    # ------------------------------------------------------------------
    # Internal: build mock parser
    # ------------------------------------------------------------------

    def _build_mock_parser(self) -> _MockParserEK80:
        """Populate a ``_MockParserEK80`` from accumulated data."""
        parser = _MockParserEK80()
        parser.sonar_model = self.sonar_model

        # Only include channels that have at least one ping
        ch_ids_sorted = sorted(
            ch_id for ch_id in self.channels if self._pings.get(ch_id)
        )

        # --- config_datagram ---
        first_ping_time = None
        for ch_id in ch_ids_sorted:
            pings = self._pings.get(ch_id, [])
            if pings and (first_ping_time is None or pings[0].timestamp < first_ping_time):
                first_ping_time = pings[0].timestamp

        if first_ping_time is None:
            first_ping_time = np.datetime64("now")

        parser.config_datagram["timestamp"] = first_ping_time
        for ch_id in ch_ids_sorted:
            cfg = self.channels[ch_id].to_config_dict()
            parser.config_datagram["configuration"][ch_id] = cfg

        # --- ch_ids: classify channels as power/complex/angle ---
        for ch_id in ch_ids_sorted:
            parser.ch_ids["power"].append(ch_id)
            # Check if any ping has angle data
            pings = self._pings.get(ch_id, [])
            if any(p.angle_samples is not None for p in pings):
                parser.ch_ids["angle"].append(ch_id)

        # --- ping_time per channel ---
        for ch_id in ch_ids_sorted:
            pings = self._pings.get(ch_id, [])
            parser.ping_time[ch_id] = [p.timestamp for p in pings]

        # --- ping_data_dict: power, angle, and per-ping metadata ---
        # SetGroupsEK80.set_beam() reads per-ping metadata (pulse_form,
        # pulse_duration, pulse_length, sample_interval, transmit_power,
        # slope, channel_mode, offset) from ping_data_dict, NOT
        # ping_data_dict_tx.  We populate both locations for compatibility.
        for ch_id in ch_ids_sorted:
            pings = self._pings.get(ch_id, [])
            cfg = self.channels[ch_id]
            for p in pings:
                # Power data
                parser.ping_data_dict["power"][ch_id].append(
                    p.power_samples.astype(np.float64)
                )
                if p.angle_samples is not None:
                    n = len(p.angle_samples) // 2
                    athwart = p.angle_samples[0::2].astype(np.float64)
                    along = p.angle_samples[1::2].astype(np.float64)
                    parser.ping_data_dict["angle"][ch_id].append(
                        np.column_stack([athwart, along])
                    )

                # Per-ping metadata expected by set_beam()
                parser.ping_data_dict["pulse_form"][ch_id].append(cfg.pulse_form)
                parser.ping_data_dict["pulse_duration"][ch_id].append(p.pulse_duration)
                parser.ping_data_dict["pulse_length"][ch_id].append(p.pulse_duration)
                parser.ping_data_dict["sample_interval"][ch_id].append(p.sample_interval)
                parser.ping_data_dict["transmit_power"][ch_id].append(p.transmit_power)
                parser.ping_data_dict["slope"][ch_id].append(0.0)
                parser.ping_data_dict["channel_mode"][ch_id].append(0)
                parser.ping_data_dict["offset"][ch_id].append(0)
                parser.ping_data_dict["frequency_start"][ch_id].append(p.frequency)
                parser.ping_data_dict["frequency_end"][ch_id].append(p.frequency)

            # Also populate ping_data_dict_tx for transmit pulse info
            for p in pings:
                parser.ping_data_dict_tx["transmit_power"][ch_id].append(p.transmit_power)
                parser.ping_data_dict_tx["pulse_duration"][ch_id].append(p.pulse_duration)
                parser.ping_data_dict_tx["sample_interval"][ch_id].append(p.sample_interval)
                parser.ping_data_dict_tx["frequency"][ch_id].append(p.frequency)

        # --- environment ---
        parser.environment = self._environment.copy()
        if not parser.environment:
            parser.environment = {
                "sound_speed": 1500.0,
                "temperature": 10.0,
                "salinity": 35.0,
                "depth": 0.0,
                "acidity": 8.1,
            }
        # Ensure acidity is present (required by calibrate for EK80)
        if "acidity" not in parser.environment:
            parser.environment["acidity"] = 8.1
        # set_env() needs a timestamp to create a valid time1 coordinate
        if "timestamp" not in parser.environment:
            parser.environment["timestamp"] = np.datetime64(first_ping_time, "ns")

        # --- NMEA / navigation ---
        if self._nav_timestamps:
            # Build synthetic GGA NMEA sentences for the Platform group
            parser.nmea["timestamp"] = [np.datetime64(t, "ns") for t in self._nav_timestamps]
            parser.nmea["nmea_string"] = []
            for i, ts in enumerate(self._nav_timestamps):
                lat = self._nav_lat[i]
                lon = self._nav_lon[i]
                # Build a minimal NMEA GGA-like string for echopype's parser
                lat_deg = abs(lat)
                lat_min = (lat_deg - int(lat_deg)) * 60
                lat_str = f"{int(lat_deg):02d}{lat_min:07.4f}"
                lat_dir = "N" if lat >= 0 else "S"
                lon_deg = abs(lon)
                lon_min = (lon_deg - int(lon_deg)) * 60
                lon_str = f"{int(lon_deg):03d}{lon_min:07.4f}"
                lon_dir = "E" if lon >= 0 else "W"

                dt = ts.astype("datetime64[us]").astype(datetime.datetime)
                time_str = dt.strftime("%H%M%S.00") if hasattr(dt, "strftime") else "000000.00"

                nmea = (
                    f"$GPGGA,{time_str},{lat_str},{lat_dir},"
                    f"{lon_str},{lon_dir},1,08,0.9,0.0,M,0.0,M,,"
                )
                parser.nmea["nmea_string"].append(nmea)

        # --- MRU data (heading from navigation) ---
        if self._nav_heading and not all(np.isnan(h) for h in self._nav_heading):
            parser.mru0["timestamp"] = [np.datetime64(t, "ns") for t in self._nav_timestamps]
            parser.mru0["heading"] = self._nav_heading
            parser.mru0["pitch"] = [0.0] * len(self._nav_timestamps)
            parser.mru0["roll"] = [0.0] * len(self._nav_timestamps)
            parser.mru0["heave"] = [0.0] * len(self._nav_timestamps)

        return parser

    def _rectangularize(self, parser: _MockParserEK80) -> None:
        """Pad ragged power/angle arrays to rectangular numpy arrays.

        After this call, ``parser.ping_data_dict["power"][ch_id]`` is
        a single 2-D numpy array of shape ``(n_pings, max_range_samples)``
        with ``NaN`` padding for shorter pings.
        """
        for data_type in ["power", "angle"]:
            ch_data = parser.ping_data_dict[data_type]
            for ch_id in list(ch_data.keys()):
                arrays = ch_data[ch_id]
                if not arrays:
                    continue

                # Find max range_sample length
                max_len = max(arr.shape[-1] if arr.ndim >= 1 else len(arr) for arr in arrays)

                # Pad and stack
                padded = []
                for arr in arrays:
                    arr = np.asarray(arr, dtype=np.float64)
                    if arr.ndim == 1:
                        pad_width = max_len - len(arr)
                        if pad_width > 0:
                            arr = np.pad(arr, (0, pad_width), constant_values=np.nan)
                    elif arr.ndim == 2:
                        # angle data: (n_samples, 2) → pad along axis 0
                        pad_width = max_len - arr.shape[0]
                        if pad_width > 0:
                            arr = np.pad(
                                arr, ((0, pad_width), (0, 0)), constant_values=np.nan
                            )
                    padded.append(arr)

                ch_data[ch_id] = np.stack(padded)

        # Transmit params: convert lists to arrays
        for param in ["transmit_power", "pulse_duration", "sample_interval", "frequency"]:
            for ch_id in list(parser.ping_data_dict_tx[param].keys()):
                vals = parser.ping_data_dict_tx[param][ch_id]
                parser.ping_data_dict_tx[param][ch_id] = np.array(vals)

    # ------------------------------------------------------------------
    # Clear / reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Discard all accumulated data, keeping channel registrations."""
        self._pings.clear()
        self._nav_timestamps.clear()
        self._nav_lat.clear()
        self._nav_lon.clear()
        self._nav_heading.clear()
        self._nav_speed.clear()


def from_ping_data(
    pings: List[Dict[str, Any]],
    channels: List[ChannelConfig] | Dict[str, ChannelConfig],
    sonar_model: str = "EK80",
    environment: Optional[Dict[str, Any]] = None,
    navigation: Optional[List[Dict[str, Any]]] = None,
) -> "EchoData":
    """Construct an ``EchoData`` object from pre-parsed ping data.

    One-shot convenience function wrapping :class:`PingAccumulator`.

    Parameters
    ----------
    pings
        List of dicts, each with at least ``timestamp``, ``channel_id``,
        ``power_samples``.  Optional keys: ``angle_samples``,
        ``transmit_power``, ``pulse_duration``, ``sample_interval``,
        ``frequency``, ``sound_speed``, ``absorption``.
    channels
        Channel configurations — either a list of :class:`ChannelConfig`
        or a dict mapping channel_id → ChannelConfig.
    sonar_model
        Echosounder model string (``"EK80"``, ``"EK60"``).
    environment
        Environment parameters dict (``sound_speed``, ``temperature``, etc.).
    navigation
        Optional list of nav fix dicts with ``timestamp``, ``latitude``,
        ``longitude``, and optional ``heading``, ``speed``.

    Returns
    -------
    EchoData
    """
    acc = PingAccumulator(sonar_model=sonar_model)

    # Register channels
    if isinstance(channels, dict):
        for cfg in channels.values():
            acc.register_channel(cfg)
    else:
        for cfg in channels:
            acc.register_channel(cfg)

    # Set environment
    if environment:
        acc.set_environment(**environment)

    # Add navigation
    if navigation:
        for nav in navigation:
            acc.add_navigation(
                timestamp=nav["timestamp"],
                latitude=nav["latitude"],
                longitude=nav["longitude"],
                heading=nav.get("heading", np.nan),
                speed=nav.get("speed", np.nan),
            )

    # Add pings
    for ping in pings:
        acc.add_ping(**ping)

    return acc.to_echodata()
