import contextlib
import ctypes
import math
import os
import re
import select
import socket
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
import typing
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
DLG_IP_DEFAULT = "192.168.1.100"
DLG_PORT_DEFAULT = 41401
LOCAL_BIND_IP_DEFAULT = ""  # "" = 0.0.0.0
LOCAL_BIND_PORT_DEFAULT = 41402

DEFAULT_RATE_HZ = 200.0
DEFAULT_DURATION_S = 10.0
DEFAULT_CHANNELS = 8
WAIT_FIRST_MS = 3000
WAIT_FIRST_SAMPLES = 3

# Default channel config (same as the DLGlogger reference)
DEFAULT_TSENSOR = 4
DEFAULT_ILPF = 0
DEFAULT_IGAIN_IDX = 5
DEFAULT_SENSPWR_IDX = 2
DEFAULT_INPUT_DC = 0
DEFAULT_INPUT_AC = 0
DEFAULT_USE_BALANCE = 0
ENCODER_CHANNEL = 3
ENCODER_SCHEMA_VERSION = 1
ENCODER_COMPLETE_REVOLUTIONS = 10
ENCODER_CALIB_SUBPATH = os.path.join(
    "LATRIB", "calibrations", "encoder_external_ch3.json"
)

# Nominal gain table (index in the protocol = position in this list)
GAIN_VALUES = (1, 3, 10, 30, 100, 300, 1000, 3000)

CSV_HEADER = "idx,t_qpc,t_s,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,atrito,err\n"
ENCODER_CSV_HEADER = (
    "idx,t_qpc,t_s,angle_deg,accepted,status,health,unwrapped_deg,relative_deg,"
    "relative_mm,progress_mm,speed_mm_s,disk_rpm,direction,stationary,reversal,"
    "extreme_relative_mm,accepted_age_s\n"
)

# --------------------------------------------------------------------------
# Protocol (packed little-endian structs, same layout as the C #pragma pack(1))
# --------------------------------------------------------------------------
OP_ACQSTOP = 0x1006
OP_ACQSETUP = 0x1000
OP_ACQSTART = 0x1002
OP_ACQDATA = 0x100B
OP_SETCHCFG = 0x2002

_HDR = struct.Struct("<HH")  # PktHdr
_ACQ_SETUP = struct.Struct("<HHfHHHHhh8h")  # PktAcqSetup
_DATA_HDR = struct.Struct("<HHiifhhiii")  # PktData up to (not incl.) smp[]
_SET_CH = struct.Struct("<HHH6hHHH")  # PktSetCh
_DATA_MAX_SMP = 720
PKT_DATA_HEADER_SIZE = _DATA_HDR.size  # == offsetof(PktData, smp) == 32
PKT_DATA_SIZE = PKT_DATA_HEADER_SIZE + 2 * _DATA_MAX_SMP

Sample = Tuple[int, ...]  # DEFAULT_CHANNELS int16 values


@dataclass
class SensorsConfig:
    ip_address: str = "192.168.1.100"
    port: int = 41401
    bind_ip: str = ""  # "" = 0.0.0.0
    bind_port: int = 41402

    @classmethod
    def from_config(cls, src) :
        return cls(
            ip_address=src["SENSORS_IP"],
            port=src["SENSORS_PORT"],
            bind_ip=src["SENSORS_BIND_IP"],
            bind_port=src["SENSORS_BIND_PORT"],
            # parity/stop_bits/byte_size stay at defaults
        )


class Sensors:
    """Wrapper class around the DLG400 UDP connection."""

    def __init__(self, config: SensorsConfig):
        self.addr = (config.ip_address, config.port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        # socket.inet_pton(socket.AF_INET, bind_ip or "0.0.0.0")
        # socket.inet_pton(socket.AF_INET, dlg_ip)
        self.sock.bind((config.bind_ip or "0.0.0.0", config.bind_port))

        # configure channels
        chan_confs = []
        for ch in range(DEFAULT_CHANNELS):
            chan_confs.append(load_calib_for_channel(ch + 1) or ChannelConfig())
        enc_ch = chan_confs[ENCODER_CHANNEL - 1]
        for channel in range(1, DEFAULT_CHANNELS + 1):
            ch_config: ChannelConfig = chan_confs[channel - 1]
            payload = _SET_CH.pack(
                OP_SETCHCFG,
                0,
                0,
                _i16(channel),
                _i16(ch_config.t_sensor),
                _i16(ch_config.i_gain_idx),
                _i16(ch_config.i_lpf),
                _i16(ch_config.i_sens_pwr),
                0,
                _u16(ch_config.input_dc),
                _u16(ch_config.input_ac),
                _u16(ch_config.use_balance),
            )
            self.sock.sendto(payload, self.addr)
        # sensors.acq_setup_multi(rate_hz, DEFAULT_CHANNELS)

        self.state = "READY"

    def __del__(self):
        self.sock.close()

    def set_nonblocking(self) -> None:
        self.sock.setblocking(False)

    def close(self) -> None:
        self.sock.close()

    def send_cmd(self, payload: bytes):
        sent = self.sock.sendto(payload, self.addr)

    def stop(self) -> None:
        payload = _HDR.pack(OP_ACQSTOP, 0)
        sent = self.sock.sendto(payload, self.addr)
        assert sent == len(payload), "SENSOR STOP MESSAGE PROBLEM"

    def start(self) -> None:
        payload = _HDR.pack(OP_ACQSTART, 0)
        sent = self.sock.sendto(payload, self.addr)
        print("sensor sending ok")
        assert sent == len(payload), "SENSOR START MESSAGE PROBLEM"

    def configure_channel(self, channel: int, config) -> None:
        payload = _SET_CH.pack(
            OP_SETCHCFG,
            0,
            0,
            _i16(channel),
            _i16(config.t_sensor),
            _i16(config.i_gain_idx),
            _i16(config.i_lpf),
            _i16(config.i_sens_pwr),
            0,
            _u16(config.input_dc),
            _u16(config.input_ac),
            _u16(config.use_balance),
        )
        sent = self.sock.sendto(payload, self.addr)
        assert sent == len(payload), "CHANNEL CONFIG MESSAGE PROBLEM"

    def acq_setup_multi(self, freq_idx: float, n_sig: int) -> None:
        icm = [0] * 8
        for i in range(min(n_sig, 8)):
            icm[i] = i + 1  # CH1..CH8
        self.send_cmd(
            _ACQ_SETUP.pack(OP_ACQSETUP, 0, freq_idx, 0, 0, 0, 0, 1, _i16(n_sig), *icm)
        )

    def read(self) -> bytes:  # NOTE: rewrite this to output a decent format
        data=bytes()
        try:
            data, _ = self.sock.recvfrom(PKT_DATA_SIZE)
        except OSError:
            pass
        print(f"sensor read ok: data{data}")
        return data

    def drain_packets_to_ring(self, ring, n_sig: int) -> bool:
        """Read every pending ACQDATA datagram into the ring. True if any data."""
        got = False
        while True:
            try:
                data, _ = self.sock.recvfrom(PKT_DATA_SIZE)
            except OSError:  # would-block or any other error: stop draining
                break
            n = len(data)
            if n < 4 or struct.unpack_from("<H", data)[0] != OP_ACQDATA:
                continue
            avail = (n - PKT_DATA_HEADER_SIZE) // 2
            if avail <= 0:
                continue
            pkt_nsig = _DATA_HDR.unpack_from(data)[6]
            stride = pkt_nsig if pkt_nsig > 0 else n_sig
            samples_per_ch = avail // stride
            if samples_per_ch <= 0:
                continue
            smp = struct.unpack_from("<%dh" % avail, data, PKT_DATA_HEADER_SIZE)
            got = True
            for sidx in range(samples_per_ch):
                base = sidx * stride
                ring.push(
                    tuple(
                        smp[base + ch] if base + ch < avail else 0
                        for ch in range(DEFAULT_CHANNELS)
                    )
                )
        return got

    def wait_first_samples(
        self, ring, n_sig: int, min_samples: int, timeout_ms: int
    ) -> Optional[int]:
        """Returns the QPC tick when ``min_samples`` are available, else None."""
        t0 = _tick_ms()
        while (_tick_ms() - t0) < timeout_ms:
            self.drain_packets_to_ring(ring, n_sig)
            if ring.count >= min_samples:
                return qpc_now_ticks()
            _sleep_1ms()
        return None


# --------------------------------------------------------------------------
# DLG UDP link
# --------------------------------------------------------------------------
class DlgLink:
    """UDP socket + DLG protocol commands."""

    def __init__(self, dlg_ip: str, dlg_port: int, bind_ip: str, bind_port: int):
        self.addr = (dlg_ip, dlg_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            socket.inet_pton(socket.AF_INET, bind_ip or "0.0.0.0")
            socket.inet_pton(socket.AF_INET, dlg_ip)
            self.sock.bind((bind_ip or "0.0.0.0", bind_port))
        except OSError:
            self.sock.close()
            raise

    def __del__(self):
        self.sock.close()

    def set_nonblocking(self) -> None:
        self.sock.setblocking(False)

    def close(self) -> None:
        self.sock.close()

    def send_cmd(self, payload: bytes) -> None:
        try:
            self.sock.sendto(payload, self.addr)
        except OSError:
            pass  # C ignored sendto() errors as well

    def stop_acq(self) -> None:
        self.send_cmd(_HDR.pack(OP_ACQSTOP, 0))

    def acq_start(self) -> None:
        self.send_cmd(_HDR.pack(OP_ACQSTART, 0))

    def configure_channel(self, ch: int, c) -> None:
        self.send_cmd(
            _SET_CH.pack(
                OP_SETCHCFG,
                0,
                0,
                _i16(ch),
                _i16(c.t_sensor),
                _i16(c.i_gain_idx),
                _i16(c.i_lpf),
                _i16(c.i_sens_pwr),
                0,
                _u16(c.input_dc),
                _u16(c.input_ac),
                _u16(c.use_balance),
            )
        )

    def acq_setup_multi(self, freq_idx: float, n_sig: int) -> None:
        icm = [0] * 8
        for i in range(min(n_sig, 8)):
            icm[i] = i + 1  # CH1..CH8
        self.send_cmd(
            _ACQ_SETUP.pack(OP_ACQSETUP, 0, freq_idx, 0, 0, 0, 0, 1, _i16(n_sig), *icm)
        )


    def wait_first_samples(
        self, ring, n_sig: int, min_samples: int, timeout_ms: int
    ) -> Optional[int]:
        """Returns the QPC tick when ``min_samples`` are available, else None."""
        t0 = _tick_ms()
        while (_tick_ms() - t0) < timeout_ms:
            self.drain_packets_to_ring(ring, n_sig)
            if ring.count >= min_samples:
                return qpc_now_ticks()
            _sleep_1ms()
        return None


def _i16(v: int) -> int:
    """C-style cast to int16_t."""
    return ((int(v) + 0x8000) & 0xFFFF) - 0x8000


def _u16(v: int) -> int:
    """C-style cast to uint16_t."""
    return int(v) & 0xFFFF


def qpc_now_ticks() -> int:
    return time.perf_counter_ns()


def qpc_freq_ticks() -> int:
    return 1_000_000_000


def _tick_ms() -> int:
    """Equivalent of GetTickCount() (monotonic, milliseconds)."""
    return int(time.monotonic() * 1000.0)


def _sleep_1ms() -> None:
    time.sleep(0.001)


@contextlib.contextmanager
def timer_resolution(ms: int = 1):
    """timeBeginPeriod / timeEndPeriod on Windows; no-op elsewhere."""
    winmm = None
    if sys.platform == "win32":
        try:
            winmm = ctypes.WinDLL("winmm")
            winmm.timeBeginPeriod(ms)
        except OSError:
            winmm = None
    try:
        yield
    finally:
        if winmm is not None:
            winmm.timeEndPeriod(ms)


# --------------------------------------------------------------------------
# Ring buffer (drop-newest on overrun, like the C version)
# --------------------------------------------------------------------------
class Ring:
    """Fixed capacity FIFO that decouples UDP reception from the slot loop."""

    def __init__(self, cap: int):
        self.cap = cap
        self._buf: List[Optional[Sample]] = [None] * cap
        self._head = 0
        self._tail = 0
        self.count = 0
        self.overrun = 0

    def push(self, s: Sample) -> None:
        if self.count >= self.cap:
            self.overrun += 1
            return  # drop newest
        self._buf[self._head] = s
        self._head = (self._head + 1) % self.cap
        self.count += 1

    def pop(self) -> Optional[Sample]:
        if self.count <= 0:
            return None
        s = self._buf[self._tail]
        self._tail = (self._tail + 1) % self.cap
        self.count -= 1
        return s

    def clear(self) -> None:
        self._head = self._tail = self.count = 0

    def free(self) -> None:
        self._buf = []
        self.cap = self._head = self._tail = self.count = 0


# --------------------------------------------------------------------------
# Calibration (text based parsing, identical semantics to the C helpers)
# --------------------------------------------------------------------------
_INT_RE = re.compile(r"[+-]?[0-9]+")
_FLOAT_RE = re.compile(
    r"[+-]?(?:(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|inf(?:inity)?|nan)",
    re.IGNORECASE,
)
_WS = " \t\n\v\f\r"


@dataclass
class ChannelConfig:
    slope: float = 1.0
    intercept: float = 0.0
    t_sensor: int = DEFAULT_TSENSOR
    i_lpf: int = DEFAULT_ILPF
    i_gain_idx: int = DEFAULT_IGAIN_IDX
    i_sens_pwr: int = DEFAULT_SENSPWR_IDX
    input_dc: int = DEFAULT_INPUT_DC
    input_ac: int = DEFAULT_INPUT_AC
    use_balance: int = DEFAULT_USE_BALANCE
    normalize_degrees: bool = False
    calibrated: bool = False


def read_text_file(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    # C strings stop at the first NUL
    return data.decode("latin-1").split("\0", 1)[0]


def channel_in_header(buf: str, ch: int) -> bool:
    p = buf.find('"channels"')
    if p < 0:
        return False
    p = buf.find("[", p)
    if p < 0:
        return False
    p += 1
    n = len(buf)
    while p < n and buf[p] != "]":
        while p < n and buf[p] not in "0123456789-]":
            p += 1
        if p >= n or buf[p] == "]":
            break
        m = _INT_RE.match(buf, p)
        if m:
            if int(m.group()) == ch:
                return True
            p = m.end()
        else:
            p += 1
    return False


def channel_matches(buf: str, ch: int) -> bool:
    return ('"channel": "CH%d"' % ch) in buf


def _value_after_key(buf: str, key: str) -> Optional[int]:
    p = buf.find(key)
    if p < 0:
        return None
    p = buf.find(":", p)
    if p < 0:
        return None
    p += 1
    while p < len(buf) and buf[p] in _WS:
        p += 1
    return p


def parse_fit_value(buf: str, key: str) -> Optional[float]:
    p = _value_after_key(buf, key)
    if p is None:
        return None
    m = _FLOAT_RE.match(buf, p)
    return float(m.group()) if m else None


def parse_int_value(buf: str, key: str) -> Optional[int]:
    p = _value_after_key(buf, key)
    if p is None:
        return None
    m = _INT_RE.match(buf, p)
    return int(m.group()) if m else None


def gain_index_from_value(value: int) -> int:
    try:
        return GAIN_VALUES.index(value)
    except ValueError:
        return -1


def get_exe_dir() -> str:
    """Directory of this module (the C code used the directory of the .exe)."""
    return os.path.dirname(os.path.abspath(__file__))


def get_encoder_calib_path() -> Optional[str]:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return None
    return os.path.join(base, ENCODER_CALIB_SUBPATH)


def load_encoder_angle_calibration() -> ChannelConfig | None:
    path = get_encoder_calib_path()
    if not path:
        return None
    buf = read_text_file(path)
    if buf is None:
        return None
    if (
        '"purpose": "encoder_ch3_angle_deg"' not in buf
        or '"unit": "deg"' not in buf
        or '"accepted": true' not in buf
        or not channel_in_header(buf, ENCODER_CHANNEL)
    ):
        return None
    if parse_int_value(buf, '"schema_version"') != ENCODER_SCHEMA_VERSION:
        return None
    if parse_int_value(buf, '"complete_revolutions"') != ENCODER_COMPLETE_REVOLUTIONS:
        return None
    s = parse_fit_value(buf, '"slope"')
    b = parse_fit_value(buf, '"intercept"')
    ints = [
        parse_int_value(buf, k)
        for k in (
            '"tSensor"',
            '"iLPF"',
            '"iGain"',
            '"iSensPwr"',
            '"fInputDCImp"',
            '"fInputACImp"',
        )
    ]
    if s is None or b is None or any(v is None for v in ints):
        return None
    if not math.isfinite(s) or not math.isfinite(b) or s == 0.0:
        return None
    ts, lpf, gain, power, dc, ac = ints
    return ChannelConfig(
        slope=s,
        intercept=b,
        t_sensor=ts,
        i_lpf=lpf,
        i_gain_idx=gain,
        i_sens_pwr=power,
        input_dc=dc,
        input_ac=ac,
        use_balance=0,
        normalize_degrees=True,
        calibrated=True,
    )


def load_calib_for_channel(ch: int) -> ChannelConfig | None:
    """Per-channel calibration; ``ch`` is 1-based.  None when not available."""
    if ch == ENCODER_CHANNEL:
        return load_encoder_angle_calibration()

    paths = [
        "calib.json",
        "calib",
        "calib_CH%d.json" % ch,
        os.path.join("out", "calib_CH%d.json" % ch),
    ]
    exe_dir = get_exe_dir()
    if exe_dir:
        paths += [
            os.path.join(exe_dir, "calib.json"),
            os.path.join(exe_dir, "calib"),
            os.path.join(exe_dir, "calib_CH%d.json" % ch),
            os.path.join(exe_dir, "out", "calib_CH%d.json" % ch),
        ]
    buf = None
    for p in paths:  # first readable file wins
        buf = read_text_file(p)
        if buf is not None:
            break
    if buf is None:
        return None
    if not channel_in_header(buf, ch) and not channel_matches(buf, ch):
        return None

    s = parse_fit_value(buf, '"slope"')
    b = parse_fit_value(buf, '"intercept"')
    ts = parse_int_value(buf, '"tSensor"')
    lpf = parse_int_value(buf, '"iLPF"')
    gain = parse_int_value(buf, '"iGain"')
    pwr = parse_int_value(buf, '"iSensPwr"')
    if gain is None:
        gain = DEFAULT_IGAIN_IDX
    elif gain > 7:
        mapped = gain_index_from_value(gain)
        gain = mapped if mapped >= 0 else DEFAULT_IGAIN_IDX
    if s is None or b is None:
        return None
    return ChannelConfig(
        slope=s,
        intercept=b,
        t_sensor=DEFAULT_TSENSOR if ts is None else ts,
        i_lpf=DEFAULT_ILPF if lpf is None else lpf,
        i_gain_idx=gain,
        i_sens_pwr=DEFAULT_SENSPWR_IDX if pwr is None else pwr,
        input_dc=DEFAULT_INPUT_DC,
        input_ac=DEFAULT_INPUT_AC,
        use_balance=DEFAULT_USE_BALANCE,
        normalize_degrees=False,
        calibrated=True,
    )


def ensure_out_dir_for_path(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass


# --------------------------------------------------------------------------
# CSV helpers
# --------------------------------------------------------------------------
def write_data_csv_row(
    f,
    idx: int,
    t_qpc: int,
    t_s: float,
    ch_buf: Sequence[str],
    has_atrito: bool,
    atrito: float,
) -> None:
    tail = ("%.6f,0" % atrito) if has_atrito else "NULL,0"
    f.write("%d,%d,%.6f,%s,%s\n" % (idx, t_qpc, t_s, ",".join(ch_buf[:8]), tail))


def write_null_csv_row(f, idx: int, t_qpc: int, t_s: float) -> None:
    f.write(
        "%d,%d,%.6f,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,1\n"
        % (idx, t_qpc, t_s)
    )


def calculate_decimation(full_rate_hz: float, compat_rate_hz: float) -> int:
    """Integer decimation factor, or 0 if the rates are not integer multiples."""
    if full_rate_hz <= 0.0 or compat_rate_hz <= 0.0 or compat_rate_hz > full_rate_hz:
        return 0
    ratio = full_rate_hz / compat_rate_hz
    decimation = int(ratio + 0.5)
    if decimation < 1 or abs(ratio - decimation) > 1.0e-9:
        return 0
    return decimation


# --------------------------------------------------------------------------
# Encoder plug-in interfaces (headers not supplied with the C file)
# --------------------------------------------------------------------------
class EncoderStateBackend(Protocol):
    """Equivalent of ``encoder_state.h`` (encoder_state_t).

    ``update_angle`` returns ``(status, output)`` where ``output`` exposes:
    accepted, initialized, health, unwrapped_deg, relative_deg, relative_mm,
    path_distance_mm, speed_mm_s, disk_rpm, direction, stationary,
    reversal_confirmed, extreme_relative_mm, accepted_age_s.
    """

    config: Any  # innovation_factor, innovation_margin_deg, innovation_min_gate_deg,
    # max_physical_speed_mm_s, stale_stop_s, failure_timeout_s
    SAMPLE_ACCEPTED: Any  # value of ``status`` meaning "accepted"
    HEALTH_FAILED: Any  # value of ``output.health`` meaning "failed"

    def update_angle(
        self, qpc: int, angle_deg: float, has_value: bool, saturated: bool
    ) -> Tuple[Any, Any]: ...
    def status_name(self, status: Any) -> str: ...
    def health_name(self, health: Any) -> str: ...
    def get_quality(self) -> Any:
        ...  # input_samples, accepted_samples, missing_samples,
        # quarantine_samples, saturated_samples,
        # non_monotonic_samples, impossible_jump_samples,
        # reversals, max_accepted_gap_s


class EncoderControlCodec(Protocol):
    """Equivalent of ``encoder_control_protocol.h``."""

    version: int
    size: int

    def build(
        self, session: int, sequence: int, t_qpc: int, output: Any
    ) -> Optional[bytes]: ...


@dataclass
class AcquisitionStreamConfig:
    """Equivalent of ``dlg_encoder_stream_config_t`` (``dlg_encoder_stream.h``)."""

    acquisition_socket: socket.socket
    control_socket: Optional[socket.socket]
    control_address: Optional[Tuple[str, int]]
    data_file: Any
    encoder_file: Any
    compat_file: Any
    timing_file: Any
    compat_decimation: int
    total_rows: int
    use_ipc: bool
    rate_hz: float
    force_normal_n: float
    frequency: int
    session: int
    slope: List[float]
    intercept: List[float]
    calibrated: List[bool]
    normalize_degrees: List[bool]
    encoder: Any
    poll_command: Callable[[], int]
    log_event: Callable[[str], None]


EncoderStateFactory = Callable[..., EncoderStateBackend]
# factory(radius_mm, max_speed_mm_s, reciprocating, qpc_frequency,
#         stale_stop_s, failure_timeout_s)  -> backend   (None = backend default)


@dataclass
class EncoderPlugins:
    state_factory: Optional[EncoderStateFactory] = None
    control_codec: Optional[EncoderControlCodec] = None
    stream_runner: Optional[Callable[[AcquisitionStreamConfig], int]] = None
    interpolate_crossing_qpc: Optional[Callable[..., Optional[int]]] = None


def default_interpolate_crossing_qpc(
    q0: int, p0: float, q1: int, p1: float, target: float
) -> Optional[int]:
    """Linear interpolation of the QPC at which progress crosses ``target``.

    Reconstruction of ``encoder_interpolate_crossing_qpc`` (header not
    available); override through ``EncoderPlugins.interpolate_crossing_qpc``.
    """
    if q1 <= q0 or p1 <= p0 or not (p0 <= target <= p1):
        return None
    return int(q0 + (target - p0) / (p1 - p0) * (q1 - q0) + 0.5)


# --------------------------------------------------------------------------
# Continuous direction lock
# --------------------------------------------------------------------------
CONT_DIRECTION_MEDIAN_SAMPLES = 51
CONT_DIRECTION_MOTION_MM = 0.2
CONT_DIRECTION_LOCK_MM = 1.0
CONT_DIRECTION_CONFIRM_SAMPLES = 5
CONT_DIRECTION_LOCK_TIMEOUT_S = 3.0
CONT_DIRECTION_INCONSISTENT_MM = 1.0

CONT_DIRECTION_FAULT_NONE = 0
CONT_DIRECTION_FAULT_LOCK_TIMEOUT = 1
CONT_DIRECTION_FAULT_INCONSISTENT = 2


def continuous_direction_fault_name(fault: int) -> str:
    return {
        CONT_DIRECTION_FAULT_LOCK_TIMEOUT: "DIRECTION_LOCK_TIMEOUT",
        CONT_DIRECTION_FAULT_INCONSISTENT: "DIRECTION_INCONSISTENT",
    }.get(fault, "NONE")


class ContinuousDirectionLock:
    def __init__(self) -> None:
        self.locked_direction = 0
        self.candidate_direction = 0
        self.candidate_samples = 0
        self._recent = [0.0] * CONT_DIRECTION_MEDIAN_SAMPLES
        self.recent_count = 0
        self._recent_next = 0
        self.filtered_relative_mm = 0.0
        self.movement_qpc = 0
        self.max_progress_mm = 0.0
        self.fault = CONT_DIRECTION_FAULT_NONE

    def _median(self) -> float:
        if self.recent_count <= 0:
            return 0.0
        return sorted(self._recent[: self.recent_count])[self.recent_count // 2]

    def update(self, output: Any, qpc: int, qpc_frequency: int) -> int:
        if self.fault != CONT_DIRECTION_FAULT_NONE:
            return self.locked_direction
        if output.accepted:
            self._recent[self._recent_next] = output.relative_mm
            self._recent_next = (self._recent_next + 1) % CONT_DIRECTION_MEDIAN_SAMPLES
            if self.recent_count < CONT_DIRECTION_MEDIAN_SAMPLES:
                self.recent_count += 1
            self.filtered_relative_mm = self._median()

            if (
                self.movement_qpc == 0
                and abs(self.filtered_relative_mm) >= CONT_DIRECTION_MOTION_MM
            ):
                self.movement_qpc = qpc
            if self.locked_direction == 0:
                if abs(self.filtered_relative_mm) >= CONT_DIRECTION_LOCK_MM:
                    sign = 1 if self.filtered_relative_mm >= 0.0 else -1
                    if self.candidate_direction == sign:
                        self.candidate_samples += 1
                    else:
                        self.candidate_direction = sign
                        self.candidate_samples = 1
                    if self.candidate_samples >= CONT_DIRECTION_CONFIRM_SAMPLES:
                        self.locked_direction = sign
                else:
                    self.candidate_direction = 0
                    self.candidate_samples = 0
            if (
                self.locked_direction != 0
                and self.locked_direction * self.filtered_relative_mm
                <= -CONT_DIRECTION_INCONSISTENT_MM
            ):
                self.fault = CONT_DIRECTION_FAULT_INCONSISTENT
        if (
            self.locked_direction == 0
            and self.movement_qpc != 0
            and qpc_frequency > 0
            and (qpc - self.movement_qpc) / qpc_frequency
            >= CONT_DIRECTION_LOCK_TIMEOUT_S
        ):
            self.fault = CONT_DIRECTION_FAULT_LOCK_TIMEOUT
        return self.locked_direction

    def progress(self, output: Any) -> float:
        if self.locked_direction == 0:
            return 0.0
        if not output.accepted:
            return self.max_progress_mm
        directional = self.locked_direction * output.relative_mm
        if directional > self.max_progress_mm:
            self.max_progress_mm = directional
        if self.max_progress_mm < 0.0:
            self.max_progress_mm = 0.0
        return self.max_progress_mm


class _SimpleOutput:
    """Minimal stand-in for encoder_state_output_t used by the self tests."""

    def __init__(self, accepted: bool = True, relative_mm: float = 0.0):
        self.accepted = accepted
        self.relative_mm = relative_mm


def run_encoder_state_self_test(plugins: Optional[EncoderPlugins] = None) -> bool:
    frequency = 1_000_000
    radius_mm = 10.0
    speed_mm_s = 5.0
    target_mm = 10.0
    mm_per_degree = (2.0 * math.pi * radius_mm) / 360.0
    interp = (
        plugins.interpolate_crossing_qpc
        if plugins and plugins.interpolate_crossing_qpc
        else default_interpolate_crossing_qpc
    )

    # Noise around zero must neither lock nor fault.
    noise = ContinuousDirectionLock()
    for idx in range(100):
        noise.update(
            _SimpleOutput(True, 0.25 if idx % 2 == 0 else -0.25), idx * 5000, frequency
        )
    if noise.locked_direction != 0 or noise.fault != CONT_DIRECTION_FAULT_NONE:
        return False

    # Needs the real encoder state machine.
    if not plugins or not plugins.state_factory:
        return False
    state = plugins.state_factory(radius_mm, 10.0, False, frequency, None, None)
    if state is None:
        return False

    lock = ContinuousDirectionLock()
    progress_mm = 0.0
    crossing_qpc = 0
    previous_progress = 0.0
    previous_qpc = 0
    output = None
    for idx in range(401):
        t_s = idx / 200.0
        angle = math.fmod((speed_mm_s * t_s) / mm_per_degree, 360.0)
        qpc = int(t_s * frequency + 0.5)
        status, output = state.update_angle(qpc, angle, True, False)
        if status != state.SAMPLE_ACCEPTED:
            return False
        lock.update(output, qpc, frequency)
        progress_mm = lock.progress(output)
        if previous_progress < target_mm <= progress_mm:
            c = interp(previous_qpc, previous_progress, qpc, progress_mm, target_mm)
            if c is None:
                return False
            crossing_qpc = c
        previous_progress = progress_mm
        previous_qpc = qpc
    if (
        lock.locked_direction != 1
        or lock.fault != CONT_DIRECTION_FAULT_NONE
        or abs(progress_mm - target_mm) >= 1.0e-9
        or crossing_qpc != 2_000_000
        or abs(output.speed_mm_s - speed_mm_s) >= 1.0e-6
    ):
        return False

    # Slow drift never reaches the lock threshold -> LOCK_TIMEOUT.
    timeout_lock = ContinuousDirectionLock()
    out = _SimpleOutput(True, 0.25)
    for idx in range(651):
        timeout_lock.update(out, idx * 5000, frequency)
    if timeout_lock.fault != CONT_DIRECTION_FAULT_LOCK_TIMEOUT:
        return False

    # Locked in one direction, then a sustained opposite excursion -> INCONSISTENT.
    inc = ContinuousDirectionLock()
    inc.locked_direction = 1
    out = _SimpleOutput(True, -2.0)
    for idx in range(CONT_DIRECTION_MEDIAN_SAMPLES):
        inc.update(out, idx * 5000, frequency)
    return inc.fault == CONT_DIRECTION_FAULT_INCONSISTENT


# --------------------------------------------------------------------------
# Configuration / CLI
# --------------------------------------------------------------------------
USAGE = """Uso:
  dlg_logger_ipc --out <csv> --duration <s> [--rate <hz>] [--ipc]
                [--ip <dlg_ip>] [--port <dlg_port>]
                [--bind-ip <ip>] [--bind-port <port>]
                [--force-normal <N>]
                [--compat-out <csv> --compat-rate <hz>]
                [--encoder-state-out <csv> --encoder-radius-mm <mm>]
                [--encoder-max-speed-mm-s <mm/s> --encoder-target-mm <mm>]
                [--encoder-control-port <localhost_udp> --encoder-session <u64>]
                [--encoder-acquisition-clock] (reciprocante IPC autoritativo)
                [--encoder-reciprocating-shadow]
                [--self-test-decimation]
                [--self-test-encoder-state]"""


def print_usage() -> None:
    print(USAGE)


@dataclass
class LoggerConfig:
    out_path: Optional[str] = None
    compat_out_path: Optional[str] = None
    encoder_state_out_path: Optional[str] = None
    dlg_ip: str = DLG_IP_DEFAULT
    bind_ip: str = LOCAL_BIND_IP_DEFAULT
    dlg_port: int = DLG_PORT_DEFAULT
    bind_port: int = LOCAL_BIND_PORT_DEFAULT
    rate_hz: float = DEFAULT_RATE_HZ
    duration_s: float = DEFAULT_DURATION_S
    force_normal_n: float = 0.0
    compat_rate_hz: float = 0.0
    encoder_radius_mm: float = 0.0
    encoder_max_speed_mm_s: float = 0.0
    encoder_target_mm: float = 0.0
    encoder_control_port: int = 0
    encoder_session_id: int = 0
    use_ipc: bool = False
    self_test_decimation: bool = False
    self_test_encoder_state: bool = False
    encoder_reciprocating_shadow: bool = False
    encoder_acquisition_clock: bool = False


def _atof(s: str) -> float:
    m = _FLOAT_RE.match(s.lstrip(_WS))
    return float(m.group()) if m else 0.0


def _atoi(s: str) -> int:
    m = _INT_RE.match(s.lstrip(_WS))
    return int(m.group()) if m else 0


def parse_args(argv: Sequence[str]) -> Optional[LoggerConfig]:
    """Parse the CLI like the C ``main``.  Returns None on an unknown argument."""
    c = LoggerConfig()
    n = len(argv)
    i = 0
    while i < n:
        a = argv[i]
        has_val = i + 1 < n
        if a == "--out" and has_val:
            i += 1
            c.out_path = argv[i]
        elif a == "--duration" and has_val:
            i += 1
            c.duration_s = _atof(argv[i])
        elif a == "--rate" and has_val:
            i += 1
            c.rate_hz = _atof(argv[i])
        elif a == "--ip" and has_val:
            i += 1
            c.dlg_ip = argv[i]
        elif a == "--port" and has_val:
            i += 1
            c.dlg_port = _u16(_atoi(argv[i]))
        elif a == "--bind-ip" and has_val:
            i += 1
            c.bind_ip = argv[i]
        elif a == "--bind-port" and has_val:
            i += 1
            c.bind_port = _u16(_atoi(argv[i]))
        elif a == "--force-normal" and has_val:
            i += 1
            c.force_normal_n = _atof(argv[i])
        elif a == "--compat-out" and has_val:
            i += 1
            c.compat_out_path = argv[i]
        elif a == "--compat-rate" and has_val:
            i += 1
            c.compat_rate_hz = _atof(argv[i])
        elif a == "--encoder-state-out" and has_val:
            i += 1
            c.encoder_state_out_path = argv[i]
        elif a == "--encoder-radius-mm" and has_val:
            i += 1
            c.encoder_radius_mm = _atof(argv[i])
        elif a == "--encoder-max-speed-mm-s" and has_val:
            i += 1
            c.encoder_max_speed_mm_s = _atof(argv[i])
        elif a == "--encoder-target-mm" and has_val:
            i += 1
            c.encoder_target_mm = _atof(argv[i])
        elif a == "--encoder-control-port" and has_val:
            i += 1
            c.encoder_control_port = _u16(_atoi(argv[i]))
        elif a == "--encoder-session" and has_val:
            i += 1
            c.encoder_session_id = _atoi(argv[i]) & 0xFFFFFFFFFFFFFFFF
        elif a == "--encoder-reciprocating-shadow":
            c.encoder_reciprocating_shadow = True
        elif a == "--encoder-acquisition-clock":
            c.encoder_acquisition_clock = True
        elif a == "--self-test-decimation":
            c.self_test_decimation = True
        elif a == "--self-test-encoder-state":
            c.self_test_encoder_state = True
        elif a == "--ipc":
            c.use_ipc = True
        else:
            return None
        i += 1
    return c


def _same_path(a: Optional[str], b: Optional[str]) -> bool:
    return a is not None and b is not None and a.lower() == b.lower()


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _out(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def _validate(c: LoggerConfig) -> bool:
    """Argument consistency checks (same conditions and messages as the C code)."""
    out_path = c.out_path
    if (
        not out_path
        or not math.isfinite(c.duration_s)
        or not math.isfinite(c.rate_hz)
        or c.duration_s <= 0
        or c.rate_hz <= 0
        or c.duration_s * c.rate_hz > 2**31 - 1
    ):
        print_usage()
        return False
    if (
        (c.compat_out_path and c.compat_rate_hz <= 0.0)
        or (not c.compat_out_path and c.compat_rate_hz > 0.0)
        or (
            c.compat_out_path and calculate_decimation(c.rate_hz, c.compat_rate_hz) <= 0
        )
        or (c.compat_out_path and _same_path(out_path, c.compat_out_path))
    ):
        _err(
            "Compatibilidade exige arquivo distinto e divisor inteiro da taxa principal."
        )
        return False
    if (
        (
            c.encoder_state_out_path
            and (
                c.encoder_radius_mm <= 0.0
                or c.encoder_max_speed_mm_s <= 0.0
                or (not c.encoder_reciprocating_shadow and c.encoder_target_mm <= 0.0)
            )
        )
        or (
            not c.encoder_state_out_path
            and (
                c.encoder_radius_mm > 0.0
                or c.encoder_max_speed_mm_s > 0.0
                or c.encoder_target_mm > 0.0
            )
        )
        or (
            c.encoder_state_out_path
            and (
                _same_path(out_path, c.encoder_state_out_path)
                or _same_path(c.compat_out_path, c.encoder_state_out_path)
            )
        )
    ):
        _err(
            "Estado do encoder exige arquivo distinto, raio, velocidade maxima e alvo positivos."
        )
        return False
    if c.encoder_reciprocating_shadow and (
        not c.encoder_state_out_path
        or c.encoder_control_port == 0
        or c.encoder_session_id == 0
    ):
        _err("Sombra reciprocante exige estado do encoder e enlace UDP local.")
        return False
    if c.encoder_acquisition_clock and (
        not c.encoder_reciprocating_shadow
        or not c.use_ipc
        or c.encoder_control_port == 0
        or c.encoder_session_id == 0
    ):
        _err("Relogio de aquisicao exige reciprocante, --ipc e enlace do encoder.")
        return False
    if (c.encoder_control_port == 0) != (c.encoder_session_id == 0) or (
        c.encoder_control_port != 0 and not c.encoder_state_out_path
    ):
        _err(
            "Enlace do encoder exige porta, sessao nao nula e estado do encoder ativo."
        )
        return False
    return True


def _open_csv(path: str):
    """Text file that is flushed on every row (the C code used _IONBF)."""
    return open(path, "w", buffering=1, encoding="utf-8")


if __name__ == "__main__":
    raise RuntimeError("DLG400 is a library, it cannot be run directly")
