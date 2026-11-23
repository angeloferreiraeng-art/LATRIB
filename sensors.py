"""Cliente UDP para o datalogger DLG (DLG0221 / DLG4K), porta 41401.

Fluxo: configurar canais -> AcqSetup -> AcqStart -> ler pacotes AcqData -> AcqStop.
Todas as structs são little-endian e sem padding (#pragma pack(1)).
"""

from __future__ import annotations

import socket
import struct
import time
import numpy as np
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Protocolo
# --------------------------------------------------------------------------
OP_ACQSETUP = 0x1000
OP_ACQSETUP_R = 0x1001
OP_ACQSTART = 0x1002
OP_ACQSTART_R = 0x1003
OP_ACQSTOP = 0x1006
OP_ACQSTOP_R = 0x1007
OP_ACQDATA = 0x100B
OP_SETCHCFG = 0x2002
OP_SETCHCFG_R = 0x2003

_HDR = struct.Struct("<HH")  # PacketCode, Reserved
_RESP = struct.Struct("<HHH")  # PacketCode, Reserved, Error
_ACQ_SETUP = struct.Struct("<HHfHHHHhh8h")
_ACQ_SETUP_R = struct.Struct("<HHHf")  # + Error, SampleFreq real
_SET_CH = struct.Struct("<HHH6hHHH")
_DATA_HDR = struct.Struct("<HHiifhhiii")  # até Samples[] (32 bytes)
DATA_HEADER_SIZE = _DATA_HDR.size
DATA_MAX_SAMPLES = 720
RECV_BUFFER = 2048  # maior que 32 + 2*720 = 1472

N_MAX_CHAN = 8
N_CHANNELS = 8


def _i16(v: int) -> int:
    return ((int(v) + 0x8000) & 0xFFFF) - 0x8000


def _u16(v: int) -> int:
    return int(v) & 0xFFFF


def parse_pack(
    bytes: bytes, n_channels: int
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Returns a list of channel values (each channel has a block of raw data)"""
    if (
        len(bytes) < DATA_HEADER_SIZE
        or struct.unpack_from("<H", bytes)[0] != OP_ACQDATA
    ):
        return np.array([[0]] * N_CHANNELS)
    h = _DATA_HDR.unpack_from(bytes)
    frame_no, ts, freq, n_bursts, n_sig = h[2], h[3], h[4], h[5], h[6]
    stride = n_channels
    avail = (len(bytes) - DATA_HEADER_SIZE) // 2
    bursts = avail // stride
    if n_bursts > 0:
        bursts = min(bursts, n_bursts)
    if bursts <= 0:
        return np.array([[0]] * N_CHANNELS)
    flat = list(struct.unpack_from("<%dh" % (bursts * stride), bytes, DATA_HEADER_SIZE))
    data = [flat[b * stride : (b + 1) * stride] for b in range(bursts)]
    transposed = [list(row) for row in zip(*data)]
    assert len(transposed) == n_channels, (
        "ERROR: Transposed parsed frame has wrong size"
    )
    return np.array(transposed)


# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------
@dataclass
class ChannelConfig:
    gain: float = 1.0
    offset: float = 0.0
    cutoff_freq: float = 0.1
    resonance: float = 0.7071
    t_sensor: int = (
        4  # 0 tensão, 1 corrente, 2 ponte completa, 3 1/4, 4 1/2 ponte, 5 ICP...
    )
    i_lpf: int = 0
    i_gain_idx: int = 5  # índice em GAIN_VALUES (5 -> 300)
    i_sens_pwr: int = 2  # índice da tensão de excitação
    input_dc: int = 0  # 1: 100K, 0: flutuante
    input_ac: int = 0
    use_balance: int = 0
    balance: int = 0

    @classmethod
    def strain_gauge(cls):
        return cls(
            gain=1.0,
            offset=0.0,
            cutoff_freq=0.1,
            resonance=0.7071,
            t_sensor=(
                4  # 0 tensão, 1 corrente, 2 ponte completa, 3 1/4, 4 1/2 ponte, 5 ICP...
            ),
            i_lpf=0,
            i_gain_idx=5,  # índice em GAIN_VALUES (5 -> 300)
            i_sens_pwr=2,  # índice da tensão de excitação
            input_dc=0,  # 1: 100K, 0: flutuante
            input_ac=0,
            use_balance=0,
            balance=0,
        )

    @classmethod
    def encoder(cls):
        return cls(
            gain=1.0,
            offset=0.0,
            cutoff_freq=0.1,
            resonance=0.7071,
            t_sensor=(
                1  # 0 tensão, 1 corrente, 2 ponte completa, 3 1/4, 4 1/2 ponte, 5 ICP...
            ),
            i_lpf=0,
            i_gain_idx=0,  # índice em GAIN_VALUES (5 -> 300)
            i_sens_pwr=0,  # índice da tensão de excitação
            input_dc=0,  # 1: 100K, 0: flutuante
            input_ac=0,
            use_balance=0,
            balance=0,
        )


@dataclass
class SensorsConfig:
    """Config info for sensors struct. for some reason the channels seem to be 1 indexed,
    so IDK what config[0] is doing particularly."""

    ip_address: str = "192.168.1.100"
    port: int = 41401
    bind_ip: str = ""  # "" = 0.0.0.0
    bind_port: int = 41402
    rate_hz: float = 200.0
    n_bursts_pack: int = 1  # amostras por canal em cada pacote
    timeout_s: float = 1.0  # timeout de leitura / confirmação
    channel_configs: list[ChannelConfig] = field(
        default_factory=lambda: [
            ChannelConfig(),
            ChannelConfig.strain_gauge(),
            ChannelConfig(),
            ChannelConfig.encoder(),
            ChannelConfig(),
            ChannelConfig(),
            ChannelConfig(),
            ChannelConfig(),
        ]
    )

    @classmethod
    def from_config(cls, src: dict[str, str]) -> SensorsConfig:
        ch_confs = []
        for i in range(N_CHANNELS):
            conf = ChannelConfig(
                gain=float(src[f"SENSORS_CHANNEL{i}_GAIN"]),
                offset=float(src[f"SENSORS_CHANNEL{i}_OFFSET"]),
                cutoff_freq=float(src[f"SENSORS_CHANNEL{i}_CUTOFF"]),
                resonance=float(src[f"SENSORS_CHANNEL{i}_RESONANCE"]),
                t_sensor=int(src[f"SENSORS_CHANNEL{i}_TYPE"]),
                i_lpf=int(src[f"SENSORS_CHANNEL{i}_LPF"]),
                i_gain_idx=int(src[f"SENSORS_CHANNEL{i}_GAIN_IDX"]),
                i_sens_pwr=int(src[f"SENSORS_CHANNEL{i}_SENS_PWR"]),
                input_dc=int(src[f"SENSORS_CHANNEL{i}_INPUT_DC"]),
                input_ac=int(src[f"SENSORS_CHANNEL{i}_INPUT_AC"]),
                use_balance=int(src[f"SENSORS_CHANNEL{i}_USE_BALANCE"]),
                balance=int(src[f"SENSORS_CHANNEL{i}_BALANCE"]),
            )
            ch_confs.append(conf)
        return cls(
            src["SENSORS_IP"],
            int(src["SENSORS_PORT"]),
            src["SENSORS_BIND_IP"],
            int(src["SENSORS_BIND_PORT"]),
            float(src["SENSORS_RATE"]),
            int(src["SENSORS_FRAME_PER_PACKET"]),
            float(src["SENSORS_TIMEOUT"]),
            ch_confs,
        )


@dataclass
class Converter:
    def __init__(self, offset: float, gain: float):
        self.offset = offset
        self.gain = gain

    def process(self, value):
        return value * self.gain + self.offset

    def process_block(self, block):
        for i in len(block):
            raw = block[i]
            block[i] = raw * self.gain + self.offset
        return block


@dataclass
class BiQuadFilter:
    _resonance_peak_offset = 1 - 1 / np.sqrt(2)

    def clear_buffer(self):
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def __init__(self, sample_rate, cutoff_frequency, resonance):
        self.cutoff = cutoff_frequency
        self.resonance = resonance
        self.a0 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.a3 = 0.0
        self.a4 = 0.0

        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

        if cutoff_frequency < 0.499 * sample_rate:
            # This equation gives the Q value which makes the desired resonance peak.
            # The error of the resultant peak height is less than 3%.
            q = resonance - self._resonance_peak_offset / (1 + 6 * (resonance - 1))

            w = 2 * np.pi * cutoff_frequency / sample_rate
            cosw = np.cos(w)
            alpha = np.sin(w) / (2 * q)

            b0 = (1 - cosw) / 2
            b1 = 1 - cosw
            b2 = (1 - cosw) / 2
            a0 = 1 + alpha
            a1 = -2 * cosw
            a2 = 1 - alpha
            self._set_coefficients(a0, a1, a2, b0, b1, b2)

    def set_unfiltered(self):
        self.a0 = 1.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.a3 = 0.0
        self.a4 = 0.0

        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def process(self, inp):
        a0, a1, a2, a3, a4 = self.a0, self.a1, self.a2, self.a3, self.a4
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        output = a0 * inp + a1 * x1 + a2 * x2 - a3 * y1 - a4 * y2

        x2 = x1
        x1 = inp
        y2 = y1
        y1 = output

        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return output

    def process_block(self, block):
        """Filters `block` in place (a list or NumPy array of floats)."""
        a0, a1, a2, a3, a4 = self.a0, self.a1, self.a2, self.a3, self.a4
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2

        for t in range(len(block)):
            inp = block[t]
            output = a0 * inp + a1 * x1 + a2 * x2 - a3 * y1 - a4 * y2

            x2 = x1
            x1 = inp
            y2 = y1
            y1 = output

            block[t] = output

        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return block

    def _set_coefficients(self, a0, a1, a2, b0, b1, b2):
        self.a0 = b0 / a0
        self.a1 = b1 / a0
        self.a2 = b2 / a0
        self.a3 = a1 / a0
        self.a4 = a2 / a0


class Sensors:
    """Wrapper da conexão UDP com o DLG."""

    def __init__(self, config: SensorsConfig):
        print(f"Connecting to Datalogger at IP {config.ip_address} PORT {config.port}")
        self.address = (config.ip_address, config.port)
        self.rate_hz = config.rate_hz
        self.samples_per_pack = config.n_bursts_pack
        self.timeout = config.timeout_s
        self.bind_point = (config.bind_ip, config.bind_port)
        self.channel_configs = config.channel_configs
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            sock.bind((config.bind_ip or "0.0.0.0", config.bind_port))
        except OSError:
            sock.close()
            raise
        sock.settimeout(self.timeout)
        self.sock: socket.socket = sock

        payload = _HDR.pack(OP_ACQSTOP, 0)
        sent = self.sock.sendto(payload, self.address)
        if sent != len(payload):
            raise RuntimeError("envio UDP incompleto")
        # self.flush()
        self.filters: list[BiQuadFilter] = []
        self.converters: list[Converter] = []
        self.types = []
        for i, cfg in enumerate(config.channel_configs):
            # print(i,cfg)
            # cfg = config.channel_configs[i]
            payload = _SET_CH.pack(
                OP_SETCHCFG,
                0,
                0,
                _i16(i),
                _i16(cfg.t_sensor),
                _i16(cfg.i_gain_idx),
                _i16(cfg.i_lpf),
                _i16(cfg.i_sens_pwr),
                _i16(cfg.balance),
                _u16(cfg.input_dc),
                _u16(cfg.input_ac),
                _u16(cfg.use_balance),
            )
            sent = self.sock.sendto(payload, self.address)
            assert sent == len(payload), (
                f"Failed to configure DataLogger CH-{i}, Config was truncated."
            )
            self.types.append(cfg.t_sensor)
            print("CHANNEL CFG:", cfg.cutoff_freq, cfg.offset, cfg.gain)
            filter = BiQuadFilter(self.rate_hz, cfg.cutoff_freq, cfg.resonance)
            # filter.set_unfiltered()
            self.filters.append(filter)
            converter = Converter(cfg.offset, cfg.gain)

            self.converters.append(converter)
        self.rate_hz = self.acq_setup(self.rate_hz, self.samples_per_pack)

    def __exit__(self) -> None:
        self.stop()
        self.sock.close()

    def __del__(self):
        self.stop()
        self.sock.close()

    def acq_setup_new(self, rate_hz: float, n_bursts: int = 1) -> float:
        """Configura a aquisição. Retorna a frequência real informada pelo aparelho."""
        icm = [0] * N_MAX_CHAN
        for i in range(N_CHANNELS):
            icm[i] = _i16(i)
        payload = _ACQ_SETUP.pack(
            OP_ACQSETUP,
            0,
            float(rate_hz),
            0,
            0,
            0,
            0,
            _i16(n_bursts),
            _i16(N_CHANNELS),
            *icm,
        )
        sent = self.sock.sendto(payload, self.address)
        assert sent == len(payload), (
            "Failed to send full setup payload to datalogger connection."
        )
        # Reads until the ACQSETUP_R shows up, skipping stray replies
        # (e.g. SETCHCFG_R from the channel configs) and leftover ACQDATA.
        deadline = time.time() + self.timeout
        while True:
            if time.time() > deadline:
                raise TimeoutError("sem resposta ao AcqSetup")
            try:
                resp, _ = self.sock.recvfrom(RECV_BUFFER)
            except socket.timeout:
                raise TimeoutError("sem resposta ao AcqSetup") from None
            except ConnectionResetError:  # Windows: ICMP "port unreachable"
                continue
            if len(resp) < _HDR.size:
                continue
            code = struct.unpack_from("<H", resp)[0]
            if code == OP_SETCHCFG_R and len(resp) >= _RESP.size:
                err = _RESP.unpack_from(resp)[2]
                if err != 0:
                    raise RuntimeError(f"SetChConfig falhou (erro {err})")
                continue
            if code != OP_ACQSETUP_R:
                continue
            assert len(resp) == _ACQ_SETUP_R.size, (
                f"resposta AcqSetup truncada, expect :{_ACQ_SETUP_R.size}, got {len(resp)}"
            )
            _, _, err, real = _ACQ_SETUP_R.unpack_from(resp)
            if err != 0:
                raise RuntimeError(f"AcqSetup falhou (erro {err})")
            if abs(real - rate_hz) > 1e-6 * max(rate_hz, 1.0):
                print(f"frequência pedida {rate_hz:.3f} Hz, real {real:.3f} Hz")
            return real

    def acq_setup(self, rate_hz: float, n_bursts: int = 1) -> float:
        """Configura a aquisição. Retorna a frequência real informada pelo aparelho."""
        icm = [0] * N_CHANNELS
        for i in range(N_CHANNELS):
            icm[i] = _i16(i)
        payload = _ACQ_SETUP.pack(
            OP_ACQSETUP,
            0,
            float(rate_hz),
            0,
            0,
            0,
            0,
            _i16(n_bursts),
            _i16(N_CHANNELS),
            *icm,
        )
        self.sock.sendto(payload, self.address)
        resp, _ = self.sock.recvfrom(RECV_BUFFER)
        if resp is not None and len(resp) >= _ACQ_SETUP_R.size:
            real = _ACQ_SETUP_R.unpack_from(resp)[3]
            if abs(real - rate_hz) > 1e-6 * max(rate_hz, 1.0):
                print("frequência pedida %.3f Hz, real %.3f Hz", rate_hz, real)
            return real
        return float(rate_hz)

    def calibrate_filter(
        self,
        idx: int,
        n_samples: int = 32,
        min_freq: float = 2.0,
        max_cutoff: float = 30.0,
    ):
        self.flush()

        # Collect n_samples frames, each containing n_channel values
        samples = np.array([self.read_raw() for _ in range(n_samples)], dtype=float)
        signal_data = samples[:, idx]

        # Detrend, window, and compute one-sided FFT magnitude
        detrended = signal_data - np.mean(signal_data)
        windowed = detrended * np.hanning(len(detrended))
        fft_vals = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(detrended), d=1.0 / self.rate_hz)

        # Focus on frequencies above min_freq to ignore near-DC drift
        mask = freqs >= min_freq
        valid_freqs = freqs[mask]
        valid_mags = fft_vals[mask]

        lowest_noise_freq = None
        if len(valid_mags) >= 3:
            median_mag = np.median(valid_mags)
            mad = np.median(np.abs(valid_mags - median_mag))
            threshold = max(median_mag + 3.0 * mad, 0.15 * np.max(valid_mags))

            peaks = (
                np.where(
                    (valid_mags[1:-1] > valid_mags[:-2])
                    & (valid_mags[1:-1] >= valid_mags[2:])
                    & (valid_mags[1:-1] >= threshold)
                )[0]
                + 1
            )

            if len(peaks) > 0:
                lowest_noise_freq = valid_freqs[peaks[0]]
            else:
                lowest_noise_freq = valid_freqs[np.argmax(valid_mags)]

        raw_cutoff = (
            lowest_noise_freq * 0.8
            if lowest_noise_freq is not None
            else self.rate_hz / 10.0
        )

        # Clamp to safe BiQuad range and caller's max_cutoff
        upper_bound = min(max_cutoff, 0.498 * self.rate_hz)
        new_cutoff = float(np.clip(raw_cutoff, 0.5, upper_bound))

        print(
            f"Channel {idx}: lowest noise peak at {lowest_noise_freq} Hz -> "
            f"cutoff set to {new_cutoff:.2f} Hz"
        )
        new_cutoff = 15.0
        self.filters[idx] = BiQuadFilter(self.rate_hz, new_cutoff, 1.0)

    def start(self):
        # self.flush()
        payload = _HDR.pack(OP_ACQSTART, 0)
        # self.sock.setblocking(False)
        self.sock.sendto(payload, self.address)

    def stop(self) -> None:
        payload = _HDR.pack(OP_ACQSTOP, 0)
        self.sock.sendto(payload, self.address)

    def set_filter(self, idx: int, cutoff: float):
        assert idx <= N_CHANNELS, "SETTING FILTER OUT OF BOUNDS"
        new_filter = BiQuadFilter(self.rate_hz, cutoff, 1.0)
        self.filters[idx] = new_filter

    def set_converter(self, idx: int, gain: float, offset: float):
        assert idx <= N_CHANNELS, "SETTING CONVERTER OUT OF BOUNDS"
        print(f"adjusting channel {idx} curve to {gain}*x+{offset}")
        new_converter = Converter(offset, gain)
        self.converters[idx] = new_converter

    def read_old(self) -> np.ndarray:
        """Reads the next data frame in a nice manner. Filters and adjust value to calibrated units."""
        # print("reading")
        try:
            raw, _ = self.sock.recvfrom(RECV_BUFFER)
            blocks = parse_pack(raw, N_CHANNELS)
            out_data = np.zeros(N_CHANNELS)
            for i in range(N_CHANNELS):
                b = float(np.mean(blocks[i]))
                b = self.filters[i].process(b)
                b = self.converters[i].process(b)
                out_data[i] = b
        except socket.timeout:
            print("timedout")
            out_data = np.zeros(N_CHANNELS)
        except ConnectionResetError:  # Windows: ICMP "port unreachable"
            print("port unreachable")
            out_data = np.zeros(N_CHANNELS)
        # print(out_data)
        return out_data

    def read(self) -> np.ndarray:
        """Reads the next data frame in a nice manner. Filters and adjust value to calibrated units."""
        raw = self.read_raw()
        out_data = []  # np.zeros(N_CHANNELS)
        for i in range(N_CHANNELS):
            r = raw[i]
            f = self.filters[i]
            c = self.converters[i]
            val_f = f.process(r)
            val_c = c.process(val_f)
            out_data.append(val_c)
            if i == 3:
                print(f"raw {r}, filtered {val_f}, converter {val_c}")

        return np.array(out_data)

    def read_nice(self):
        """Reads the next data frame in a nice manner. Filters and adjust value to calibrated units."""
        raw = self.read_buffered()
        out_data = []  # np.zeros(N_CHANNELS)
        for i in range(N_CHANNELS):
            r = raw[i]
            f = self.filters[i]
            c = self.converters[i]
            val_f = f.process(r)
            val_c = c.process(val_f)
            out_data.append(val_c)
        #   if i == 3:
        #        print(f"raw {r}, filtered {val_f}, converter {val_c}")

        return np.array(out_data)

    def read_raw(self):
        """Reads until there is no errors, garantees a value"""
        while True:
            try:
                return self.read_unchecked()
            except Exception as e:
                print(f"Reading exception :{e}")
                pass

    def read_unchecked(self) -> np.ndarray:
        """Reads the next data frame raw, shits the bed on timeout."""
        raw, _ = self.sock.recvfrom(RECV_BUFFER)
        blocks = parse_pack(raw, N_CHANNELS)
        out_data = np.nan * np.ones(N_CHANNELS)
        for i in range(N_CHANNELS):
            b = float(np.mean(blocks[i]))
            out_data[i] = b
        return out_data

    def read_buffered(self, b_size: int = 16):
        out = np.zeros((N_CHANNELS, b_size))
        for i in range(b_size):
            frame = self.read_raw()
            out[:, i] = frame
        mean = np.mean(out, axis=1)
        return mean

    def flush(self):
        """Discard every datagram currently queued in the socket."""
        self.sock.setblocking(False)
        try:
            while True:
                self.sock.recvfrom(RECV_BUFFER)
        except (BlockingIOError, ConnectionResetError):
            pass
        finally:
            self.sock.settimeout(self.timeout)

    def register_configs(self, dst: dict[str, str]):
        """Pushes the current configs to the output table for calibration record."""
        for i in range(N_CHANNELS):
            cfg = self.channel_configs[i]
            # values that can change at runtime come from the live objects
            dst[f"SENSORS_CHANNEL{i}_GAIN"] = str(self.converters[i].gain)
            dst[f"SENSORS_CHANNEL{i}_OFFSET"] = str(self.converters[i].offset)
            dst[f"SENSORS_CHANNEL{i}_CUTOFF"] = str(self.filters[i].cutoff)
            dst[f"SENSORS_CHANNEL{i}_RESONANCE"] = str(self.filters[i].resonance)
            dst[f"SENSORS_CHANNEL{i}_TYPE"] = str(self.types[i])
            # static values come from the stored config
            dst[f"SENSORS_CHANNEL{i}_LPF"] = str(cfg.i_lpf)
            dst[f"SENSORS_CHANNEL{i}_GAIN_IDX"] = str(cfg.i_gain_idx)
            dst[f"SENSORS_CHANNEL{i}_SENS_PWR"] = str(cfg.i_sens_pwr)
            dst[f"SENSORS_CHANNEL{i}_INPUT_DC"] = str(cfg.input_dc)
            dst[f"SENSORS_CHANNEL{i}_INPUT_AC"] = str(cfg.input_ac)
            dst[f"SENSORS_CHANNEL{i}_USE_BALANCE"] = str(cfg.use_balance)
            dst[f"SENSORS_CHANNEL{i}_BALANCE"] = str(cfg.balance)

        dst["SENSORS_IP"] = str(self.address[0])
        dst["SENSORS_PORT"] = str(self.address[1])
        dst["SENSORS_BIND_IP"] = str(self.bind_point[0])
        dst["SENSORS_BIND_PORT"] = str(self.bind_point[1])
        dst["SENSORS_RATE"] = str(self.rate_hz)
        dst["SENSORS_FRAME_PER_PACKET"] = str(self.samples_per_pack)
        dst["SENSORS_TIMEOUT"] = str(self.timeout)
