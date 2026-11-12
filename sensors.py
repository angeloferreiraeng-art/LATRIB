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
from typing import Any, Iterator, Optional



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


def parse_pack(bytes: bytes, n_channels: int) -> np.ndarray[int]:
    """Returns a list of channel values (each channel has a block of raw data)"""
    if (
        len(bytes) < DATA_HEADER_SIZE
        or struct.unpack_from("<H", bytes)[0] != OP_ACQDATA
    ):
        return np.array([[0]]*N_CHANNELS)
    h = _DATA_HDR.unpack_from(bytes)
    frame_no, ts, freq, n_bursts, n_sig = h[2], h[3], h[4], h[5], h[6]
    stride = n_channels
    avail = (len(bytes) - DATA_HEADER_SIZE) // 2
    bursts = avail // stride
    if n_bursts > 0:
        bursts = min(bursts, n_bursts)
    if bursts <= 0:
        return np.array([[0]]*N_CHANNELS)
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
    cutoff_freq: float = 1000.0
    resonance: float = 0.0
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


@dataclass
class SensorsConfig:
    ip_address: str = "192.168.1.100"
    port: int = 41401
    bind_ip: str = ""  # "" = 0.0.0.0
    bind_port: int = 41402
    rate_hz: float = 200.0
    n_bursts_pack: int = 1  # amostras por canal em cada pacote
    timeout_s: float = 1.0  # timeout de leitura / confirmação
    channel_configs: list[ChannelConfig] = field(default_factory=lambda: [ChannelConfig()]*N_CHANNELS)

    @classmethod
    def from_config(cls, src: dict[str, Any]) -> SensorsConfig:
        ch_confs = []
        for i in range(N_CHANNELS):
            ch_cutoff = float(src[f"SENSORS_CHANNEL{i}_CUTOFF"])
            ch_resonance = float(src[f"SENSORS_CHANNEL{i}_RESONANCE"])
            ch_gain = float(src[f"SENSORS_CHANNEL{i}_GAIN"])
            ch_offset = float(src[f"SENSORS_CHANNEL{i}_OFFSET"])
            conf = ChannelConfig(ch_gain, ch_offset, ch_cutoff, ch_resonance)
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

class Converter:
    def __init__(self, offset: float, gain: float):
        self.offset = offset
        self.gain = gain

    def process(self, block):
        for i in len(block):
            raw = block[i]
            block[i] = raw * self.gain + self.offset
        return block


class BiQuadFilter:
    _resonance_peak_offset = 1 - 1 / np.sqrt(2)

    def clear_buffer(self):
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def __init__(self, sample_rate, cutoff_frequency, resonance):
        self.cutoff=cutoff_frequency
        self.resonance=resonance
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
        self.address = (config.ip_address, config.port)
        self.rate_hz = config.rate_hz
        self.samples_per_pack = config.n_bursts_pack
        self.timeout = config.timeout_s
        self.bind_point = (config.bind_ip, config.bind_port)
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
        #self.flush()
        self.filters = []
        self.converters = []
        for i, cfg in enumerate(config.channel_configs):
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
            self.sock.sendto(payload, self.address)
            filter = BiQuadFilter(self.rate_hz, cfg.cutoff_freq, cfg.resonance)
            self.filters.append(filter)
            converter = Converter(cfg.offset, cfg.gain)
            self.converters.append(converter)
        self.real_rate_hz = self.acq_setup(
            self.rate_hz,  self.samples_per_pack
        )

    def __exit__(self) -> None:
        self.stop()
        self.sock.close()

    def __del__(self):
        self.stop()
        self.sock.close()

    def acq_setup(
        self, rate_hz: float, n_bursts: int = 1
    ) -> float:
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
        self.sock.sendto(payload,self.address)
        resp,_=self.sock.recvfrom(RECV_BUFFER)
        if resp is not None and len(resp) >= _ACQ_SETUP_R.size:
            real = _ACQ_SETUP_R.unpack_from(resp)[3]
            if abs(real - rate_hz) > 1e-6 * max(rate_hz, 1.0):
                print("frequência pedida %.3f Hz, real %.3f Hz", rate_hz, real)
            return real
        return float(rate_hz)

    def calibrate_converter(self, channel_idx: int,n_samples: int = 512):
        raw_values = []
        start_time = time.time()
        
        # Collect raw samples from incoming data packets
        while len(raw_values) < n_samples:
            if time.time() - start_time > self.timeout * max(1, n_samples // 10):
                break
            try:
                raw_packet, _ = self.sock.recvfrom(RECV_BUFFER)
                n_ch = len(self.channels) if hasattr(self, "channels") else N_CHANNELS
                frames = parse_pack(raw_packet, n_ch)
                if frames and len(frames) > channel_idx:
                    channel_block = frames[channel_idx]
                    if channel_block:
                        raw_values.extend(channel_block)
            except socket.timeout:
                continue

        if not raw_values:
            raise RuntimeError(f"Could not read data from channel {channel_idx} during calibration. Ensure acquisition is running.")

        # Average the raw readings to minimize noise
        mean_raw = float(np.mean(raw_values))
        
        # Get current converter for this channel
        converter = self.converters[channel_idx]
        
        # Single-point offset calibration formula:
        # target_value = (mean_raw * gain) + offset
        # Therefore: offset = target_value - (mean_raw * gain)
        converter.offset = target_value - (mean_raw * converter.gain)

    def calibrate_filter(self, channel_idx: int, n_samples: int = 512):
        """Automatically listens to the current signal on channel `channel_idx`, 
        analyzes noise and main frequencies via FFT, and updates the BiQuadFilter 
        cutoff frequency to filter out the noise and make the signal constant.
        
        Args:
            channel_idx: Index of the channel to calibrate.
            n_samples: Number of raw samples to capture for noise analysis.
            
        Returns:
            The newly calculated and applied cutoff frequency in Hz.
        """
        raw_values = []
        start_time = time.time()
        # 1. Listen to the current signal stream to capture noise
        while len(raw_values) < n_samples:
            if time.time() - start_time > self.timeout * max(1, n_samples // 50):
                break
            try:
                raw_packet, _ = self.sock.recvfrom(RECV_BUFFER)
                n_ch = len(self.channels) if hasattr(self, "channels") else N_CHANNELS
                frames = parse_pack(raw_packet, n_ch)
                if frames and len(frames) > channel_idx:
                    block = frames[channel_idx]
                    if block:
                        raw_values.extend(block)
            except socket.timeout:
                continue

        signal_data = np.array(raw_values, dtype=float)
        
        # 2. Analyze frequency spectrum using FFT (ignoring DC offset at index 0)
        detrended = signal_data - np.mean(signal_data)
        fft_vals = np.abs(np.fft.rfft(detrended))
        freqs = np.fft.rfftfreq(len(signal_data), d=1.0 / self.rate_hz)
        
        if len(freqs) > 1:
            peak_idx = np.argmax(fft_vals[1:]) + 1
            dominant_noise_freq = freqs[peak_idx]
        else:
            dominant_noise_freq = self.rate_hz / 4.0

        # 3. Determine optimal cutoff frequency below the dominant noise/main frequency
        # If mains hum or noise is detected, set cutoff below it to flatten the signal
        if dominant_noise_freq > 2.0:
            new_cutoff = dominant_noise_freq * 0.8
        else:
            new_cutoff = self.rate_hz / 10.0  # Fallback safety margin

        # Bound check according to BiQuadFilter constraints (< 0.499 * sample_rate)
        max_allowed_cutoff = 0.499 * self.rate_hz
        new_cutoff = max(0.5, min(new_cutoff, max_allowed_cutoff))

        new_filter = BiQuadFilter(self.rate_hz, new_cutoff, 1.0)
        self.filters[channel_idx] = new_filter

    def start(self):
        #self.flush()
        payload = _HDR.pack(OP_ACQSTART, 0)
        self.sock.sendto(payload, self.address)

    def stop(self) -> None:
        payload = _HDR.pack(OP_ACQSTOP, 0)
        self.sock.sendto(payload, self.address)



    def read(self) -> list[int] :
        """Reads the next data frame in a nice manner. Filters and adjust value to calibrated units."""
        try:
            raw, _ = self.sock.recvfrom(RECV_BUFFER)
            blocks = parse_pack(raw, N_CHANNELS)
            out_data = [0]*N_CHANNELS
            for i in range(8):
                b=float(np.mean(blocks[i]))
                #b = self.filters[i].process(b)
                out_data[i]=b
        except socket.timeout:
            out_data= [0]*N_CHANNELS
        except ConnectionResetError:  # Windows: ICMP "port unreachable"
            out_data= [0]*N_CHANNELS
        return out_data

    def register_configs(self, dst):
        """Pushes the current configs to the output table for calibration record."""
        for i, ch in enumerate(self.channel_configs):
            dst[f"SENSORS_CHANNEL{i}_CUTOFF"] = str(self.filters[i].cutoff)
            dst[f"SENSORS_CHANNEL{i}_RESONANCE"] = str(self.filters[i].resonance)
            dst[f"SENSORS_CHANNEL{i}_GAIN"] = str(self.converters[i].gain)
            dst[f"SENSORS_CHANNEL{i}_OFFSET"] = str(self.converters[i].offset)

        dst["SENSORS_IP"] = str(self.address[0])
        dst["SENSORS_PORT"] = str(self.address[1])
        dst["SENSORS_BIND_IP"] = str(self.bind_point[0])
        dst["SENSORS_BIND_PORT"] = str(self.bind_point[1])
        dst["SENSORS_RATE"] = str(self.rate_hz)
        dst["SENSORS_FRAME_PER_PACKET"] = str(self.samples_per_pack)
        dst["SENSORS_TIMEOUT"] = str(self.timeout)
