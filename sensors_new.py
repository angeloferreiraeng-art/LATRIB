"""Cliente UDP para o datalogger DLG (DLG0221 / DLG4K), porta 41401.

Fluxo: configurar canais -> AcqSetup -> AcqStart -> ler pacotes AcqData -> AcqStop.
Todas as structs são little-endian e sem padding (#pragma pack(1)).
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional
from warnings import deprecated


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


def parse_pack(bytes: bytes, n_channels: int) -> list[list[int]]:
    """Returns a list of channel values (each channel has a block of raw data)"""
    if (
        len(bytes) < DATA_HEADER_SIZE
        or struct.unpack_from("<H", bytes)[0] != OP_ACQDATA
    ):
        return []
    h = _DATA_HDR.unpack_from(bytes)
    frame_no, ts, freq, n_bursts, n_sig = h[2], h[3], h[4], h[5], h[6]
    stride = n_channels
    avail = (len(bytes) - DATA_HEADER_SIZE) // 2
    bursts = avail // stride
    if n_bursts > 0:
        bursts = min(bursts, n_bursts)
    if bursts <= 0:
        return []
    flat = list(struct.unpack_from("<%dh" % (bursts * stride), bytes, DATA_HEADER_SIZE))
    data = [flat[b * stride : (b + 1) * stride] for b in range(bursts)]
    transposed = [list(row) for row in zip(*data)]
    assert len(transposed) == n_channels, (
        "ERROR: Transposed parsed frame has wrong size"
    )
    return transposed


# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------
@dataclass
class ChannelConfig:
    gain: float = 1.0
    offset: float = 0.0
    cutoff_freq: float = 0.0
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
    channel_configs: list[ChannelConfig] = field(default_factory=list)

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


@dataclass
class DataPacket:
    frame_no: int
    timestamp_ms: int
    sample_freq: float
    n_bursts: int
    n_signals: int
    samples: list[tuple[int, ...]]  # [burst][canal], na ordem de ICM


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
    _resonance_peak_offset = 1 - 1 / math.sqrt(2)

    def clear_buffer(self):
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def __init__(self, sample_rate, cutoff_frequency, resonance):
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

            w = 2 * math.pi * cutoff_frequency / sample_rate
            cosw = math.cos(w)
            alpha = math.sin(w) / (2 * q)

            b0 = (1 - cosw) / 2
            b1 = 1 - cosw
            b2 = (1 - cosw) / 2
            a0 = 1 + alpha
            a1 = -2 * cosw
            a2 = 1 - alpha
            self._set_coefficients(a0, a1, a2, b0, b1, b2)

    def process(self, block):
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
        self.flush()
        filters = []
        converters = []
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
            filters.append(filter)
            converter = Converter(cfg.offset, cfg.gain)
            converters.append(converter)
            self.filters = filters
            self.converters = converters
        self.real_rate_hz = self.acq_setup(
            self.rate_hz, self.channels, self.samples_per_pack
        )

    def __exit__(self) -> None:
        self.stop()
        self.sock.close()

    def __del__(self):
        self.stop()
        self.sock.close()

    # -- baixo nível -------------------------------------------------------
    def _send(self, payload: bytes) -> None:
        if self.sock is None:
            raise RuntimeError("socket fechado")
        sent = self.sock.sendto(payload, self.address)
        if sent != len(payload):
            raise RuntimeError("envio UDP incompleto")

    def _recv(self, timeout: float) -> Optional[bytes]:
        """Um datagrama vindo do DLG, ou None no timeout."""
        assert self.sock is not None
        self.sock.settimeout(max(timeout, 0.0001))
        while True:
            try:
                data, src = self.sock.recvfrom(RECV_BUFFER)
            except socket.timeout:
                return None
            except ConnectionResetError:  # Windows: ICMP "port unreachable"
                continue
            if src[0] != self.address[0]:
                continue
            return data

    def _transact(self, payload: bytes, resp_op: int) -> Optional[bytes]:
        """Envia comando e espera a resposta com o opcode esperado."""
        self._send(payload)
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("CONNECTION TIMED OUT.")
            data = self._recv(remaining)
            if data is None:
                break
            if len(data) < _HDR.size:
                continue
            if struct.unpack_from("<H", data)[0] != resp_op:
                continue  # pacote de dados atrasado, etc.
            if len(data) >= _RESP.size:
                err = _RESP.unpack_from(data)[2]
                if err != 0:
                    raise RuntimeError(f" aparelho retornou erro {err}")
            return data
        return None

    def acq_setup(
        self, rate_hz: float, channels: tuple[int, ...], n_bursts: int = 1
    ) -> float:
        """Configura a aquisição. Retorna a frequência real informada pelo aparelho."""
        icm = [0] * N_MAX_CHAN
        for i, ch in enumerate(channels[:N_MAX_CHAN]):
            icm[i] = _i16(ch)
        payload = _ACQ_SETUP.pack(
            OP_ACQSETUP,
            0,
            float(rate_hz),
            0,
            0,
            0,
            0,
            _i16(n_bursts),
            _i16(len(channels)),
            *icm,
        )
        resp = self._transact(payload, OP_ACQSETUP_R)
        if resp is not None and len(resp) >= _ACQ_SETUP_R.size:
            real = _ACQ_SETUP_R.unpack_from(resp)[3]
            if abs(real - rate_hz) > 1e-6 * max(rate_hz, 1.0):
                print("frequência pedida %.3f Hz, real %.3f Hz", rate_hz, real)
            return real
        return float(rate_hz)

    def start(self):
        self.flush()
        payload = _HDR.pack(OP_ACQSTART, 0)
        self.sock.sendto(payload, self.address)

    def stop(self) -> None:
        payload = _HDR.pack(OP_ACQSTOP, 0)
        self.sock.sendto(payload, self.address)

    # -- leitura -----------------------------------------------------------
    def flush(self) -> None:
        """Descarta tudo que já chegou no socket."""
        assert self.sock is not None
        self.sock.setblocking(False)
        try:
            while True:
                try:
                    self.sock.recvfrom(RECV_BUFFER)
                except (BlockingIOError, ConnectionResetError):
                    if True:
                        break
                except OSError:
                    break
        finally:
            self.sock.settimeout(self.timeout)

    def _parse(self, data: bytes) -> DataPacket | None:
        if (
            len(data) < DATA_HEADER_SIZE
            or struct.unpack_from("<H", data)[0] != OP_ACQDATA
        ):
            return None
        h = _DATA_HDR.unpack_from(data)
        frame_no, ts, freq, n_bursts, n_sig = h[2], h[3], h[4], h[5], h[6]
        stride = n_sig if n_sig > 0 else len(self.channels)
        avail = (len(data) - DATA_HEADER_SIZE) // 2
        bursts = avail // stride
        if n_bursts > 0:
            bursts = min(bursts, n_bursts)
        if bursts <= 0:
            return None
        flat = struct.unpack_from("<%dh" % (bursts * stride), data, DATA_HEADER_SIZE)
        rows = [tuple(flat[b * stride : (b + 1) * stride]) for b in range(bursts)]

        if self.last_frame is not None and frame_no > self.last_frame + 1:
            self.lost_packets += frame_no - self.last_frame - 1
        self.last_frame = frame_no
        return DataPacket(frame_no, ts, freq, n_bursts, stride, rows)

    def read(self) -> tuple[int] | None:
        """Reads the next data frame in a nice manner. Filters and adjust value to calibrated units."""
        try:
            raw, _ = self.sock.recvfrom(RECV_BUFFER)
            frames = parse_pack(raw, len(self.channels))
            out_data = []
            for f in frames:
                for block, filter in zip(f, self.filters):
                    block = filter.process(block)

        except socket.timeout:
            return None
        except ConnectionResetError:  # Windows: ICMP "port unreachable"
            return None
        return data

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

    def drain(self) -> list[DataPacket]:
        """Lê, sem bloquear, todos os pacotes de dados pendentes."""
        assert self.sock is not None
        out: list[DataPacket] = []
        self.sock.setblocking(False)
        try:
            while True:
                try:
                    data, src = self.sock.recvfrom(RECV_BUFFER)
                except ConnectionResetError:
                    continue
                except (BlockingIOError, OSError):
                    break
                if src[0] != self.address[0]:
                    continue
                pkt = self._parse(data)
                if pkt is not None:
                    out.append(pkt)
        finally:
            self.sock.settimeout(self.timeout)
        return out

    @deprecated("STREAM LOGIC WILL BE AT THE ORCHESTRATOR LEVEL")
    def stream(self, duration: Optional[float] = None) -> Iterator[DataPacket]:
        """Gerador de pacotes. Inicia a aquisição se necessário e para ao final."""
        if self.state != "RUNNING":
            self.start()
        t0 = time.monotonic()
        try:
            while duration is None or time.monotonic() - t0 < duration:
                pkt = self.read()
                if pkt is not None:
                    yield pkt
        finally:
            if self.state == "RUNNING":
                self.stop()
