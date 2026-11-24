"""Cliente UDP para o datalogger DLG (DLG0221 / DLG4K), porta 41401.

Fluxo: configurar canais -> AcqSetup -> AcqStart -> ler pacotes AcqData -> AcqStop.
Todas as structs são little-endian e sem padding (#pragma pack(1)).
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Protocolo
# --------------------------------------------------------------------------
OP_ACQSETUP: int = 0x1000
OP_ACQSETUP_R: int = 0x1001
OP_ACQSTART: int = 0x1002
OP_ACQSTART_R: int = 0x1003
OP_ACQSTOP: int = 0x1006
OP_ACQSTOP_R: int = 0x1007
OP_ACQDATA: int = 0x100B
OP_SETCHCFG: int = 0x2002
OP_SETCHCFG_R: int = 0x2003

_HDR: struct.Struct = struct.Struct("<HH")  # PacketCode, Reserved
_RESP: struct.Struct = struct.Struct("<HHH")  # PacketCode, Reserved, Error
_ACQ_SETUP: struct.Struct = struct.Struct("<HHfHHHHhh8h")
_ACQ_SETUP_R: struct.Struct = struct.Struct("<HHHf")  # + Error, SampleFreq real
_SET_CH: struct.Struct = struct.Struct("<HHH6hHHH")
_DATA_HDR: struct.Struct = struct.Struct("<HHiifhhiii")  # até Samples[] (32 bytes)

DATA_HEADER_SIZE: int = _DATA_HDR.size
DATA_MAX_SAMPLES: int = 720
RECV_BUFFER: int = 2048  # maior que 32 + 2*720 = 1472

N_MAX_CHAN: int = 8
N_CHANNELS: int = 8


def _i16(v: int) -> int:
    """Converte um inteiro para o intervalo assinado de 16 bits (-32768..32767).

    Args:
        v: Valor inteiro a ser convertido.

    Returns:
        O valor reinterpretado como inteiro de 16 bits com sinal.
    """
    return ((int(v) + 0x8000) & 0xFFFF) - 0x8000


def _u16(v: int) -> int:
    """Converte um inteiro para o intervalo não assinado de 16 bits (0..65535).

    Args:
        v: Valor inteiro a ser convertido.

    Returns:
        O valor mascarado para 16 bits sem sinal.
    """
    return int(v) & 0xFFFF


def parse_pack(
    bytes: bytes, n_channels: int
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Interpreta um datagrama AcqData e retorna os valores por canal.

    Cada pacote contém um cabeçalho seguido de ``n_bursts`` amostras para cada
    um dos ``n_channels`` canais, intercaladas (burst-major). A função
    transpoõe os dados para que cada linha corresponda a um canal.

    Args:
        bytes: Datagrama UDP cru recebido do datalogger.
        n_channels: Número de canais presentes no pacote.

    Returns:
        Array NumPy de forma ``(n_channels, n_bursts)`` com as amostras
        inteiras convertidas para ``float64``. Se o pacote for inválido ou
        vazio, retorna um array de forma ``(N_CHANNELS, 1)`` preenchido com
        zeros.
    """
    if (
        len(bytes) < DATA_HEADER_SIZE
        or struct.unpack_from("<H", bytes)[0] != OP_ACQDATA
    ):
        return np.array([[0]] * N_CHANNELS)
    h = _DATA_HDR.unpack_from(bytes)
    frame_no: int = h[2]
    ts: int = h[3]
    freq: float = h[4]
    n_bursts: int = h[5]
    n_sig: int = h[6]
    stride: int = n_channels
    avail: int = (len(bytes) - DATA_HEADER_SIZE) // 2
    bursts: int = avail // stride
    if n_bursts > 0:
        bursts = min(bursts, n_bursts)
    if bursts <= 0:
        return np.array([[0]] * N_CHANNELS)
    flat: list[int] = list(
        struct.unpack_from("<%dh" % (bursts * stride), bytes, DATA_HEADER_SIZE)
    )
    data: list[list[int]] = [flat[b * stride : (b + 1) * stride] for b in range(bursts)]
    transposed: list[list[int]] = [list(row) for row in zip(*data)]
    assert len(transposed) == n_channels, (
        "ERROR: Transposed parsed frame has wrong size"
    )
    return np.array(transposed)


# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------
@dataclass
class ChannelConfig:
    """Configuração de um único canal de aquisição.

    Attributes:
        gain: Ganho aplicado na conversão de engenharia.
        offset: Offset aplicado na conversão de engenharia.
        cutoff_freq: Frequência de corte do filtro BiQuad (Hz).
        resonance: Ressonância (Q) do filtro BiQuad.
        t_sensor: Tipo de sensor (0=tensão, 1=corrente, 2=ponte completa,
            3=1/4 ponte, 4=1/2 ponte, 5=ICP, ...).
        i_lpf: Índice do filtro passa-baixa interno do hardware.
        i_gain_idx: Índice em ``GAIN_VALUES`` (5 -> 300).
        i_sens_pwr: Índice da tensão de excitação do sensor.
        input_dc: 1 para acoplamento DC (100K), 0 para flutuante.
        input_ac: 1 para acoplamento AC.
        use_balance: 1 para habilitar o balanceamento.
        balance: Valor de balanceamento.
    """

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
    def strain_gauge(cls) -> ChannelConfig:
        """Retorna uma configuração típica para extensômetro (1/2 ponte).

        Returns:
            Uma instância de ``ChannelConfig`` pré-preenchida para uso com
            extensômetros.
        """
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
    def encoder(cls) -> ChannelConfig:
        """Retorna uma configuração típica para encoder (sensor de corrente).

        Returns:
            Uma instância de ``ChannelConfig`` pré-preenchida para uso com
            encoders.
        """
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
    """Configuração completa da conexão e dos canais do datalogger.

    Attributes:
        ip_address: Endereço IP do datalogger.
        port: Porta UDP do datalogger.
        bind_ip: IP local para bind (``""`` = 0.0.0.0).
        bind_port: Porta local para bind.
        rate_hz: Frequência de amostragem desejada (Hz).
        n_bursts_pack: Amostras por canal em cada pacote.
        timeout_s: Timeout de leitura/confirmação (s).
        channel_configs: Lista com a configuração de cada canal.
    """

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
        """Constrói uma ``SensorsConfig`` a partir de um dicionário de strings.

        O dicionário deve conter chaves no formato
        ``SENSORS_CHANNEL{i}_{PARAM}`` para cada canal, além das chaves
        globais ``SENSORS_IP``, ``SENSORS_PORT``, etc.

        Args:
            src: Dicionário de configuração (valores como strings).

        Returns:
            Uma instância de ``SensorsConfig`` preenchida com os valores
            convertidos.
        """
        ch_confs: list[ChannelConfig] = []
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
    """Conversão linear de engenharia: ``y = x * gain + offset``.

    Attributes:
        offset: Offset da reta de conversão.
        gain: Ganho da reta de conversão.
    """

    def __init__(self, offset: float, gain: float) -> None:
        """Inicializa o conversor.

        Args:
            offset: Offset da reta de conversão.
            gain: Ganho da reta de conversão.
        """
        self.offset: float = offset
        self.gain: float = gain

    def process(self, value: float) -> float:
        """Aplica a conversão a um único valor.

        Args:
            value: Valor bruto de entrada.

        Returns:
            O valor convertido ``value * gain + offset``.
        """
        return value * self.gain + self.offset

    def process_block(self, block: list[float]) -> list[float]:
        """Aplica a conversão a um bloco de valores, in-place.

        Args:
            block: Lista de valores brutos.

        Returns:
            O mesmo bloco, com cada elemento convertido.
        """
        for i in range(len(block)):
            raw = block[i]
            block[i] = raw * self.gain + self.offset
        return block


@dataclass
class BiQuadFilter:
    """Filtro BiQuad de segunda ordem com resposta ressonante configurável.

    Attributes:
        cutoff: Frequência de corte (Hz).
        resonance: Ressonância (Q) desejada.
        a0, a1, a2, a3, a4: Coeficientes normalizados do filtro.
        x1, x2: Amostras de entrada anteriores.
        y1, y2: Amostras de saída anteriores.
    """

    _resonance_peak_offset: float = 1 - 1 / np.sqrt(2)

    def clear_buffer(self) -> None:
        """Zera o estado interno do filtro (histórico de entrada/saída)."""
        self.x1: float = 0.0
        self.x2: float = 0.0
        self.y1: float = 0.0
        self.y2: float = 0.0

    def __init__(
        self, sample_rate: float, cutoff_frequency: float, resonance: float
    ) -> None:
        """Inicializa o filtro e calcula os coeficientes.

        Args:
            sample_rate: Frequência de amostragem (Hz).
            cutoff_frequency: Frequência de corte desejada (Hz).
            resonance: Ressonância (Q) desejada.
        """
        self.cutoff: float = cutoff_frequency
        self.resonance: float = resonance
        self.a0: float = 0.0
        self.a1: float = 0.0
        self.a2: float = 0.0
        self.a3: float = 0.0
        self.a4: float = 0.0

        self.x1: float = 0.0
        self.x2: float = 0.0
        self.y1: float = 0.0
        self.y2: float = 0.0

        if cutoff_frequency < 0.499 * sample_rate:
            # This equation gives the Q value which makes the desired resonance peak.
            # The error of the resultant peak height is less than 3%.
            q: float = resonance - self._resonance_peak_offset / (
                1 + 6 * (resonance - 1)
            )

            w: float = 2 * np.pi * cutoff_frequency / sample_rate
            cosw: float = np.cos(w)
            alpha: float = np.sin(w) / (2 * q)

            b0: float = (1 - cosw) / 2
            b1: float = 1 - cosw
            b2: float = (1 - cosw) / 2
            a0: float = 1 + alpha
            a1: float = -2 * cosw
            a2: float = 1 - alpha
            self._set_coefficients(a0, a1, a2, b0, b1, b2)

    def set_unfiltered(self) -> None:
        """Configura o filtro para pass-through (sem filtragem)."""
        self.a0: float = 1.0
        self.a1: float = 0.0
        self.a2: float = 0.0
        self.a3: float = 0.0
        self.a4: float = 0.0

        self.x1: float = 0.0
        self.x2: float = 0.0
        self.y1: float = 0.0
        self.y2: float = 0.0

    def process(self, inp: float) -> float:
        """Processa uma única amostra através do filtro.

        Args:
            inp: Amostra de entrada.

        Returns:
            A amostra filtrada.
        """
        a0, a1, a2, a3, a4 = self.a0, self.a1, self.a2, self.a3, self.a4
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        output: float = a0 * inp + a1 * x1 + a2 * x2 - a3 * y1 - a4 * y2

        x2 = x1
        x1 = inp
        y2 = y1
        y1 = output

        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return output

    def process_block(
        self, block: list[float] | np.ndarray
    ) -> list[float] | np.ndarray:
        """Filtra um bloco de amostras in-place.

        Args:
            block: Lista ou array NumPy de floats.

        Returns:
            O mesmo bloco, com cada amostra filtrada.
        """
        a0, a1, a2, a3, a4 = self.a0, self.a1, self.a2, self.a3, self.a4
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2

        for t in range(len(block)):
            inp: float = block[t]
            output: float = a0 * inp + a1 * x1 + a2 * x2 - a3 * y1 - a4 * y2

            x2 = x1
            x1 = inp
            y2 = y1
            y1 = output

            block[t] = output

        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return block

    def _set_coefficients(
        self, a0: float, a1: float, a2: float, b0: float, b1: float, b2: float
    ) -> None:
        """Normaliza e armazena os coeficientes do filtro.

        Args:
            a0, a1, a2: Coeficientes do denominador (feedback).
            b0, b1, b2: Coeficientes do numerador (feedforward).
        """
        self.a0 = b0 / a0
        self.a1 = b1 / a0
        self.a2 = b2 / a0
        self.a3 = a1 / a0
        self.a4 = a2 / a0


class Sensors:
    """Wrapper da conexão UDP com o DLG.

    Gerencia a configuração dos canais, o setup da aquisição, a leitura de
    pacotes e o pós-processamento (filtragem e conversão de engenharia).

    Attributes:
        address: Tupla ``(ip, port)`` do datalogger.
        rate_hz: Frequência de amostragem efetiva (Hz).
        samples_per_pack: Amostras por canal em cada pacote.
        timeout: Timeout de socket (s).
        bind_point: Tupla ``(ip, port)`` local de bind.
        channel_configs: Configuração dos canais.
        sock: Socket UDP.
        filters: Filtros BiQuad por canal.
        converters: Conversores lineares por canal.
        types: Tipo de sensor por canal.
    """

    def __init__(self, config: SensorsConfig) -> None:
        """Inicializa a conexão, configura os canais e faz o AcqSetup.

        Args:
            config: Configuração completa da conexão e dos canais.

        Raises:
            OSError: Se o bind da porta local falhar.
            RuntimeError: Se o envio UDP for incompleto ou a configuração de
                algum canal for truncada.
        """
        print(f"Connecting to Datalogger at IP {config.ip_address} PORT {config.port}")
        self.address: tuple[str, int] = (config.ip_address, config.port)
        self.rate_hz: float = config.rate_hz
        self.samples_per_pack: int = config.n_bursts_pack
        self.timeout: float = config.timeout_s
        self.bind_point: tuple[str, int] = (config.bind_ip, config.bind_port)
        self.channel_configs: list[ChannelConfig] = config.channel_configs
        sock: socket.socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        try:
            sock.bind((config.bind_ip or "0.0.0.0", config.bind_port))
        except OSError:
            sock.close()
            raise
        sock.settimeout(self.timeout)
        self.sock: socket.socket = sock

        payload: bytes = _HDR.pack(OP_ACQSTOP, 0)
        sent: int = self.sock.sendto(payload, self.address)
        if sent != len(payload):
            raise RuntimeError("envio UDP incompleto")
        # self.flush()
        self.filters: list[BiQuadFilter] = []
        self.converters: list[Converter] = []
        self.types: list[int] = []
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
            filter: BiQuadFilter = BiQuadFilter(
                self.rate_hz, cfg.cutoff_freq, cfg.resonance
            )
            # filter.set_unfiltered()
            self.filters.append(filter)
            converter: Converter = Converter(cfg.offset, cfg.gain)

            self.converters.append(converter)
        self.rate_hz = self.acq_setup(self.rate_hz, self.samples_per_pack)

    def __exit__(self) -> None:
        """Para a aquisição e fecha o socket."""
        self.stop()
        self.sock.close()

    def __del__(self) -> None:
        """Para a aquisição e fecha o socket ao destruir o objeto."""
        self.stop()
        self.sock.close()

    def acq_setup_new(self, rate_hz: float, n_bursts: int = 1) -> float:
        """Configura a aquisição aguardando a resposta correta do aparelho.

        Descarta respostas espúrias (ex.: ``SETCHCFG_R`` ou ``ACQDATA``
        remanescentes) até receber um ``ACQSETUP_R``.

        Args:
            rate_hz: Frequência de amostragem desejada (Hz).
            n_bursts: Amostras por canal em cada pacote.

        Returns:
            A frequência real informada pelo aparelho (Hz).

        Raises:
            TimeoutError: Se nenhuma resposta chegar dentro do timeout.
            RuntimeError: Se o aparelho reportar erro em SetChConfig ou
                AcqSetup.
            AssertionError: Se a resposta AcqSetup estiver truncada.
        """
        icm: list[int] = [0] * N_MAX_CHAN
        for i in range(N_CHANNELS):
            icm[i] = _i16(i)
        payload: bytes = _ACQ_SETUP.pack(
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
        sent: int = self.sock.sendto(payload, self.address)
        assert sent == len(payload), (
            "Failed to send full setup payload to datalogger connection."
        )
        # Reads until the ACQSETUP_R shows up, skipping stray replies
        # (e.g. SETCHCFG_R from the channel configs) and leftover ACQDATA.
        deadline: float = time.time() + self.timeout
        while True:
            if time.time() > deadline:
                raise TimeoutError("sem resposta ao AcqSetup")
            try:
                resp: bytes
                resp, _ = self.sock.recvfrom(RECV_BUFFER)
            except socket.timeout:
                raise TimeoutError("sem resposta ao AcqSetup") from None
            except ConnectionResetError:  # Windows: ICMP "port unreachable"
                continue
            if len(resp) < _HDR.size:
                continue
            code: int = struct.unpack_from("<H", resp)[0]
            if code == OP_SETCHCFG_R and len(resp) >= _RESP.size:
                err: int = _RESP.unpack_from(resp)[2]
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
        """Configura a aquisição. Retorna a frequência real informada pelo aparelho.

        Versão simplificada que lê uma única resposta.

        Args:
            rate_hz: Frequência de amostragem desejada (Hz).
            n_bursts: Amostras por canal em cada pacote.

        Returns:
            A frequência real informada pelo aparelho (Hz), ou ``rate_hz``
            caso a resposta seja inválida.
        """
        icm: list[int] = [0] * N_CHANNELS
        for i in range(N_CHANNELS):
            icm[i] = _i16(i)
        payload: bytes = _ACQ_SETUP.pack(
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
        resp: bytes
        resp, _ = self.sock.recvfrom(RECV_BUFFER)
        if resp is not None and len(resp) >= _ACQ_SETUP_R.size:
            real: float = _ACQ_SETUP_R.unpack_from(resp)[3]
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
    ) -> None:
        """Estima a frequência de corte ideal para um canal a partir do ruído.

        Coleta amostras, calcula o espectro e escolhe um cutoff baseado no
        primeiro pico significativo acima de ``min_freq``. Ao final, o filtro
        do canal é substituído por um com cutoff fixo de 15 Hz e ressonância 1.

        Args:
            idx: Índice do canal a calibrar.
            n_samples: Número de amostras a coletar.
            min_freq: Frequência mínima (Hz) para ignorar deriva DC.
            max_cutoff: Cutoff máximo permitido (Hz).
        """
        self.flush()

        # Collect n_samples frames, each containing n_channel values
        samples: np.ndarray = np.array(
            [self.read_raw() for _ in range(n_samples)], dtype=float
        )
        signal_data: np.ndarray = samples[:, idx]

        # Detrend, window, and compute one-sided FFT magnitude
        detrended: np.ndarray = signal_data - np.mean(signal_data)
        windowed: np.ndarray = detrended * np.hanning(len(detrended))
        fft_vals: np.ndarray = np.abs(np.fft.rfft(windowed))
        freqs: np.ndarray = np.fft.rfftfreq(len(detrended), d=1.0 / self.rate_hz)

        # Focus on frequencies above min_freq to ignore near-DC drift
        mask: np.ndarray = freqs >= min_freq
        valid_freqs: np.ndarray = freqs[mask]
        valid_mags: np.ndarray = fft_vals[mask]

        lowest_noise_freq: float | None = None
        if len(valid_mags) >= 3:
            median_mag: float = np.median(valid_mags)
            mad: float = np.median(np.abs(valid_mags - median_mag))
            threshold: float = max(median_mag + 3.0 * mad, 0.15 * np.max(valid_mags))

            peaks: np.ndarray = (
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

        raw_cutoff: float = (
            lowest_noise_freq * 0.8
            if lowest_noise_freq is not None
            else self.rate_hz / 10.0
        )

        # Clamp to safe BiQuad range and caller's max_cutoff
        upper_bound: float = min(max_cutoff, 0.498 * self.rate_hz)
        new_cutoff: float = float(np.clip(raw_cutoff, 0.5, upper_bound))

        print(
            f"Channel {idx}: lowest noise peak at {lowest_noise_freq} Hz -> "
            f"cutoff set to {new_cutoff:.2f} Hz"
        )
        new_cutoff = 15.0
        self.filters[idx] = BiQuadFilter(self.rate_hz, new_cutoff, 1.0)

    def start(self) -> None:
        """Envia o comando AcqStart para iniciar a aquisição."""
        # self.flush()
        payload: bytes = _HDR.pack(OP_ACQSTART, 0)
        # self.sock.setblocking(False)
        self.sock.sendto(payload, self.address)

    def stop(self) -> None:
        """Envia o comando AcqStop para parar a aquisição."""
        payload: bytes = _HDR.pack(OP_ACQSTOP, 0)
        self.sock.sendto(payload, self.address)

    def set_filter(self, idx: int, cutoff: float) -> None:
        """Substitui o filtro de um canal por um novo com o cutoff dado.

        Args:
            idx: Índice do canal.
            cutoff: Nova frequência de corte (Hz).

        Raises:
            AssertionError: Se ``idx`` estiver fora dos limites.
        """
        assert idx <= N_CHANNELS, "SETTING FILTER OUT OF BOUNDS"
        new_filter: BiQuadFilter = BiQuadFilter(self.rate_hz, cutoff, 1.0)
        self.filters[idx] = new_filter

    def set_converter(self, idx: int, gain: float, offset: float) -> None:
        """Substitui o conversor de um canal.

        Args:
            idx: Índice do canal.
            gain: Novo ganho.
            offset: Novo offset.

        Raises:
            AssertionError: Se ``idx`` estiver fora dos limites.
        """
        assert idx <= N_CHANNELS, "SETTING CONVERTER OUT OF BOUNDS"
        print(f"adjusting channel {idx} curve to {gain}*x+{offset}")
        new_converter: Converter = Converter(offset, gain)
        self.converters[idx] = new_converter

    def read_old(self) -> np.ndarray:
        """Lê o próximo frame, filtrando e convertendo para unidades calibradas.

        Em caso de timeout ou reset de conexão, retorna um vetor de zeros.

        Returns:
            Array de ``N_CHANNELS`` valores calibrados.
        """
        # print("reading")
        try:
            raw: bytes
            raw, _ = self.sock.recvfrom(RECV_BUFFER)
            blocks: np.ndarray = parse_pack(raw, N_CHANNELS)
            out_data: np.ndarray = np.zeros(N_CHANNELS)
            for i in range(N_CHANNELS):
                b: float = float(np.mean(blocks[i]))
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
        """Lê o próximo frame, filtra e converte cada canal.

        Returns:
            Array de ``N_CHANNELS`` valores calibrados.
        """
        raw: np.ndarray = self.read_raw()
        out_data: list[float] = []  # np.zeros(N_CHANNELS)
        for i in range(N_CHANNELS):
            r: float = raw[i]
            f: BiQuadFilter = self.filters[i]
            c: Converter = self.converters[i]
            val_f: float = f.process(r)
            val_c: float = c.process(val_f)
            out_data.append(val_c)
            if i == 3:
                print(f"raw {r}, filtered {val_f}, converter {val_c}")

        return np.array(out_data)

    def read_nice(self) -> np.ndarray:
        """Lê uma média de frames e aplica filtro/conversão por canal.

        Returns:
            Array de ``N_CHANNELS`` valores calibrados (média de 16 frames).
        """
        raw: np.ndarray = self.read_buffered()
        out_data: list[float] = []  # np.zeros(N_CHANNELS)
        for i in range(N_CHANNELS):
            r: float = raw[i]
            f: BiQuadFilter = self.filters[i]
            c: Converter = self.converters[i]
            val_f: float = f.process(r)
            val_c: float = c.process(val_f)
            out_data.append(val_c)
        #   if i == 3:
        #        print(f"raw {r}, filtered {val_f}, converter {val_c}")

        return np.array(out_data)

    def read_raw(self) -> np.ndarray:
        """Lê até obter um frame válido, ignorando exceções.

        Returns:
            Array de ``N_CHANNELS`` valores brutos (média dos bursts).
        """
        limit = 100
        for i in range(limit):
            try:
                return self.read_unchecked()
            except Exception as e:
                print(f"Reading exception :{e}")
                pass
        raise RuntimeError(
            f"Reached read retry limit of {limit}, connection is likely dead."
        )

    def read_unchecked(self) -> np.ndarray:
        """Lê o próximo frame sem tratamento de erro.

        Returns:
            Array de ``N_CHANNELS`` valores brutos (média dos bursts).

        Raises:
            socket.timeout: Se nenhum pacote chegar dentro do timeout.
        """
        raw: bytes
        raw, _ = self.sock.recvfrom(RECV_BUFFER)
        blocks: np.ndarray = parse_pack(raw, N_CHANNELS)
        out_data: np.ndarray = np.nan * np.ones(N_CHANNELS)
        for i in range(N_CHANNELS):
            b: float = float(np.mean(blocks[i]))
            out_data[i] = b
        return out_data

    def read_buffered(self, b_size: int = 16) -> np.ndarray:
        """Lê ``b_size`` frames e retorna a média por canal.

        Args:
            b_size: Número de frames a acumular.

        Returns:
            Array de ``N_CHANNELS`` médias.
        """
        out: np.ndarray = np.zeros((N_CHANNELS, b_size))
        for i in range(b_size):
            frame: np.ndarray = self.read_raw()
            out[:, i] = frame
        mean: np.ndarray = np.mean(out, axis=1)
        return mean

    def flush(self) -> None:
        """Descarta todos os datagramas pendentes no socket."""
        self.sock.setblocking(False)
        try:
            while True:
                self.sock.recvfrom(RECV_BUFFER)
        except (BlockingIOError, ConnectionResetError):
            pass
        finally:
            self.sock.settimeout(self.timeout)

    def register_configs(self, dst: dict[str, str]) -> None:
        """Grava a configuração atual no dicionário de destino.

        Valores que podem mudar em runtime vêm dos objetos vivos
        (conversores e filtros); valores estáticos vêm de
        ``channel_configs``.

        Args:
            dst: Dicionário de configuração a ser atualizado in-place.
        """
        for i in range(N_CHANNELS):
            cfg: ChannelConfig = self.channel_configs[i]
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
