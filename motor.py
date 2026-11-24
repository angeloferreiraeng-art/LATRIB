from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException
import numpy as np
import math


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Serial port used by the Modbus client (default; overridden by MotorConfig).
MODBUS_PORT: str = f"COM{5}"
#: Default Modbus slave id.
MODBUS_SLAVE_ID: int = 1
#: Default Modbus baud rate.
MODBUS_BAUD_RATE: int = 19200
#: Default parity setting ("E" = even).
MODBUS_PARITY: str = "E"
#: Default number of stop bits.
MODBUS_STOPBITS: int = 1
#: Default byte size.
MODBUS_BYTESIZE: int = 8

#: Command value that starts the motor.
START_CMD: int = 0x0001
#: Command value that stops the motor.
STOP_CMD: int = 0x0000


# ---------------------------------------------------------------------------
# Controller memory addresses
# ---------------------------------------------------------------------------

#: RPM setpoint (int16).
ADDR_SPEED_VAL: int = 0x0600 + 3
#: Control mode (speed / position / torque).
ADDR_CTRL_MODE: int = 0x0200 + 0
#: Digital input 1 function (disable).
ADDR_DI1_FN: int = 0x0300 + 2
#: Main speed source A.
ADDR_SPEED_SRC_A: int = 0x0600 + 0
#: Auxiliary speed source B.
ADDR_SPEED_SRC_B: int = 0x0600 + 1
#: Speed command source selector.
ADDR_SPEED_SRC_SEL: int = 0x0600 + 2
#: Communication VDI enable.
ADDR_VDI_ENABLE: int = 0x0C00 + 9
#: Virtual digital input level.
ADDR_VDI_LEVEL: int = 0x3100 + 0
#: Multi-speed acceleration for segment 1.
ADDR_MULTI_SPEED_ACCEL_1: int = 0x1200 + 3
#: Multi-speed deceleration for segment 1.
ADDR_MULTI_SPEED_DECCEL_1: int = 0x1200 + 4
#: Multi-speed profile header.
ADDR_MULTI_SPEED_HEAD: int = 0x1200 + 0
#: First multi-speed segment data register.
ADDR_MULTI_SPEED_FIRST: int = 0x1200 + 20
#: Digital input flags base address.
ADDR_DI_FLAGS: int = 0x0300

#: Analog input offset.
ADDR_ANALOG_OFFSET: int = 0x0300 + 36
#: Analog input time constant.
ADDR_ANALOG_TIME_CTE: int = 0x0300 + 37
#: Analog input deadzone.
ADDR_ANALOG_DEADZONE: int = 0x0300 + 39
#: Analog input drift compensation.
ADDR_ANALOG_DRIFT: int = 0x0300 + 40
#: Analog input speed at 10 V.
ADDR_ANALOG_SPEED_10V: int = 0x0300 + 51

#: Position source selector.
ADDR_POS_SRC_SEL: int = 0x0500 + 0
#: Multi-position profile header.
ADDR_MULTI_POS_HEAD: int = 0x1100 + 0
#: First multi-position segment data register.
ADDR_MULTI_POS_FIRST: int = 0x1100 + 12
#: Pulses per revolution for position commands.
ADDR_PULSE_PER_ROT: int = 0x0500 + 2

#: Virtual digital input function 1.
ADDR_VDI_FN1: int = 0x1700 + 0
#: Virtual digital input function 2.
ADDR_VDI_FN2: int = 0x1700 + 2

#: Position loop low-pass filter.
ADDR_POS_LOW_PASS: int = 0x0500 + 4
#: Position loop time constant.
ADDR_POS_TIME_CTE: int = 0x0500 + 6

#: Encoder angle register.
ADDR_ENCODER_ANGLE: int = 0x0B00 + 10

#: P0B-07 (absolute position, 32-bit) — historical naming.
REG_P0B_07: int = 0x0B07
#: P0B-09 (position per revolution, 0..65535).
REG_P0B_09: int = 0x0B09
#: P0B-00 (actual motor speed, rpm).
REG_P0B_00: int = 0x0B00
#: P0B-33 (fault selector / history index). BUG: WRONG ADDRESS.
REG_P0B_33: int = 0x0B21
#: P0B-34 (fault code selected by P0B-33). BUG: WRONG ADDRESS.
REG_P0B_34: int = 0x0B22
#: P05-02 (command units per revolution).
REG_P05_02: int = 0x0502
#: Word order configuration.
REG_P0C_26: int = 0x0C26


# ---------------------------------------------------------------------------
# Peak-shape helpers
# ---------------------------------------------------------------------------


def peak_integral(a: float, b: float) -> float:
    """Compute the normalized area of a peak-shaped speed profile.

    The profile is a blend of a sine wave and a ``|sin|**b`` curve, where
    ``a`` controls the blend weight and ``b`` controls the sharpness of
    the peaks. The result is the ratio of the profile's half-period area
    to that of a pure sine wave, which is used to scale the peak amplitude
    so that a given stroke size is preserved.

    Args:
        a: Blend factor in ``[0, 1]``. ``1`` yields a pure sine; ``0``
            yields a pure ``|sin|**b`` shape.
        b: Exponent applied to ``|sin|`` for the non-sine component.

    Returns:
        The normalized half-period area as a float. A pure sine returns
        ``1.0``.
    """
    p1: float = a * (np.cos(0) - np.cos(np.pi))
    p2: float = (
        (1 - a) * np.sqrt(np.pi) * math.gamma((b + 1) / 2) / math.gamma((b + 2) / 2)
    )
    return (p1 + p2) / 2


def peak_generator(
    omega: float,
    a: float,
    b: float,
    max_rpm: float = 1.0,
) -> Callable[[float], float]:
    """Create a callable that evaluates a peak-shaped speed profile.

    Args:
        omega: Angular frequency of the profile (rad per unit time).
        a: Blend factor in ``[0, 1]`` between sine and ``|sin|**b``.
        b: Exponent for the non-sine component.
        max_rpm: Peak amplitude of the returned profile.

    Returns:
        A function ``f(x) -> float`` that returns the signed speed at
        time ``x``, alternating sign every half period.
    """

    def f(x: float) -> float:
        s = np.abs(np.sin(omega * x))
        return max_rpm * np.sign(np.sin(omega * x)) * (a * s + (1 - a) * (s**b))

    return f


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MotorConfig:
    """Connection parameters for a :class:`Motor`.

    Attributes:
        port: Serial port name (e.g. ``"COM5"``).
        slave_id: Modbus slave id of the drive.
        baud_rate: Serial baud rate.
        parity: Serial parity setting (``"N"``, ``"E"``, ``"O"``).
        stop_bits: Number of stop bits.
        byte_size: Number of data bits per byte.
    """

    port: str = "COM5"
    slave_id: int = 1
    baud_rate: int = 115200
    parity: str = "E"
    stop_bits: int = 1
    byte_size: int = 8

    @classmethod
    def from_config(cls, src: Dict[str, str]) -> "MotorConfig":
        """Build a config from a string-keyed mapping.

        Reads ``MOTOR_PORT``, ``MOTOR_ID``, and ``MOTOR_BAUD_RATE`` from
        ``src``. Parity, stop bits, and byte size keep their defaults.

        Args:
            src: Mapping containing the motor connection keys.

        Returns:
            A new :class:`MotorConfig` instance.
        """
        return cls(
            port=src["MOTOR_PORT"],
            slave_id=int(src["MOTOR_ID"]),
            baud_rate=int(src["MOTOR_BAUD_RATE"]),
        )

    def set_slave_id(self, slave_id: int) -> "MotorConfig":
        """Set the Modbus slave id and return ``self`` for chaining.

        Args:
            slave_id: New slave id.

        Returns:
            This config instance.
        """
        self.slave_id = slave_id
        return self

    def set_baudrate(self, baud_rate: int) -> "MotorConfig":
        """Set the serial baud rate and return ``self`` for chaining.

        Args:
            baud_rate: New baud rate.

        Returns:
            This config instance.
        """
        self.baud_rate = baud_rate
        return self


# ---------------------------------------------------------------------------
# Motor controller
# ---------------------------------------------------------------------------


class Motor:
    """Modbus-based controller for a servo motor drive.

    The class exposes helpers to configure the drive for analog speed
    control, multi-segment speed control, and multi-segment position
    control, along with basic start/stop and position-read operations.

    Attributes:
        client: Underlying pymodbus serial client.
        state: High-level state string (``"INIT"``, ``"READY"``,
            ``"RUN"``, ``"STOP"``).
        mode: Active control mode (``"NONE"``, ``"VEL_ANALOG"``,
            ``"VEL_CTRL"``, ``"POS_SEC"``, ``"POS_CTRL"``).
        port: Serial port name.
        baud_rate: Serial baud rate.
        parity: Serial parity.
        stop_bits: Number of stop bits.
        byte_size: Data bits per byte.
        a_factor: Blend factor used by :func:`peak_generator`.
        b_factor: Exponent used by :func:`peak_generator`.
    """

    client: ModbusSerialClient
    state: str = "INIT"
    mode: str = "NONE"

    def __init__(self, config: Optional[MotorConfig]) -> None:
        """Open the serial connection and initialize motor state.

        Args:
            config: Connection parameters. If ``None``, a default
                :class:`MotorConfig` is used.

        Raises:
            RuntimeError: If the serial connection cannot be established.
        """
        if not config:
            config = MotorConfig()
        try:
            self.port: str = config.port
            self.baud_rate: int = config.baud_rate
            self.parity: str = config.parity
            self.stop_bits: int = config.stop_bits
            self.byte_size: int = config.byte_size
            self.client = ModbusSerialClient(
                port=config.port,
                baudrate=config.baud_rate,
                parity=config.parity,
                stopbits=config.stop_bits,
                bytesize=config.byte_size,
                timeout=(100 / 1_000_000) + 0.1,
            )
            self.client.connect()
            self.state = "READY"
            self.mode = "NONE"
            self.a_factor: float = 1
            self.b_factor: float = 0
        except Exception as e:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(f"CONNECTION ERROR:::{e}") from e

    def __del__(self) -> None:
        """Close the Modbus client on garbage collection, if present."""
        client = getattr(self, "client", None)
        if client is not None:
            client.close()

    # -- Low-level Modbus helpers -----------------------------------------

    def _write(self, addr: int, value: int) -> None:
        """Write a single 16-bit register.

        Args:
            addr: Register address.
            value: Value to write.

        Raises:
            RuntimeError: If the Modbus write reports an error.
        """
        r = self.client.write_register(addr, value)
        if r.isError():
            print(r)
            raise RuntimeError("MODBUS WRITE FAILED")

    def _write_many(self, addr: int, values: List[int]) -> None:
        """Write a contiguous block of 16-bit registers.

        Args:
            addr: Starting register address.
            values: Values to write.

        Raises:
            RuntimeError: If the Modbus write reports an error.
        """
        r = self.client.write_registers(addr, values)
        if r.isError():
            print(r)
            raise RuntimeError("MODBUS WRITE FAILED")

    def _write_u32(self, addr: int, value: int) -> None:
        """Write a signed 32-bit value as two little-endian 16-bit words.

        Args:
            addr: Starting register address.
            value: Signed value to write (must fit in 32 bits).

        Raises:
            AssertionError: If ``value`` does not fit in 32 bits.
            RuntimeError: If the Modbus write reports an error.
        """
        assert np.abs(value) < (1 << 31), "POSITION TOO BIG"
        b_high: int = (value >> 16) & 0xFFFF
        b_low: int = value & 0xFFFF
        r = self.client.write_registers(addr, [b_low, b_high])
        if r.isError():
            print(r)
            raise RuntimeError("MODBUS WRITE FAILED")

    def _read_u32(self, addr: int) -> int:
        """Read a 32-bit value from two little-endian 16-bit registers.

        Args:
            addr: Starting register address.

        Returns:
            The reconstructed 32-bit value.

        Raises:
            RuntimeError: If the Modbus read reports an error.
        """
        r = self.client.read_holding_registers(addr, count=2)
        if r.isError():
            print(r)
            raise RuntimeError("MODBUS READ FAILED")
        b_low, b_high = r.registers[0], r.registers[1]
        return b_high << 16 | b_low

    def _read(self, addr: int) -> int:
        """Read a single 16-bit holding register.

        Args:
            addr: Register address.

        Returns:
            The register value as an int.

        Raises:
            RuntimeError: If the Modbus read reports an error.
        """
        r = self.client.read_holding_registers(addr)
        if r.isError():
            print(r)
            raise RuntimeError("MODBUS READ FAILED")
        return r.registers[0]

    def _read_many(self, addr: int, count: int) -> List[int]:
        """Read a contiguous block of 16-bit holding registers.

        Args:
            addr: Starting register address.
            count: Number of registers to read.

        Returns:
            The register values as a list of ints.

        Raises:
            RuntimeError: If the Modbus read reports an error.
        """
        r = self.client.read_holding_registers(addr, count=count)
        if r.isError():
            print(r)
            raise RuntimeError("MODBUS READ FAILED")
        return list(r.registers)

    # -- Configuration persistence ----------------------------------------

    def register_configs(self, dst: Dict[str, str]) -> None:
        """Write the connection settings into ``dst`` as strings.

        Args:
            dst: Mapping to populate with ``MOTOR_PORT``, ``MOTOR_ID``,
                and ``MOTOR_BAUD_RATE``.
        """
        dst["MOTOR_PORT"] = str(self.port)
        dst["MOTOR_ID"] = str(1)
        dst["MOTOR_BAUD_RATE"] = str(self.baud_rate)
        # TODO: parity / stop_bits / byte_size are currently ignored.

    # -- High-level operations --------------------------------------------

    def get_pos(self) -> int:
        """Read the current encoder angle register.

        Returns:
            The raw encoder angle value.
        """
        return self._read(ADDR_ENCODER_ANGLE)

    def start(self) -> None:
        """Enable the drive and transition to the ``"RUN"`` state.

        Raises:
            RuntimeError: If no control mode has been selected.
        """
        print(f":::::STARTING MOTOR WITH MODE {self.mode}::::::")
        if self.mode == "NONE":
            raise RuntimeError("Trying to run motor on undefined mode.")
        self._write(ADDR_VDI_LEVEL, 0b11)
        self.state = "RUN"

    def stop(self) -> None:
        """Disable the drive and transition to the ``"STOP"`` state."""
        print(":::::STOPING MOTOR::::::")
        self._write(ADDR_VDI_LEVEL, 0)
        self.state = "STOP"

    def analog_mode(self) -> None:
        """Configure the drive for analog speed control.

        Sets the control mode to speed, selects analog input A as the
        speed command source, and zeroes the analog input conditioning
        parameters. On return, :attr:`mode` is ``"VEL_ANALOG"``.
        """
        print(":::::SWITCHING TO ANALOG INPUT::::::")
        # NOTE: I don't know what any of these do, keep it just to be safe.
        self._write(ADDR_VDI_LEVEL, 0)
        self._write(ADDR_VDI_ENABLE, 1)
        self._write(ADDR_DI1_FN, 0)

        # Set mode to speed
        self._write(ADDR_CTRL_MODE, 0)

        # Sets speed command to analog A
        data: List[int] = [1, 2, 0]
        self._write_many(ADDR_SPEED_SRC_A, data)

        # Set analog input params
        self._write(ADDR_ANALOG_OFFSET, 0)
        self._write(ADDR_ANALOG_TIME_CTE, 0)
        self._write(ADDR_ANALOG_DEADZONE, 0)
        self._write(ADDR_ANALOG_DRIFT, 0)
        self._write(ADDR_ANALOG_DRIFT, 0)
        self._write(ADDR_ANALOG_SPEED_10V, 500)
        self.mode = "VEL_ANALOG"

    def pos_ctrl(
        self,
        f: Callable[[float], float],
        times: Sequence[float],
    ) -> None:
        """Configure a multi-segment position profile from a function.

        The profile is sampled at the cumulative sum of ``times``. The
        final position is used as the starting position so the profile
        wraps continuously.

        Args:
            f: Position function returning rotations as a function of
                elapsed time.
            times: Segment durations. Must contain at most 16 entries.

        Raises:
            AssertionError: If more than 16 segments are supplied.
        """
        assert len(times) <= 16, "TIMES ARRAY CANT BE BIGGER THAN 16 ELEMENTS"
        print(":::::SETTING POSITION SINE::::::")
        PULSE_PER_ROT: int = 10000
        # NOTE: I don't know what any of these do, keep it just to be safe.
        self._write(ADDR_VDI_LEVEL, 0)
        self._write(ADDR_VDI_ENABLE, 1)
        self._write(ADDR_DI1_FN, 0)

        self._write(ADDR_VDI_FN1, 1)
        self._write(ADDR_VDI_FN2, 28)

        # Set mode to position
        self._write(ADDR_CTRL_MODE, 1)

        self._write(ADDR_POS_SRC_SEL, 2)  # multi segment
        self._write(ADDR_PULSE_PER_ROT, PULSE_PER_ROT)
        self._write(ADDR_POS_LOW_PASS, 10)
        self._write(ADDR_POS_TIME_CTE, 10)

        n_segments: int = len(times)

        TIME_UNIT: int = 0
        header: List[int] = [1, n_segments, 0, TIME_UNIT, 0, 1]
        self._write_many(ADDR_MULTI_POS_HEAD, header)

        t: float = 0.0
        x0: int = f(times[-1])
        addr: int = ADDR_MULTI_POS_FIRST
        for dt in times:
            t += dt
            raw: float = f(t)  # rotations
            print(f"RAW ROTATIONS{raw}")
            x1: int = int(np.floor(PULSE_PER_ROT * raw))
            self._write_u32(addr, x1)
            _ = self._read_u32(addr)  # echo check (unused)
            addr += 2
            speed: int = int((x1 - x0) / dt) & 0xFFFF
            x0 = x1
            self._write(addr, speed)
            addr += 1
            accel: int = int(500 * dt)  # TODO: do real accel calc
            self._write(addr, accel)
            addr += 1
            wait: int = 0
            self._write(addr, wait)
            addr += 1

        _ = self._read_many(ADDR_MULTI_POS_FIRST, 5 * len(times))
        self.mode = "POS_SEC"

    def vel_ctrl(
        self,
        f: Callable[[float], float],
        times: Sequence[float],
    ) -> None:
        """Configure a symmetric multi-segment velocity profile.

        Samples ``f`` over the cumulative half-period ``times`` and builds
        a full-period profile by mirroring the samples with negated speed.
        Time units are divided by 10 before sampling.

        Args:
            f: Speed function returning non-negative RPM.
            times: Half-period segment durations. At most 8 entries.

        Raises:
            AssertionError: If more than 8 segments are supplied, if any
                sampled speed is negative, or if the generated profile
                lengths are inconsistent.
        """
        assert len(times) <= 8, "TIMES ARRAY CANT BE BIGGER THAN 8 ELEMENTS"
        v_plus: List[int] = []
        v_minus: List[int] = []
        t: float = 0.0
        for dt in times:
            t += dt / 10
            speed: int = int(np.floor(f(t)))
            print(speed)
            assert speed >= 0, "HALFSTROKE SPEED NEEDS TO BE POSITIVE."
            v_plus.append(speed)
            v_minus.append(-speed)
        speeds: np.ndarray = np.append(v_plus, np.flip(v_minus))
        full_times: np.ndarray = np.append(times, times)
        print(speeds)
        assert len(speeds) == len(full_times), "VEL CONTROL THING BROKE"
        self.vel_ctrl_unchecked(list(full_times), list(speeds))

    def set_vel_profile(self) -> None:
        """Build and upload a default 8-segment velocity profile.

        Uses :attr:`a_factor` and :attr:`b_factor` with a fixed 50 RPM
        stroke and 1-second segments. On return, :attr:`mode` is
        ``"VEL_CTRL"``.
        """
        NUM_SEGMENTS: int = 8
        SEG_TIME: float = 1
        times: np.ndarray = np.ones(NUM_SEGMENTS) * SEG_TIME
        half_period: float = float(np.sum(times) / 10)
        print("period", 2 * half_period)
        STROKE_SIZE: int = 50  # rpm
        omega: float = np.pi / half_period
        print("omega", omega)
        modifier: float = peak_integral(self.a_factor, self.b_factor)
        max_rpm: float = 0.5 * STROKE_SIZE / modifier
        f: Callable[[float], float] = peak_generator(
            omega, self.a_factor, self.b_factor, max_rpm=max_rpm
        )
        v_plus: List[int] = []
        v_minus: List[int] = []
        t: float = 0.0
        for dt in times:
            t += dt / 10
            speed: int = int(np.floor(f(t)))
            print(speed)
            assert speed >= 0, "HALFSTROKE SPEED NEEDS TO BE POSITIVE."
            v_plus.append(speed)
            v_minus.append(-speed)
        speeds: np.ndarray = np.append(v_plus, np.flip(v_minus))
        full_times: np.ndarray = np.append(times, times)
        print(speeds)
        assert len(speeds) == len(full_times), "VEL CONTROL THING BROKE"
        self.vel_ctrl_unchecked(list(full_times), list(speeds))
        self.mode = "VEL_CTRL"

    def vel_ctrl_unchecked(
        self,
        times: Sequence[float],
        speeds: Sequence[float],
    ) -> None:
        """Upload a pre-built multi-segment velocity profile.

        Unlike :meth:`vel_ctrl`, this does not validate that speeds are
        positive or that the profile is symmetric.

        Args:
            times: Segment durations.
            speeds: Signed speed per segment (RPM).
        """
        # NOTE: THIS THROWS ERROR
        print(":::::SETTING SPEED SINE::::::")
        self._write(ADDR_VDI_LEVEL, 0)
        self._write(ADDR_VDI_ENABLE, 1)
        self._write(ADDR_DI1_FN, 0)

        # Set mode to speed
        self._write(ADDR_CTRL_MODE, 0)

        # Sets speed command to multi segment
        data: List[int] = [0, 5, 1]
        self._write_many(ADDR_SPEED_SRC_A, data)

        n_segments: int = len(times)
        header: List[int] = [1, n_segments, 0]
        self._write_many(ADDR_MULTI_SPEED_HEAD, header)

        self._write(ADDR_MULTI_SPEED_ACCEL_1, 1)
        self._write(ADDR_MULTI_SPEED_DECCEL_1, 1)

        payload: List[int] = []
        t: float = 0.0
        for s, dt in zip(speeds, times):
            t += dt
            speed: int = int(np.floor(s))
            if speed < 0:
                speed &= 0xFFFF
            payload.append(speed)
            payload.append(int(dt))
            payload.append(0)

        for i, d in enumerate(payload):
            self._write(ADDR_MULTI_SPEED_FIRST + i, d)
            v: int = self._read(ADDR_MULTI_SPEED_FIRST + i)
            assert d == v, f"VALUE WRITE ERROR: TRIED TO WRITE {d}, READ BACK IS {v}"

    def set_pos_profile(self) -> None:
        """Build and upload a default 8-segment position profile.

        Uses :attr:`a_factor` and :attr:`b_factor` with a fixed 50 RPM
        stroke and 1-second segments. On return, :attr:`mode` is
        ``"POS_CTRL"``.
        """
        NUM_SEGMENTS: int = 8
        SEG_TIME: float = 1
        times: np.ndarray = np.ones(NUM_SEGMENTS) * SEG_TIME
        half_period: float = float(np.sum(times) / 10)
        print("period", 2 * half_period)
        STROKE_SIZE: int = 50  # rpm
        omega: float = np.pi / half_period
        print("omega", omega)
        modifier: float = peak_integral(self.a_factor, self.b_factor)
        max_rpm: float = 0.5 * STROKE_SIZE / modifier
        f: Callable[[float], float] = peak_generator(
            omega, self.a_factor, self.b_factor, max_rpm=max_rpm
        )
        v_plus: List[int] = []
        v_minus: List[int] = []
        t: float = 0.0
        for dt in times:
            t += dt / 10
            speed: int = int(np.floor(f(t)))
            print(speed)
            assert speed >= 0, "HALFSTROKE SPEED NEEDS TO BE POSITIVE."
            v_plus.append(speed)
            v_minus.append(-speed)
        speeds: np.ndarray = np.append(v_plus, np.flip(v_minus))
        full_times: np.ndarray = np.append(times, times)
        assert len(speeds) == len(full_times), "VEL CONTROL THING BROKE"
        # NOTE: positions are derived from the same speed samples here.
        self.pos_ctrl_unchecked(list(full_times), list(speeds))
        self.mode = "POS_CTRL"

    def pos_ctrl_unchecked(
        self,
        times: Sequence[float],
        positions: Sequence[float],
    ) -> None:
        """Upload a pre-built multi-segment position profile.

        Args:
            times: Segment durations.
            positions: Target position per segment, in rotations.
        """
        print(":::::SETTING POSITION SINE::::::")
        PULSE_PER_ROT: int = 10000
        # NOTE: I don't know what any of these do, keep it just to be safe.
        self._write(ADDR_VDI_LEVEL, 0)
        self._write(ADDR_VDI_ENABLE, 1)
        self._write(ADDR_DI1_FN, 0)

        self._write(ADDR_VDI_FN1, 1)
        self._write(ADDR_VDI_FN2, 28)

        # Set mode to position
        self._write(ADDR_CTRL_MODE, 1)

        self._write(ADDR_POS_SRC_SEL, 2)  # multi segment
        self._write(ADDR_PULSE_PER_ROT, PULSE_PER_ROT)
        self._write(ADDR_POS_LOW_PASS, 10)
        self._write(ADDR_POS_TIME_CTE, 10)

        n_segments: int = len(times)

        TIME_UNIT: int = 0
        header: List[int] = [1, n_segments, 0, TIME_UNIT, 0, 1]
        self._write_many(ADDR_MULTI_POS_HEAD, header)

        t: float = 0.0
        x0: int = int(np.floor(PULSE_PER_ROT * positions[-1]))
        addr: int = ADDR_MULTI_POS_FIRST
        for dt, rot in zip(times, positions):
            t += dt
            x1: int = int(np.floor(PULSE_PER_ROT * rot))
            self._write_u32(addr, x1)
            _ = self._read_u32(addr)  # echo check (unused)
            addr += 2
            speed: int = int((x1 - x0) / dt) & 0xFFFF
            x0 = x1
            self._write(addr, speed)
            addr += 1
            accel: int = 0  # int(500 * dt)  # TODO: do real accel calc
            self._write(addr, accel)
            addr += 1
            wait: int = 0
            self._write(addr, wait)
            addr += 1

        _ = self._read_many(ADDR_MULTI_POS_FIRST, 5 * len(times))


if __name__ == "__main__":
    raise RuntimeError("Motor is a library, it cannot be run directly")
