from dataclasses import dataclass
from io import FileIO, TextIOWrapper
from pathlib import Path
import threading
import time
import csv
import numpy as np
from queue import Queue, Empty, Full

from motor import Motor, MotorConfig, peak_generator, peak_integral
from sensors import Sensors, SensorsConfig
from enum import Enum
import math
from copy import copy

from typing import Any, Dict, List, Optional, Tuple


#: Number of sensor channels supported by the system.
N_CHANNELS: int = 8


class State(Enum):
    """High-level state of the :class:`Orchestrator`.

    Attributes:
        INIT: System constructed but not yet initialized.
        ERROR: Unrecoverable error state; transitions are frozen.
        IDLE: Idle; ready to accept commands (note: value is ``(1,)``,
            an accidental tuple — kept for backward compatibility).
        READY: Configured and ready to start.
        RUNNING: Actively acquiring / driving.
        CALIBRATING: Per-channel calibration in progress.
        STOP: Stopped after a run.
    """

    INIT = 0
    ERROR = -1
    IDLE = (1,)  # NOTE: accidental tuple; preserved for compatibility
    READY = 2
    RUNNING = 3
    CALIBRATING = 4
    STOP = 5


def parse_config(path: Path) -> Dict[str, str]:
    """Parse a key-value config file.

    Lines are split on the first ``=`` so values may contain ``=``.
    Empty lines and lines starting with ``#`` are ignored.

    Args:
        path: Path to the config file.

    Returns:
        Mapping of stripped keys to stripped values.
    """
    path = Path(path)
    if not path.is_file():
        return None
    config: Dict[str, str] = {}

    with open(path, "r") as file:
        for line in file:
            # Remove leading/trailing whitespace
            line = line.strip()

            # Skip empty lines and comment lines
            if not line or line.startswith("#"):
                continue

            # Split into key and value (only at the first '=' to allow '=' in values)
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    return config


class Orchestrator:
    """Top-level state object coordinating sensors and motor.

    The orchestrator owns a :class:`Sensors` and a :class:`Motor`
    instance and exposes a small command API (start / stop / calibrate
    / read) driven by a simple state machine (:class:`State`).

    Attributes:
        state: Current state of the state machine.
        last_calibration: Human-readable timestamp of the last
            calibration, or ``"NEVER"``.
        enables: Per-channel enabled flags (length :data:`N_CHANNELS`).
        ch_names: Per-channel human-readable names.
        channel_idx: Index of the channel currently being calibrated.
        read_list: Sensor readings collected during calibration.
        ref_list: Reference values collected during calibration.
        sensors: Sensor manager instance.
        motor: Motor controller instance.
    """

    state: State = State.INIT

    def __init__(self, config: Optional[Dict[str, str]] = None) -> None:
        """Construct the orchestrator and its subsystems.

        Args:
            config: Parsed configuration mapping (see
                :func:`parse_config`). When ``None``, defaults are used
                and all channels start disabled with ``"UNDEFINED"``
                names.
        """
        self.state = State.INIT
        if config is None:
            print("No config given, using default values.")
            self.last_calibration: str = "NEVER"
            self.enables: List[bool] = [False for _ in range(N_CHANNELS)]
            self.ch_names: List[str] = ["UNDEFINED" for _ in range(N_CHANNELS)]
        else:
            self.last_calibration = config["SYSTEM_LAST_CALIBRATION_DATE"]
            self.enables, self.ch_names = [], []
            for i in range(N_CHANNELS):
                print(config[f"SYSTEM_CHANNEL{i}_ENABLE"])
                enable: bool = (
                    True if config[f"SYSTEM_CHANNEL{
                        i}_ENABLE"] == "True" else False
                )
                self.enables.append(enable)
                self.ch_names.append(config[f"SYSTEM_CHANNEL{i}_NAME"])

        self.channel_idx: int = 0
        self.read_list: List[float] = []
        self.ref_list: List[float] = []
        s_config: SensorsConfig = (
            SensorsConfig() if config is None else SensorsConfig.from_config(config)
        )
        self.sensors: Sensors = Sensors(s_config)
        m_config: MotorConfig = (
            MotorConfig() if config is None else MotorConfig.from_config(config)
        )
        self.motor: Motor = Motor(m_config)

        self.state = State.IDLE
        print(self.enables)

    def eval_state(
        self,
        init_ok: bool = False,
        start_on: bool = False,
        stop_on: bool = False,
        calibrate_on: bool = False,
    ) -> None:
        """Evaluate the state-machine transition from the current state.

        Args:
            init_ok: Whether initialization succeeded.
            start_on: Whether a start request is currently asserted.
            stop_on: Whether a stop request is currently asserted.
            calibrate_on: Whether a calibration request is asserted.
                Currently unused by the transition table.
        """
        match self.state:
            case State.ERROR:
                return
            case State.INIT:
                if init_ok:
                    self.state = State.READY
                else:
                    self.state = State.ERROR
            case State.READY:
                if not stop_on and start_on:
                    self.state = State.RUNNING
                    return
            case State.RUNNING:
                if stop_on and not start_on:
                    self.state = State.STOP

    def check_state(self) -> State:
        """Return the current state.

        Returns:
            The current :class:`State` value.
        """
        return self.state

    def start_calibration(self, channel: int) -> None:
        """Enter calibration mode for a single channel.

        Args:
            channel: Index of the channel to calibrate.
        """
        self.state = State.CALIBRATING
        self.channel_idx = channel
        self.sensors.start()
        self.sensors.calibrate_filter(channel, max_cutoff=15)

    def profile_factors(self, a: float, b: float) -> None:
        """Set the motor's peak-shape blend factors.

        Args:
            a: Blend factor in ``[0, 1]`` between sine and ``|sin|**b``.
            b: Exponent for the non-sine component.
        """
        self.motor.a_factor = a
        self.motor.b_factor = b

    def ref_point(self, value: float) -> None:
        """Record a calibration reference point against the current reading.

        Only meaningful while in :attr:`State.CALIBRATING`; otherwise the
        call is logged and ignored.

        Args:
            value: The known reference value for the current channel.
        """
        if self.state == State.CALIBRATING:
            print("CHANNEL", self.channel_idx)
            self.sensors.flush()
            frame = self.sensors.read_buffered()
            read: float = frame[self.channel_idx]
            print(value, read)
            self.ref_list.append(value)
            self.read_list.append(read)
        else:
            print("SYSTEM NOT IN CALIBRATION MODE, IGNORING.")

    def commit_calibration(self) -> None:
        """Fit a linear curve to the collected calibration points and apply it.

        Requires at least two points with non-identical readings. On
        success, clears the buffers, stops the sensors, and returns to
        :attr:`State.IDLE`.
        """
        if self.state is not State.CALIBRATING:
            print("Not in calibration mode, ignoring.")
            return

        x: np.ndarray = np.asarray(self.read_list, dtype=float)
        y: np.ndarray = np.asarray(self.ref_list, dtype=float)
        print("POINTS::", x, y)

        assert len(x) == len(y), "listx and listy must have the same length"
        assert len(x) >= 2, "At least 2 data points are required"
        assert np.ptp(x) > 0, "Cannot fit: all x values are identical"

        # Least-squares fit: y = a*x + b
        a: float
        b: float
        a, b = np.polyfit(x, y, 1)

        print(f"SETTED CH-{self.channel_idx} CURVE TO {a}x+{b}")
        self.sensors.set_converter(self.channel_idx, a, b)
        print("VALUES:", a * x + b)
        self.read_list.clear()
        self.ref_list.clear()
        self.sensors.stop()
        self.state = State.IDLE

    def commit_profile(self) -> None:
        """Upload the default velocity profile and transition to READY."""
        try:
            self.motor.set_vel_profile()
            self.state = State.READY
        except Exception as e:
            print(f"Received Exception:{e}")
            self.state = State.ERROR

    def commit_pos_profile(self) -> None:
        """Upload the default position profile and transition to READY."""
        try:
            self.motor.set_pos_profile()
            self.state = State.READY
        except Exception as e:
            print(f"Received Exception:{e}")
            self.state = State.ERROR

    def dump_config(self, path: Path) -> None:
        """Write the current configuration to a key-value file.

        Args:
            path: Destination file path.
        """
        cfg: Dict[str, str] = {}

        cfg["SYSTEM_LAST_CALIBRATION_DATE"] = self.last_calibration
        for i in range(N_CHANNELS):
            cfg[f"SYSTEM_CHANNEL{i}_ENABLE"] = str(self.enables[i])
            cfg[f"SYSTEM_CHANNEL{i}_NAME"] = str(self.ch_names[i])
        self.sensors.register_configs(cfg)
        self.motor.register_configs(cfg)
        with open(path, "w") as f:
            for key, value in cfg.items():
                f.write(f"{key}={value}\n")

    def stop(self) -> None:
        """Stop the motor and sensors and transition via the state machine."""
        self.motor.stop()
        self.sensors.stop()
        self.eval_state(stop_on=True)

    def start(self) -> bool:
        """Start both sensors and motor if the current state permits.

        Refuses to start when already :attr:`State.RUNNING` or in
        :attr:`State.ERROR`.

        Returns:
            ``True`` if the start succeeded, ``False`` otherwise.
        """
        if self.state == State.RUNNING or self.state == State.ERROR:
            print(f"Trying to start on state {self.state.name}, ignoring.")
            return False

        else:
            self.sensors.start()
            self.motor.start()
            self.state = State.RUNNING
            return True

    def start_raw(self) -> None:
        """Start the sensors only (no motor), transitioning to RUNNING."""
        self.sensors.start()
        self.state = State.RUNNING

    def velocity_func(self):
        a = self.motor.a_factor
        b = self.motor.b_factor
        modif = peak_integral(a, b)
        max_val = 1 / modif
        omega = self.motor.get_omega()
        return peak_generator(omega, a, b, max_rpm=max_val)

    def velocity_discrete(self):
        a = self.motor.a_factor
        b = self.motor.b_factor
        modif = peak_integral(a, b)
        max_val = 1 / modif
        omega = self.motor.get_omega()
        f = peak_generator(omega, a, b, max_rpm=max_val)
        step = 0.1
        def g(x): return f(step * (0.5 + (x // step)))
        return g

    def reset(self) -> None:
        """Flush the sensor buffers."""
        self.sensors.flush()

    def add_channel(self, idx: int, name: str) -> None:
        """Enable a channel and assign it a name.

        Args:
            idx: Channel index; must be ``<= N_CHANNELS``.
            name: Human-readable channel name.

        Raises:
            AssertionError: If ``idx`` exceeds :data:`N_CHANNELS`.
        """
        assert idx <= N_CHANNELS, "CHANNEL INDEX TOO BIG"
        self.enables[idx] = True
        self.ch_names[idx] = name

    def channels(self) -> List[Dict[str, Any]]:
        """Return descriptors for all enabled channels.

        Returns:
            A list of ``{"number": int, "name": str}`` dicts, one per
            enabled channel, in ascending index order.
        """
        out: List[Dict[str, Any]] = []
        for i in range(N_CHANNELS):
            if self.enables[i]:
                out.append({"number": i, "name": self.ch_names[i]})
        return out

    def read(self):
        if self.state == State.RUNNING or self.state == State.CALIBRATING:
            # self.sensors.flush()

            frame = self.sensors.read_nice()
            out_frame = []
            for i in range(N_CHANNELS):
                if self.enables[i]:
                    out_frame.append(frame[i])
            return out_frame
        else:
            print("Sensors are not activated!")
            return []

    def read_many(self) -> np.ndarray:
        """Lê todos os frames disponíveis apenas dos canais habilitados.

        Args:
            **kwargs: Argumentos repassados para ``self.sensors.read_many``.

        Returns:
            Array 2D de formato ``(n_frames, n_enabled)`` com os valores dos canais
            ativos, ou array vazio quando inativo ou sem frames.
        """
        enabled_indices = [i for i in range(N_CHANNELS) if self.enables[i]]
        n_enabled = len(enabled_indices)

        if self.state in (State.RUNNING, State.CALIBRATING):
            # self.sensors.flush()
            frames: np.ndarray = self.sensors.read_many()
            if frames.shape[0] == 0 or n_enabled == 0:
                return np.empty((0, n_enabled), dtype=float)
            return frames[:, enabled_indices]

        print("Sensors are not activated!")
        return np.empty((0, n_enabled), dtype=float)
