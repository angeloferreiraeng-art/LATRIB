from dataclasses import dataclass
from io import FileIO, TextIOWrapper
from pathlib import Path
import threading
import time
import csv
import numpy as np
from queue import Queue, Empty, Full

from motor import Motor, MotorConfig
from sensors import Sensors, SensorsConfig
from enum import Enum
import math
from copy import copy


class State(Enum):
    INIT = 0
    ERROR = -1
    READY = 1
    RUNNING = 2
    CALIBRATING = 3
    STOP = 4


def parse_config(path):
    config = {}

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
    """Top class state object"""

    state: State = State.INIT

    def __init__(self, config: dict[str, str] | None) -> None:
        self.state = State.INIT
        self.channel_idx: int = 0
        self.read_list: list[float] = []
        self.ref_list: list[float] = []
        try:
            s_config = (
                SensorsConfig() if config is None else SensorsConfig.from_config(config)
            )
            self.sensors: Sensors = Sensors(s_config)
            m_config = (
                MotorConfig() if config is None else MotorConfig.from_config(config)
            )
            self.motor: Motor = Motor(m_config)
            # self.motor.stop()
        except Exception as e:
            print("::::Orchestrator launch failed!:::::")
            print(e)
            self.eval_state(init_ok=False)
        self.state = State.READY

    def eval_state(
        self,
        init_ok: bool = False,
        start_on: bool = False,
        stop_on: bool = False,
        calibrate_on: bool = False,
    ):
        """Receives all system signals for the grafcet of the system and
        evaluate the proper transition for the state machine
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

    def check_state(self) -> str:
        return self.state.name

    def start_calibration(self, channel: int):
        self.state = State.CALIBRATING
        self.channel_idx = channel
        self.sensors.start()
        self.sensors.calibrate_filter(channel)

    def profile_factors(self, a, b):
        self.motor.a_factor = a
        self.motor.b_factor = b

    def ref_point(self, value: float):
        if self.state == State.CALIBRATING:
            print("CHANNEL", self.channel_idx)
            frame = self.sensors.read_buffered()
            read = frame[self.channel_idx]
            print(value, read)
            self.ref_list.append(value)
            self.read_list.append(read)
        else:
            print("SYSTEM NOT IN CALIBRATION MODE, IGNORING.")

    def commit_calibration(self):
        x = np.array(self.read_list, dtype=float)
        y = np.array(self.ref_list, dtype=float)
        assert len(x) == len(y), "listx and listy must have the same length"
        assert len(x) >= 2, "At least 2 data points are required"
        # Least-squares solution via closed-form formula
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        denominator = n * sum_x2 - sum_x**2
        assert denominator != 0, "Cannot fit: all x values are identical"
        a = (n * sum_xy - sum_x * sum_y) / denominator
        b = (sum_y * sum_x2 - sum_x * sum_xy) / denominator
        self.sensors.set_converter(self.channel_idx, a, b)
        self.read_list = []
        self.ref_list = []
        self.state = State.READY

    def commit_profile(self):
        self.motor.set_vel_profile()

    def dump_config(self, path: Path):
        cfg = {}
        self.sensors.register_configs(cfg)
        self.motor.register_configs(cfg)
        with open(path, "w") as f:
            for key, value in cfg.items():
                f.write(f"{key}={value}\n")

    def stop(self):
        self.motor.stop()
        self.sensors.stop()
        self.eval_state(stop_on=True)

    def start(self):
        if self.state == State.READY:
            self.sensors.start()
            # self.motor.start()
            print("started")
        else:
            print(f"Trying to start on state {self.state.name}, ignoring.")

    def read_channel(self, channel) -> float:
        """Outputs buffered data to the given file. May fail."""
        frame = self.sensors.read()
        return frame[channel]
