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

class Command:
    pass

def peak_integral(a: float, b: float) -> float:
    """Calcs the % of the area of the given peak function compared to a sine wave"""
    p1 = a * (np.cos(0) - np.cos(np.pi))
    p2 = (1 - a) * np.sqrt(np.pi) * math.gamma((b + 1) / 2) / math.gamma((b + 2) / 2)
    return (p1 + p2) / 2


def peak_generator(omega: float, a: float, b: float, max_rpm: float = 1.0):
    """Generates a function that outputs the speed curve with the given factors"""
    f = lambda x: (
        max_rpm
        * np.sign(np.sin(omega * x))
        * (a * np.abs(np.sin(omega * x)) + (1 - a) * (np.abs(np.sin(omega * x)) ** b))
    )
    return f


def _run_motor(
    fail_event: threading.Event,
    cmd_buffer: Queue,
    out_buffer: Queue,
    config: dict[str, str] | None = None,
):
    motor = None
    try:
        m_config = MotorConfig() if config is None else MotorConfig.from_config(config)
        motor = Motor(m_config)
        #motor.stop()
    except Exception as e:
        print(f"Motor initialization failure: {e}")
        fail_event.set()
        return

    NUM_SEGMENTS = 8
    SEG_TIME = 1
    times = np.ones(NUM_SEGMENTS) * SEG_TIME
    half_period = np.sum(times) / 10
    print("period", 2 * half_period)
    STROKE_SIZE = 50  # rpm
    omega = np.pi / (half_period)
    print("omega", omega)
    a = 1
    b = 0

    current_cmd = None

    while True:
        try:
            new_cmd = _pop_from(cmd_buffer)
            match new_cmd:
                case Command(code="SET_A"):
                    val = new_cmd.value
                    a = val
                    print(f"A parameter set to {val}")
                case Command(code="SET_B"):
                    val = new_cmd.value
                    b = val
                    print(f"B parameter set to {val}")
                case Command(code="VEL_CTRL"):
                    modifier = peak_integral(a, b)
                    max_rpm = 0.5 * STROKE_SIZE / modifier
                    f = peak_generator(omega, a, b, max_rpm=max_rpm)
                    #motor.vel_ctrl(f, times)
                case Command(code="POS_CTRL"):
                    raise RuntimeError("NOT IMPLEMENTED!")
                case Command(code="ANALOG_MODE"):
                    #motor.analog_mode()
                    pass
                case Command(code="START"):
                    print("MOTR STARTING")
                    # motor.start()
                case Command(code="STOP"):
                    print("MOTR STOPING")
                    #motor.stop()
                case None:
                    pass
        except Exception as e:
            print(f"Motor operation failure: {e}")
            fail_event.set()
            #motor.stop()
            return


class Orchestrator:
    """Top class state object"""
    state: State = State.INIT

    def __init__(self) -> None:
        self.state = State.INIT
        self.motor = None
        self.sensor = None
        self.channel_idx=0
        self.read_list=[]
        self.ref_list=[]

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

    def launch(self, config: dict[str, str] | None = None):
        try:
            s_config = (
                SensorsConfig() if config is None else SensorsConfig.from_config(config)
            )
            self.sensors = Sensors(s_config)
            m_config = MotorConfig() if config is None else MotorConfig.from_config(config)
            self.motor = Motor(m_config)
            #self.motor.stop()
        except Exception as e:
            print("::::Orchestrator launch failed!:::::")
            self.eval_state(init_ok=False)
            # self.state = State.ERROR
        self.state=State.READY
        print("launced")

    def start_calibration(self, channel: int):
        self.state = State.CALIBRATING
        self.channel_idx=channel
        self.sensors.calibrate_filter(channel)


    def ref_point(self, value: float):
        if self.state == State.CALIBRATING:
            print("CHANNEL",self.channel_idx)
            read = self.sensors.read_channel_raw(self.channel_idx)
            print(value,read)
            self.ref_list.append(value)
            self.read_list.append(read)
        # self.eval_state()
        else:
            print("SYSTEM NOT IN CALIBRATION MODE, IGNORING.")

    def commit_calibration(self, channel: int):
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
        self.read_list=[]
        self.ref_list=[]
        self.state = State.READY

    def dump_config(self, path: Path):
        cfg={}
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
            #self.motor.start()
            print("started")
        else:
            print(f"Trying to start on state {self.state.name}, ignoring.")

    def motor_vel(self):
        if self.state == State.READY:
            self.motor.vel_ctrl()
        else:
            print("Trying to set velocity outside READY state, ignoring")
        # self.sensors_cmd.put("START")

    def motor_a_factor(self, value):
        pass

    def print_data(self):
        """Outputs buffered data to the given file. May fail."""
        while not self.sensors_out.empty():
            data = self.sensors_out.get()
            if data is None:
                break
            else:
                print(data)

    def get_data(self,channel=3) -> int:
        """Outputs buffered data to the given file. May fail."""
        return self.sensors.read_channel(channel)


    def write_data(self, file: TextIOWrapper):
        """Outputs buffered data to the given file. May fail."""
        w = csv.writer(file)
        while not self.sensors_out.empty():
            data = self.sensors_out.get()
            if data is None:
                break
            else:
                w.writerow([data])
