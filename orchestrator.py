from dataclasses import dataclass
from io import FileIO, TextIOWrapper
import threading
import time
import csv
import numpy as np
from queue import Queue, Empty, Full

from motor import Motor, MotorConfig
from sensors import Sensors, SensorsConfig
from enum import Enum
import math


class State(Enum):
    INIT = 0
    ERROR = -1
    READY = 1
    RUNNING = 2
    CALIBRATING = 3
    STOP = 4


@dataclass
class Command:
    code: str
    value: int | float


def peak_integral(a: float, b: float) -> float:
    """Calcs the % of the area of the given peak function compared to a sine wave"""
    p1 = a * (np.cos(0) - np.cos(np.pi))
    p2 = (1 - a) * np.sqrt(np.pi) * math.gamma((b + 1) / 2) / math.gamma((b + 2) / 2)
    return (p1 + p2) / 2


def peak_generator(omega: float, a: float, b: float, max_value: float = 1.0):
    """Generates a function that outputs the speed curve with the given factors"""
    f = lambda x: (
        max_value
        * np.sign(np.sin(omega * x))
        * (a * np.abs(np.sin(omega * x)) + (1 - a) * (np.abs(np.sin(omega * x)) ** b))
    )
    return f


def _pop_from(queue: Queue) -> Command | None:
    """Helper function to return an optional result from the queue"""
    if queue.empty():
        return None
    else:
        return queue.get()


def _run_motor(fail_event: threading.Event, cmd_buffer: Queue, out_buffer: Queue):
    motor = None
    try:
        config = MotorConfig()
        motor = Motor(config)
        motor.stop()
    except Exception as e:
        print(f"Motor initialization failure: {e}")
        fail_event.set()
        return

    NUM_SEGMENTS = 16
    SEG_TIME = 1
    times = np.ones(NUM_SEGMENTS) * SEG_TIME
    period = np.sum(times)
    STROKE_SIZE = 10
    omega = 2 * np.pi / (period)
    a = 0
    b = 1

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
                    max_value = 0.5 * STROKE_SIZE / modifier
                    f = peak_generator(omega, a, b, max_value=max_value)
                    motor.vel_ctrl(f, times)
                case Command(code="POS_CTRL"):
                    raise RuntimeError("NOT IMPLEMENTED!")
                case Command(code="ANALOG_MODE"):
                    motor.analog_mode()
                case Command(code="START"):
                    motor.start()
                case Command(code="STOP"):
                    motor.stop()
                case None:
                    pass
        except Exception as e:
            print(f"Motor operation failure: {e}")
            fail_event.set()
            motor.stop()
            return


# top level scope for the dlg400 sensor thread
def _run_sensors(fail_event: threading.Event, cmd_buffer: Queue, out_buffer: Queue):
    try:
        config = SensorsConfig()
        sensors = Sensors(config)
    except Exception as e:
        print(f"Sensors initialization failure: {e}")
        fail_event.set()
        return

    # acquisition loop
    reading = False
    calibrating = False
    target_list = []
    read_list = []
    channel_idx: int = 0
    while True:
        try:
            new_cmd = _pop_from(cmd_buffer)
            match new_cmd:
                case Command(code="START"):
                    sensors.start()
                    reading = True
                case Command(code="STOP"):
                    sensors.stop()
                    reading = False
                case Command(code="CALIBRATE"):
                    channel_idx = int(new_cmd.value)
                    sensors.calibrate_filter(channel_idx)
                case Command(code="REF_POINT"):
                    read = sensors.read_channel(channel_idx)
                    target = new_cmd.value
                    read_list.append(read)
                    target_list.append(target)
                case Command(code="CAL_COMMIT"):  # commits to calibration set
                    x = np.array(read_list, dtype=float)
                    y = np.array(target_list, dtype=float)
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
                    sensors.set_converter(channel_idx, a, b)
                case None:
                    pass
            if reading:
                frame = sensors.read()
                out_buffer.put(frame)

        except Exception as e:
            print(f"Sensors failed: {e}")
            fail_event.set()
            return


def global_config():
    config = {"MOTOR_PORT": "COM5", "MOTOR_ID": 1, "MOTOR_BAUD_RATE": 19200}
    return config


class Orchestrator:
    """Top class state object"""

    state: State = State.INIT
    motor_thread: threading.Thread | None = None
    sensor_thread: threading.Thread | None = None

    motor_fail: threading.Event = threading.Event()
    motor_cmd: Queue[Command] = Queue()
    motor_out: Queue = Queue()

    sensors_fail: threading.Event = threading.Event()
    sensors_cmd: Queue[Command] = Queue()
    sensors_out: Queue = Queue()

    def __init__(self) -> None:
        self.state = State.INIT
        self.motor_thread = None
        self.sensor_thread = None

        self.motor_fail = threading.Event()
        self.motor_cmd = Queue()
        self.motor_out = Queue()

        self.sensors_fail = threading.Event()
        self.sensors_cmd = Queue()
        self.sensors_out = Queue()

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
                    cmd = Command(code="START", value=0)
                    self.motor_cmd.put(cmd)
                    self.sensors_cmd.put(cmd)
                    self.state = State.RUNNING
                    return
            case State.RUNNING:
                if stop_on and not start_on:
                    cmd = Command(code="STOP", value=0)
                    self.motor_cmd.put(cmd)
                    self.sensors_cmd.put(cmd)
                    self.state = State.STOP

    def check_state(self) -> str:
        return self.state

    def launch(self):
        try:
            self.motor_thread = threading.Thread(
                target=_run_motor,
                args=(self.motor_fail, self.motor_cmd, self.motor_out),
                daemon=True,
            )
            self.motor_thread.start()

            self.sensor_thread = threading.Thread(
                target=_run_sensors,
                args=(self.sensors_fail, self.sensors_cmd, self.sensors_out),
                daemon=True,
            )
            self.sensor_thread.start()
            self.eval_state(init_ok=True)
            # self.state = State.READY
        except Exception as e:
            print("::::Orchestrator launch failed!:::::")
            self.eval_state(init_ok=False)
            # self.state = State.ERROR

    def start_calibration(self, channel: int):
        cmd = Command(code="CALIBRATE", value=channel)
        self.sensors_cmd.put(cmd)
        self.state = State.CALIBRATING
        # self.eval_state()

    def ref_point(self, value: float):
        if self.state == State.CALIBRATING:
            cmd = Command(code="REF_POINT", value=value)
            self.sensors_cmd.put(cmd)
        else:
            print("SYSTEM NOT IN CALIBRATION MODE, IGNORING.")

    def commit_calibration(self, channel: int):
        cmd = Command(code="CALIBRATE", value=channel)
        self.sensors_cmd.put(cmd)
        self.state = State.READY

    def stop(self):
        self.eval_state(stop_on=True)
        # self.state = State.STOP

    def start(self):
        self.eval_state(start_on=True)
        # self.state = State.RUNNING

    def motor_vel(self):
        cmd = Command(code="VEL_CTRL", value=0)
        self.motor_cmd.put(cmd)
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

    def write_data(self, file: TextIOWrapper):
        """Outputs buffered data to the given file. May fail."""
        w = csv.writer(file)
        while not self.sensors_out.empty():
            data = self.sensors_out.get()
            if data is None:
                break
            else:
                w.writerow([data])

    def update(self):
        if self.motor_fail.is_set() or self.sensors_fail.is_set():
            print("FOUND ERROR DURING UPDATE")
            self.state = State.ERROR
