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
    STOP = 3


@dataclass
class Command:
    code: str
    value: int


def peak_integral(a, b):
    """Calcs the % of the area of the given peak function compared to a sine wave"""
    p1 = a * (np.cos(0) - np.cos(np.pi))
    p2 = (1 - a) * np.sqrt(np.pi) * math.gamma((b + 1) / 2) / math.gamma((b + 2) / 2)
    return (p1 + p2) / 2


def peak_generator(omega, a, b):
    """Generates a function that outputs the speed curve with the given factors"""
    f = lambda x: (
        np.sign(np.sin(omega * x))
        * (a * np.abs(np.sin(omega * x)) + (1 - a) * (np.abs(np.sin(omega * x)) ** b))
    )
    return f


def _pop_from(queue: Queue) -> str | None:
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
                case Command(code="VEL_CTRL"):
                    modifier = peak_integral(a, b)
                    max_val = 0.5 * STROKE_SIZE / modifier
                    f = peak_generator(omega, a, b)
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
        sensors.set_nonblocking()
    except Exception as e:
        print(f"Sensors initialization failure: {e}")
        fail_event.set()
        return

    # acquisition loop
    reading = False
    # sensors.acq_start()
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
                    channel=new_cmd.value
                    sensors.calibrate_filter(channel)
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
    motor_cmd: Queue = Queue()
    motor_out: Queue = Queue()

    sensors_fail: threading.Event = threading.Event()
    sensors_cmd: Queue = Queue()
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
        self, init_ok: bool = False, start_on: bool = False, stop_on: bool = False
    ):
        """Receives all system signals for the grafcet of the system and
        evaluate the proper transition for the state machine
        """
        state = self.state
        match state:
            case State.ERROR:
                return
            case State.INIT:
                if init_ok:
                    self.state = State.READY
                else:
                    self.state = State.ERROR
            case State.READY:
                if not stop_on and start_on:
                    self.motor_cmd.put("START")
                    self.sensors_cmd.put("START")
                    self.state = State.RUNNING
                    return
            case State.RUNNING:
                if stop_on and not start_on:
                    self.motor_cmd.put("STOP")
                    self.sensors_cmd.put("STOP")
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

    def stop(self):
        self.motor_cmd.put("STOP")
        self.sensors_cmd.put("STOP")
        self.eval_state(stop_on=True)
        # self.state = State.STOP

    def start(self):
        self.motor_cmd.put("START")
        self.sensors_cmd.put("START")
        self.eval_state(start_on=True)
        # self.state = State.RUNNING

    def motor_vel(self):
        self.motor_cmd.put("VEL_CTRL")
        # self.sensors_cmd.put("START")

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
