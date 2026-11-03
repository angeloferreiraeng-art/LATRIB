import threading
import time
import numpy as np

from motor import Motor, MotorConfig
from dlg400 import Sensors

import dlg400 as dlg


def try_loop(func):
    N_TRIES = 5
    while N_TRIES:
        try:
            return func()
        except Exception as e:
            print(f"THREW Exception {e}, retrying operation")
            N_TRIES = N_TRIES - 1


def run_motor():
    while True:
        print("MOTOR THREAD LAUNCHED")
        time.sleep(1)
    NUM_SEGMENTS = 16
    SEG_TIME = 1
    times = np.ones(NUM_SEGMENTS) * SEG_TIME
    period = np.sum(times)
    motor = Motor()

    angular_speed = 2 * np.pi / (period)
    print("freq:", angular_speed)
    MAX_VAL = 50

    f_sine = lambda i: 100 * MAX_VAL * np.sin(angular_speed * i)

    r_sine = lambda i: 0.1 * np.sin(angular_speed * i)

    try_loop(lambda: motor.stop())

    # motor.vel_ctrl(test,times)
    # motor.vel_ctrl(f_sine,times)
    motor.pos_ctrl(r_sine, times)

    try_loop(lambda: motor.start())


def run_dlg():
    DLG_ARGS = None
    cfg = dlg.parse_args(DLG_ARGS)
    if cfg is None:
        dlg.print_usage()
        exit()
    dlg.run(cfg)


config = {"MOTOR_PORT": "shit", "MOTOR_ID": 1, "MOTOR_BAUD_RATE": 110000}

MotorConfig.from_config(config)
print("worked")

# m_thread = threading.Thread(target=run_motor, daemon=True)
# m_thread.start()

# e_thread = threading.Thread(target=run_dlg(), daemon=True)
# e_thread.start()

# GUI code goes here.
# time.sleep(5)
# print("no sleep")
