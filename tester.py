from motor import Motor, MotorConfig
import time
import numpy as np

motor = None
try:
    config = MotorConfig()
    motor = Motor(config)
    motor.stop()
except Exception as e:
    print(f"Motor initialization failure: {e}")
    exit()

NUM_SEGMENTS = 4
SEG_TIME = 1
times = np.ones(NUM_SEGMENTS) * SEG_TIME
period = np.sum(times)
angular_speed = 2 * np.pi / (period)
print("freq:", angular_speed)
MAX_VAL = 10

#f_sine = lambda i: MAX_VAL * np.sin(angular_speed * i)
f_sine = lambda i: MAX_VAL * np.sin(angular_speed * i)
current_cmd = None


#motor.pos_ctrl(f_sine, times)
motor.vel_ctrl(f_sine, times)
#exit()
motor.start()

time.sleep(10)
motor.stop()