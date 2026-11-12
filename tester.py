from motor import Motor, MotorConfig
from sensors import Sensors,SensorsConfig
import time
import numpy as np

sensor = None

config = SensorsConfig()#MotorConfig()
sensor = Sensors(config)#Motor(config)
sensor.stop()
    #config=MotorConfig()
    #motor=Motor(config)





sensor.start()
time.sleep(1)
win_size=128
idx=0
memory=[]
while True:
    for i in range(win_size):
        channels=sensor.read()
        memory.append(channels[3])
    print(np.average(memory))
    memory=[]
time.sleep(1)
sensor.stop()
exit()

exit()
NUM_SEGMENTS = 8
SEG_TIME = 1
times = np.ones(NUM_SEGMENTS) * SEG_TIME
half_period = np.sum(times)


angular_speed = 2 * np.pi / (2*half_period)
MAX_VAL = 30
A=0.1
b=5

def peak_func(i):
    sine_x=np.sin(angular_speed * i)
    return MAX_VAL *np.sign(sine_x)*(A*np.abs(sine_x)+(1-A)*np.abs(sine_x)**b)


#f_sine = lambda i: MAX_VAL * np.sin(angular_speed * i)
current_cmd = None


#motor.pos_ctrl(f_sine, times)
motor.vel_ctrl(peak_func, times)
#exit()
motor.start()

time.sleep(10)
motor.stop()
exit()




