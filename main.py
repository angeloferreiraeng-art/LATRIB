import time

from orchestrator import Orchestrator

config = {"MOTOR_PORT": "shit", "MOTOR_ID": 1, "MOTOR_BAUD_RATE": 110000}


orc = Orchestrator()
orc.launch()
orc.stop()
#orc.start_calibration(1)
orc.motor_vel()
orc.start()
#orc.update()
#exit()
#time.sleep(50)
for i in range(10000):
    pass
orc.stop()
time.sleep(5)
exit()

