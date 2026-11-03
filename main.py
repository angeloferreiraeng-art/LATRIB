import time

from orchestrator import Orchestrator

config = {"MOTOR_PORT": "shit", "MOTOR_ID": 1, "MOTOR_BAUD_RATE": 110000}


orc = Orchestrator()
orc.launch()

while True:
    orc.update()
    time.sleep(2)
