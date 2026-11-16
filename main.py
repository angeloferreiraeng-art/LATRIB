import time

from orchestrator import Orchestrator

config = {"MOTOR_PORT": "shit", "MOTOR_ID": 1, "MOTOR_BAUD_RATE": 110000}


orc = Orchestrator()
orc.launch()
orc.stop()
orc.start_calibration(1)
orc.motor_vel()
#orc.start()
#exit()
time.sleep(100)
orc.update()
#orc.stop()
exit()

with open("data.csv", "w") as file:
    while True:
        orc.update()
        print(f"current state: {orc.check_state()}")
        orc.write_data(file)
        time.sleep(2)
