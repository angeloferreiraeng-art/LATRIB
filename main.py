import time

from sensors import Sensors,SensorsConfig

s_config = SensorsConfig()
sensors = Sensors(s_config)

sensors.start()
while True:
    v=sensors.read_channel(3)
    print(v)

