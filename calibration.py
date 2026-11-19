"""Routine that calibrates the orchestrator and saves the config to a file"""

import time
from orchestrator import Orchestrator,parse_config
import numpy as np
from collections import deque
import matplotlib.animation as animation
import matplotlib.pyplot as plt

CONFIG_FILE="config.txt"
cfg=parse_config(CONFIG_FILE)
#cfg=None
orc = Orchestrator()
orc.launch(cfg)
orc.stop()

print("CALLIBRATION CH-1:")
orc.start_calibration(1)
input("SET TO ZERO")
orc.ref_point(0)

input("SET TO TEN")
orc.ref_point(10)
orc.commit_calibration(1)


print("CALLIBRATION CH-3:")
orc.start_calibration(3)
input("SET TO ZERO")
orc.ref_point(0)

input("SET TO TEN")
orc.ref_point(10)
orc.commit_calibration(3)

orc.dump_config(CONFIG_FILE)
