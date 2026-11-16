from motor import Motor, MotorConfig
from sensors import Sensors,SensorsConfig
import time
import numpy as np


win_size=64
CHANNEL=1
config = SensorsConfig()#MotorConfig()
sensor = Sensors(config)#Motor(config)
sensor.stop()

sensor.start()

for i in range (10):
    memory=[]
    for i in range(win_size):
        read=sensor.read_channel(CHANNEL)
        memory.append(read)
    read_val=np.average(memory)
    amplitude=np.max(memory)-np.min(memory)
    print(f"LIMITS {np.max(memory)} , {np.min(memory)}")
    print(f" ADJUSTED VALUE: {read_val}, amplitude over window :{amplitude}")

sensor.calibrate_filter(CHANNEL,max_cutoff=25)


input("ENTER TO CONTINUE")
for i in range (10):
    memory=[]
    for i in range(win_size):
        channels=sensor.read()
        memory.append(channels[CHANNEL])
    read_val=np.average(memory)
    amplitude=np.max(memory)-np.min(memory)
    print(f"LIMITS {np.max(memory)} , {np.min(memory)}")
    print(f" ADJUSTED VALUE: {read_val}, amplitude over window :{amplitude}")


#exit()

read_list=[]
target_list=[]
input("SET TO ZERO")
memory=[]
for i in range(win_size):
    channels=sensor.read()
    memory.append(channels[CHANNEL])
read_val=np.average(memory)
print(read_val)
time.sleep(1)
target_list.append(0.0)
read_list.append(read_val)

input("SET TO TEN")
memory=[]
for i in range(win_size):
    channels=sensor.read()
    memory.append(channels[CHANNEL])
read_val=np.average(memory)
print(read_val)
time.sleep(1)
target_list.append(10.0)
read_list.append(read_val)

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
sensor.set_converter(CHANNEL, a, b)


input("ENTER TO CONTINUE")
def eval_value():
    memory=[]
    for i in range(win_size):
        channels=sensor.read()
        memory.append(channels[CHANNEL])
    read_val=np.average(memory)
    amplitude=np.max(memory)-np.min(memory)
    print(f"LIMITS {np.max(memory)} , {np.min(memory)}")
    print(f" ADJUSTED VALUE: {read_val}, amplitude over window :{amplitude}")
    return read_val


from collections import deque
import math
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
WINDOW_SIZE = 200  # Number of points visible on screen
INTERVAL_MS = 30  # Refresh rate in milliseconds (~33 FPS)

# Rolling buffer initialized with zeros
data = deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
start_time = time.time()


def read_value() -> float:
    """Replace this with your actual sensor/socket/filter read."""
    t = time.time() - start_time
    return math.sin(2 * math.pi * 0.5 * t) + np.random.normal(0, 0.15)


# --- Setup Plot ---
fig, ax = plt.subplots(figsize=(8, 4))
(line,) = ax.plot(range(WINDOW_SIZE), data, color="#007acc", linewidth=1.8)
ax.set_title("Live Signal Stream")
ax.set_xlabel("Samples")
ax.set_ylabel("Value")
ax.grid(True, alpha=0.3)


def update(_frame):
    val = eval_value()
    data.append(val)
    line.set_ydata(data)

    # Auto-scale Y axis smoothly with a small margin
    ymin, ymax = min(data), max(data)
    margin = max((ymax - ymin) * 0.1, 0.1)
    ax.set_ylim(ymin - margin, ymax + margin)
    return (line,)


ani = animation.FuncAnimation(
    fig,
    update,
    interval=INTERVAL_MS,
    blit=False,  # Set True if Y-axis limits are fixed for even higher FPS
    cache_frame_data=False,
)

plt.tight_layout()
plt.show()



