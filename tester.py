import time
from orchestrator import Orchestrator
import numpy as np
from collections import deque
import matplotlib.animation as animation
import matplotlib.pyplot as plt

orc = Orchestrator()
orc.launch()
# orc.stop()
# orc.start_calibration(1)
# orc.motor_vel()


win_size = 64
CHANNEL = 3

# orc.start_calibration(CHANNEL)
# time.sleep(1)
# input("SET TO ZERO")
# orc.ref_point(0)
# input("SET TO TEN")
# orc.ref_point(10)
# orc.commit_calibration(CHANNEL)

input("ENTER TO CONTINUE")
orc.start()
# orc.update()


input("ENTER TO CONTINUE")
orc.stop()
time.sleep(5)
exit()

for i in range(1000):
    read_val = orc.get_data()
    print(read_val)


def eval_value():
    read_val = orc.get_data()
    # read_val=np.average(buff)
    # amplitude=np.max(buff)-np.min(buff)
    # print(f"LIMITS {np.max(buff)} , {np.min(buff)}")
    # print(f" ADJUSTED VALUE: {read_val}, amplitude over window :{amplitude}")
    return read_val


# --- Configuration ---
WINDOW_SIZE = 200  # Number of points visible on screen
INTERVAL_MS = 30  # Refresh rate in milliseconds (~33 FPS)

# Rolling buffer initialized with zeros
data = deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
start_time = time.time()


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

# plt.tight_layout()
# plt.show()
orc.stop()
time.sleep(5)
