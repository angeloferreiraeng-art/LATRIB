import time
from orchestrator import Orchestrator, parse_config
import numpy as np
from collections import deque
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def run(orc):
    orc.commit_profile()
    orc.start()
    print("state:", orc.check_state())

    # rig.set_channel(
    def eval_value():
        orc.reset()
        read_val = orc.read_channel(3)
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

    start_time = time.perf_counter()

    flags = [False, False]

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
    orc.stop()

cfg=parse_config("config.txt")
orc = Orchestrator()
#print(orc.read_channel(3))
#print(cfg)
run(orc)

    
