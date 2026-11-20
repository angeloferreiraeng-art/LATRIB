import time
from orchestrator import Orchestrator, parse_config
import numpy as np
from collections import deque
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from simulator import HardwareSim
from orchestrator import Orchestrator

with HardwareSim() as rig:
    # rig.start()
    orc = Orchestrator(rig.make_config())
    orc.start()
    print("state:", orc.check_state())
    orc.start_calibration(1)

    # rig.set_channel(
    def eval_value():
        read_val = orc.read_channel(1)
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

    def update(_frame):
        val = eval_value()
        data.append(val)
        line.set_ydata(data)
        WAIT = 0.1
        # Auto-scale Y axis smoothly with a small margin
        ymin, ymax = min(data), max(data)
        margin = max((ymax - ymin) * 0.1, 0.1)
        ax.set_ylim(ymin - margin, ymax + margin)
        calibrated = False
        delta = time.perf_counter() - start_time
        if delta > 3 and delta < 4 and not calibrated:
            rig.set_channel(1, 300, 1)
            time.sleep(WAIT)
        if delta > 4 and delta < 5 and not calibrated:
            # orc.ref_point(0)
            time.sleep(WAIT)
        if delta > 5 and delta < 6 and not calibrated:
            rig.set_channel(1, 600, 1)
            time.sleep(WAIT)
        if delta > 7 and delta < 8 and not calibrated:
            # orc.ref_point(10)
            time.sleep(WAIT)
        if delta > 8 and delta < 9 and not calibrated:
            calibrated = True
            # orc.commit_calibration()
            time.sleep(WAIT)
        if delta > 10:
            rig.set_channel(1, 450, 10)

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
