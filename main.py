import time
from orchestrator import Orchestrator, parse_config
import numpy as np
from collections import deque
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from simulator import HardwareSim


def run(orc):
    orc.commit_profile()
    # orc.start()
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

    def cal():
        WAIT = 0.1
        delta = time.perf_counter() - start_time
        if delta < 3 and not flags[1]:
            flags[0], flags[1] = False, True
            orc.start_calibration(3)
        if delta > 3 and delta < 4 and not flags[0]:
            flags[0], flags[1] = True, False
            rig.set_channel(3, 300, 1)
            # orc.reset()
            # time.sleep(WAIT)
        if delta > 4 and delta < 5 and not flags[1]:
            flags[0], flags[1] = False, True
            orc.ref_point(0)
            # time.sleep(WAIT)
        if delta > 5 and delta < 6 and not flags[0]:
            flags[0], flags[1] = True, False
            rig.set_channel(3, 600, 1)
            # orc.reset()
            # time.sleep(WAIT)
        if delta > 7 and delta < 8 and not flags[1]:
            flags[0], flags[1] = False, True
            orc.ref_point(10)
            # time.sleep(WAIT)
        if delta > 8 and delta < 9 and not flags[0]:
            flags[0], flags[1] = True, False
            orc.commit_calibration()
            orc.dump_config("config.txt")
            # orc.start()

    def update(_frame):
        val = eval_value()
        print(val)
        data.append(val)
        line.set_ydata(data)
        # Auto-scale Y axis smoothly with a small margin
        ymin, ymax = min(data), max(data)
        margin = max((ymax - ymin) * 0.1, 0.1)
        ax.set_ylim(ymin - margin, ymax + margin)
        # cal()
        t = time.perf_counter()
        if t - start_time > 10:
            orc.motor.stop()
        return (line,)

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=INTERVAL_MS,
        blit=False,  # Set True if Y-axis limits are fixed for even higher FPS
        cache_frame_data=False,
    )
    orc.start_raw()
    plt.tight_layout()
    plt.show()
    orc.stop()


# orc = Orchestrator()
# cfg=parse_config("config.txt")
# run(orc)
# exit()
with HardwareSim() as rig:
    # rig.start()
    rig.set_channel(1, 500, 10)
    orc = Orchestrator(rig.make_config())
    run(orc)
