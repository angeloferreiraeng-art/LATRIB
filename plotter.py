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
    val = read_value()
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