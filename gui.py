"""Minimal supervisory UI: start/stop the DLG logger over IPC and plot CoF (CH1 / normal force) live."""

import os
import time
import tkinter as tk
from collections import deque
from pathlib import Path

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from orchestrator import Orchestrator
from simulator import HardwareSim

OUT_DIR = Path.home()
FILE_NAME = "data"
OUT_CSV = os.path.join(OUT_DIR, f"{FILE_NAME}.csv")
RATE_HZ = 50

WINDOW_SIZE = 200  # points visible on the plot
INTERVAL_MS = 33  # ~30 FPS
CSV_TAIL_HZ = 50  # how often to poll the CSV (optional)

TIMER_SECONDS = 30 * 60  # countdown duration: 30 min
AUTO_STOP_AT_ZERO = False  # set True to stop the orchestrator when the timer hits 0


def format_mmss(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class GUI:
    def __init__(self, config):
        self.root = tk.Tk()
        self.root.title("Minimal supervisory")

        # ----- top bar -----
        bar = tk.Frame(self.root)
        bar.pack(fill="x", padx=6, pady=6)
        tk.Label(bar, text="Normal force [N]").pack(side="left")
        self.force_entry = tk.Entry(bar, width=8)
        self.force_entry.insert(0, "1")
        self.force_entry.pack(side="left", padx=6)

        tk.Label(bar, text="Output path:").pack(side="left")
        self.output_path = tk.Entry(bar, width=32)
        self.output_path.insert(0, OUT_CSV)
        self.output_path.pack(side="left", padx=6)

        tk.Button(bar, text="Start", command=self.start).pack(side="left", padx=2)
        tk.Button(bar, text="Stop", command=self.stop).pack(side="left", padx=2)

        tk.Button(bar, text="Velocity mode", command=self.velocity_mode).pack(
            side="left", padx=2
        )

        # countdown timer (right side of the bar)
        self.timer_label = tk.Label(
            bar,
            text=format_mmss(TIMER_SECONDS),
            font=("Helvetica", 16, "bold"),
            width=6,
        )
        self.timer_label.pack(side="right", padx=6)
        tk.Label(bar, text="Time left:").pack(side="right")

        # ----- plot -----
        fig = Figure(figsize=(7, 3.5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.data = deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
        (self.line_1,) = self.ax.plot(
            range(WINDOW_SIZE), self.data, color="#007acc", linewidth=1.8
        )
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("CoF [-]")
        self.ax.set_xlim(0, WINDOW_SIZE)
        self.ax.grid(alpha=0.5)

        # ----- state indicator -----
        self.state_canvas = tk.Canvas(
            self.root, width=400, height=160, bg="white", highlightthickness=0
        )
        self.state_canvas.pack(fill="x")
        square_size = 120
        x1 = (400 - square_size) // 2
        y1 = 20
        x2 = x1 + square_size
        y2 = y1 + square_size
        self.orch_state_rect = self.state_canvas.create_rectangle(
            x1, y1, x2, y2, fill="gray", outline="black", width=2
        )
        self.orch_state_text = self.state_canvas.create_text(
            (x1 + x2) // 2,
            (y1 + y2) // 2,
            text="NONE",
            fill="white",
            font=("Helvetica", 16, "bold"),
        )

        # ----- orchestrator -----
        self.orch = Orchestrator(config)

        # ----- state -----
        self._running = False
        self._after_id = None
        self._t0 = None
        self._timer_t0 = None  # monotonic start of the countdown
        self._t_data = deque(maxlen=WINDOW_SIZE)
        self._y_data = deque(maxlen=WINDOW_SIZE)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------
    def _normal_force(self):
        try:
            f = float(self.force_entry.get())
            return f if f > 0 else 1.0
        except ValueError:
            return 1.0

    def _reset_timer(self):
        self._timer_t0 = None
        self.timer_label.config(text=format_mmss(TIMER_SECONDS), fg="black")

    def _update_timer(self):
        if self._timer_t0 is None:
            return
        remaining = TIMER_SECONDS - (time.monotonic() - self._timer_t0)
        self.timer_label.config(
            text=format_mmss(remaining + 0.999),  # round up so 00:00 only at zero
            fg="red" if remaining <= 0 else "black",
        )
        if remaining <= 0 and AUTO_STOP_AT_ZERO and self._running:
            self.stop(reset_timer=False)

    def start(self):
        if not self.orch.start():
            return

        self._t0 = time.time()
        self._t_data.clear()
        self._y_data.clear()
        self.data.clear()
        self.data.extend([0.0] * WINDOW_SIZE)

        self._running = True
        self._reset_timer()
        self._timer_t0 = time.monotonic()

    def stop(self, reset_timer=True):
        self.orch.stop()
        self._running = False
        if reset_timer:
            pass  # self._reset_timer()
        else:
            self._timer_t0 = None  # freeze the display (at 00:00)

    def velocity_mode(self):
        self.orch.commit_profile()

    def redraw(self):
        try:
            if self._running:
                val = self.orch.read_channel(1) / self._normal_force()
                self.data.append(val)
                self._t_data.append(time.time() - self._t0)
                self._y_data.append(val)

                self.line_1.set_data(range(len(self.data)), list(self.data))
                self.ax.relim()
                self.ax.autoscale_view(scalex=False, scaley=True)
                self.canvas.draw_idle()

                self._update_timer()

            state = self.orch.check_state()
            self.state_canvas.itemconfig(self.orch_state_text, text=state)
            fill = "red" if state == "ERROR" else "green"
            self.state_canvas.itemconfig(self.orch_state_rect, fill=fill)
        except Exception as e:
            print("redraw error:", e)

        self._after_id = self.root.after(INTERVAL_MS, self.redraw)

    def on_close(self):
        try:
            self.stop()
        finally:
            if self._after_id is not None:
                self.root.after_cancel(self._after_id)
            self.root.destroy()

    def run(self):
        self.redraw()  # schedule first frame
        self.root.mainloop()


def main():
    with HardwareSim() as rig:
        rig.set_channel(1, 500, 10)
        gui = GUI(rig.make_config())
        gui.run()


if __name__ == "__main__":
    main()
