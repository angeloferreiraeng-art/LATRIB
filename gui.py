"""Minimal supervisory UI: start/stop the DLG logger over IPC and plot CoF (CH1 / normal force) live."""

import os
import time
import tkinter as tk
from tkinter import ttk
from collections import deque
from pathlib import Path

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from orchestrator import Orchestrator, State
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


def ask_value(parent, title="Enter value", prompt="Value:", default=""):
    """Show a modal popup with one entry field and an OK button.
    Returns the entered string, or None if cancelled."""
    win = tk.Toplevel(parent)
    win.title("Informe o Canal")
    win.transient(parent)
    win.grab_set()
    win.resizable(False, False)

    ttk.Label(win, text=prompt).pack(padx=16, pady=(16, 4))

    ch_frame = ttk.LabelFrame(win, text="Numero do Canal")
    ch_frame.pack(fill="x", padx=10, pady=8)

    ch_entry = ttk.Entry(ch_frame, width=20)
    ch_entry.insert(0, default)
    ch_entry.pack(padx=16)
    ch_entry.focus_set()
    ch_entry.select_range(0, "end")

    name_frame = ttk.LabelFrame(win, text="Identificação:")
    name_frame.pack(fill="x", padx=10, pady=8)
    name_entry = ttk.Entry(name_frame, width=20)
    name_entry.insert(0, default)
    name_entry.pack(padx=16)
    name_entry.focus_set()
    name_entry.select_range(0, "end")

    result = {"channel": None, "name": "UNNAMED"}

    def on_ok():
        # TODO: TRY CATCH BLOCK HERE.
        result["value"] = ch_entry.get()
        result["name"] = name_entry.get()
        win.destroy()

    def on_cancel():
        win.destroy()

    ttk.Button(win, text="OK", command=on_ok).pack(pady=16)

    win.bind("<Return>", lambda e: on_ok())
    win.bind("<Escape>", lambda e: on_cancel())
    win.protocol("WM_DELETE_WINDOW", on_cancel)

    parent.wait_window(win)
    return result


class GUI:
    def __init__(self, config):
        self.orch = Orchestrator(config)
        self.root = tk.Tk()
        self.root.title("Minimal supervisory")

        # ==================================================================
        # PERSISTENT HEADER (shared across all tabs)
        # ==================================================================
        header = tk.Frame(self.root)
        header.pack(fill="x", padx=6, pady=6)

        # --- left: state indicator ---
        state_frame = tk.Frame(header)
        state_frame.pack(side="left", padx=(0, 12))

        tk.Label(state_frame, text="Orchestrator state:").pack(side="left")
        self.state_canvas = tk.Canvas(
            state_frame, width=140, height=40, bg="white", highlightthickness=0
        )
        self.state_canvas.pack(side="left", padx=6)
        self.orch_state_rect = self.state_canvas.create_rectangle(
            2, 2, 138, 38, fill="gray", outline="black", width=2
        )
        self.orch_state_text = self.state_canvas.create_text(
            70, 20, text="NONE", fill="white", font=("Helvetica", 12, "bold")
        )

        # --- right: countdown timer ---
        self.timer_label = tk.Label(
            header,
            text=format_mmss(TIMER_SECONDS),
            font=("Helvetica", 16, "bold"),
            width=6,
        )
        self.timer_label.pack(side="right", padx=6)
        tk.Label(header, text="Time left:").pack(side="right")

        # ==================================================================
        # TABS
        # ==================================================================
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.tab_supervision = ttk.Frame(self.notebook)
        self.tab_calibration = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_supervision, text="Supervision")
        self.notebook.add(self.tab_calibration, text="Calibration")

        self._build_supervision_tab()
        self._build_calibration_tab()

        # ==================================================================
        # ORCHESTRATOR + STATE
        # ==================================================================
        self._running = False
        self._after_id = None
        self._t0 = None
        self._timer_t0 = None  # monotonic start of the countdown
        self._t_data = deque(maxlen=WINDOW_SIZE)
        self._y_data = deque(maxlen=WINDOW_SIZE)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------
    # SUPERVISION TAB
    # ------------------------------------------------------------------
    def _build_supervision_tab(self):
        tab = self.tab_supervision

        # ----- top control bar -----
        bar = tk.Frame(tab)
        bar.pack(fill="x", padx=6, pady=6)

        tk.Label(bar, text="Output path:").pack(side="left")
        self.output_path = tk.Entry(bar, width=32)
        self.output_path.insert(0, OUT_CSV)
        self.output_path.pack(side="left", padx=6)

        tk.Button(bar, text="Start", command=self.start).pack(side="left", padx=2)
        tk.Button(bar, text="Stop", command=self.stop).pack(side="left", padx=2)
        tk.Button(bar, text="Velocity mode", command=self.velocity_mode).pack(
            side="left", padx=2
        )

        # ----- plot -----
        fig = Figure(figsize=(7, 3.5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        self.data = deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
        (self.line_1,) = self.ax.plot(
            range(WINDOW_SIZE), self.data, color="#007acc", linewidth=1.8
        )
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("CoF [-]")
        self.ax.set_xlim(0, WINDOW_SIZE)
        self.ax.grid(alpha=0.5)

    # ------------------------------------------------------------------
    # CALIBRATION TAB (placeholder)
    # ------------------------------------------------------------------
    def _build_calibration_tab(self):
        tab = self.tab_calibration
        pad = {"padx": 8, "pady": 4}

        # --- config file ---
        configs = ttk.LabelFrame(tab, text="Configuration")
        configs.pack(fill="x", padx=10, pady=8)

        ttk.Label(configs, text="Config path:").grid(row=0, column=0, sticky="w", **pad)
        self.cfg_path_entry = ttk.Entry(configs, width=10).grid(
            row=0, column=1, sticky="w", **pad
        )
        ttk.Button(configs, text="Adicional Canal", command=self.add_channel).grid(
            row=0, column=2, **pad
        )

        # --- Normal force calibration ---
        lf_force = ttk.LabelFrame(tab, text="Normal force calibration")
        lf_force.pack(fill="x", padx=10, pady=8)

        ttk.Label(lf_force, text="Known force [N]:").grid(
            row=0, column=0, sticky="w", **pad
        )
        ttk.Entry(lf_force, width=10).grid(row=0, column=1, sticky="w", **pad)
        ttk.Button(lf_force, text="Capture zero").grid(row=0, column=2, **pad)
        ttk.Button(lf_force, text="Calibrate").grid(row=0, column=3, **pad)

        # --- Friction force calibration ---
        lf_fric = ttk.LabelFrame(tab, text="Friction force calibration")
        lf_fric.pack(fill="x", padx=10, pady=8)

        ttk.Label(lf_fric, text="Known CoF [-]:").grid(
            row=0, column=0, sticky="w", **pad
        )
        ttk.Entry(lf_fric, width=10).grid(row=0, column=1, sticky="w", **pad)
        ttk.Button(lf_fric, text="Capture zero").grid(row=0, column=2, **pad)
        ttk.Button(lf_fric, text="Calibrate").grid(row=0, column=3, **pad)

        # --- Populate Channels ---
        for ch in self.orch.channels():
            ch_frame = ttk.LabelFrame(tab, text=ch)
            ch_frame.pack(fill="x", padx=10, pady=8)
            ttk.Button(ch_frame, text="Começar Calibração").grid(row=0, column=0, **pad)
            ttk.Button(ch_frame, text="Terminar Calibração").grid(
                row=0, column=1, **pad
            )

            ttk.Label(ch_frame, text="Valor referência:").grid(
                row=1, column=0, sticky="w", **pad
            )
            ttk.Entry(ch_frame, width=10).grid(row=1, column=1, sticky="w", **pad)
            ttk.Button(ch_frame, text="Capturar Sinal").grid(row=1, column=2, **pad)

        # --- Action buttons ---
        actions = tk.Frame(tab)
        actions.pack(fill="x", padx=10, pady=12)
        ttk.Button(actions, text="Save calibration").pack(side="left", padx=4)
        ttk.Button(actions, text="Load calibration").pack(side="left", padx=4)
        ttk.Button(actions, text="Reset to defaults").pack(side="left", padx=4)

    # ------------------------------------------------------------------
    # TIMER
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------------------
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

    def add_channel(self):
        ask_value(self.root, "nome", "coiso")

    def calibration_mode(self):
        self.orch.start_calibration()

    def save_config(self):
        path = Path(self.cfg_path_entry.get())
        self.orch.dump_config(path)

    # ------------------------------------------------------------------
    # REDRAW LOOP
    # ------------------------------------------------------------------
    def redraw(self):
        try:
            if self._running:
                val = self.orch.read_channel(3)
                self.data.append(val)
                self._t_data.append(time.time() - self._t0)
                self._y_data.append(val)

                self.line_1.set_data(range(len(self.data)), list(self.data))
                self.ax.relim()
                self.ax.autoscale_view(scalex=False, scaley=True)
                self.canvas.draw_idle()

                self._update_timer()

            # state indicator lives in the header now
            fill = "gray"
            orc_state = self.orch.check_state()
            match orc_state:
                case State.INIT:
                    fill = "gray"
                case State.IDLE:
                    fill = "blue"
                case State.ERROR:
                    fill = "red"
                case State.STOP:
                    fill = "green"
                case State.CALIBRATING:
                    fill = "yellow"
                case State.READY:
                    fill = "green"
                case State.RUNNING:
                    fill = "yellow"
            self.state_canvas.itemconfig(self.orch_state_text, text=orc_state.name)
            self.state_canvas.itemconfig(self.orch_state_rect, fill=fill)
        except Exception as e:
            print("redraw error:", e)

        self._after_id = self.root.after(INTERVAL_MS, self.redraw)

    # ------------------------------------------------------------------
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
