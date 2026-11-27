import csv
import os
import time
import tkinter as tk
from tkinter import ttk
from collections import deque
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from orchestrator import Orchestrator, State, parse_config
from simulator import HardwareSim

OUT_DIR = Path.home()
FILE_NAME = "data"
OUT_CSV = os.path.join(OUT_DIR, f"{FILE_NAME}.csv")  # valor padrão do output path
RATE_HZ = 50

WINDOW_SIZE = 200  # points visible on the plot
INTERVAL_MS = 33  # ~30 FPS
CSV_TAIL_HZ = 50  # how often to poll the CSV (optional)

TIMER_SECONDS = 30 * 60  # valor padrão do tempo de ensaio: 30 min
AUTO_STOP_AT_ZERO = False  # set True to stop the orchestrator when the timer hits 0


def safe_float(value: str):
    try:
        return float(value)
    except Exception:
        return 0.0


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
        try:
            result["channel"] = int(ch_entry.get())
            result["name"] = name_entry.get() or "UNNAMED"
        except ValueError:
            # Invalid channel number — leave as None so caller can ignore
            result["channel"] = None
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
        # STATE (definido antes das abas, pois elas usam estes atributos)
        # ==================================================================
        self._running = False
        self._after_id = None
        self._t0 = None
        self._timer_t0 = None  # monotonic start of the countdown
        self._timer_total = TIMER_SECONDS  # duração congelada no start()
        self._t_data = deque(maxlen=WINDOW_SIZE)
        self._y_data = deque(maxlen=WINDOW_SIZE)
        self._csv_file = None  # output CSV handle (open while running)
        self._csv_writer = None

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
        self.orch_cal_text = self.state_canvas.create_text(
            70, 20, text="", fill="white", font=("Helvetica", 12, "bold")
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
        self.tab_plan = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_supervision, text="Supervisório")
        self.notebook.add(self.tab_calibration, text="Calibração")
        self.notebook.add(self.tab_plan, text="Plano de Ensaio")

        self._build_supervision_tab()
        self._build_calibration_tab()
        self._build_plan_tab()
        self._populate_channels()

        # sincroniza o relógio do cabeçalho com o tempo definido na aba Plano
        self._on_time_changed()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------
    # SUPERVISION TAB
    # ------------------------------------------------------------------
    def _build_supervision_tab(self):
        tab = self.tab_supervision

        # ----- top control bar -----
        bar = tk.Frame(tab)
        bar.pack(fill="x", padx=6, pady=6)

        tk.Button(bar, text="Start", command=self.start).pack(side="left", padx=2)
        tk.Button(bar, text="Stop", command=self.stop).pack(side="left", padx=2)
        tk.Button(bar, text="Velocity mode", command=self.velocity_mode).pack(
            side="left", padx=2
        )
        tk.Button(bar, text="Position mode", command=self.orch.commit_pos_profile).pack(
            side="left", padx=2
        )
        # ----- channel selector -----
        tk.Label(bar, text="Channel:").pack(side="left", padx=(12, 2))
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(
            bar, textvariable=self.channel_var, state="readonly", width=22
        )
        self.channel_combo.pack(side="left", padx=2)
        self.channel_combo.bind("<<ComboboxSelected>>", self._on_channel_selected)
        self._refresh_channel_combo()

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
        self._update_ylabel()
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
        self.cfg_path_entry = ttk.Entry(configs, width=10)
        self.cfg_path_entry.grid(row=0, column=1, sticky="w", **pad)
        ttk.Button(configs, text="Adicional Canal", command=self.add_channel).grid(
            row=0, column=2, **pad
        )
        # --- Action buttons ---
        actions = tk.Frame(tab)
        actions.pack(fill="x", padx=10, pady=12)
        ttk.Button(actions, text="Save calibration", command=self.save_config).pack(
            side="left", padx=4
        )
        ttk.Button(actions, text="Load calibration").pack(side="left", padx=4)
        ttk.Button(actions, text="Reset to defaults").pack(side="left", padx=4)

    # ------------------------------------------------------------------
    # PLAN TAB
    # ------------------------------------------------------------------
    def _build_plan_tab(self):
        tab = self.tab_plan
        pad = {"padx": 8, "pady": 4}

        # --- config file ---
        profile = ttk.LabelFrame(tab, text="Configuration")
        profile.pack(fill="x", padx=10, pady=8)

        ttk.Label(profile, text="Fator A:").grid(row=0, column=0, sticky="w", **pad)
        self.a_factor_entry = ttk.Entry(profile, width=10)
        self.a_factor_entry.grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(profile, text="Fator B:").grid(row=0, column=2, sticky="w", **pad)
        self.b_factor_entry = ttk.Entry(profile, width=10)
        self.b_factor_entry.grid(row=0, column=3, sticky="w", **pad)

        # --- tempo de ensaio (em minutos) ---
        ttk.Label(profile, text="Tempo de ensaio [min]:").grid(
            row=1, column=0, sticky="w", **pad
        )
        self.time_entry = ttk.Entry(profile, width=10)
        self.time_entry.insert(0, str(TIMER_SECONDS // 60))
        self.time_entry.grid(row=1, column=1, sticky="w", **pad)
        self.time_entry.bind("<KeyRelease>", lambda e: self._on_time_changed())

        # --- output path ---
        ttk.Label(profile, text="Output path:").grid(row=2, column=0, sticky="w", **pad)
        self.output_path = ttk.Entry(profile, width=50)
        self.output_path.insert(0, OUT_CSV)
        self.output_path.grid(row=2, column=1, columnspan=3, sticky="we", **pad)

        # Bind changes on A and B entries to redraw the plot
        self.a_factor_entry.bind("<KeyRelease>", lambda e: self._update_curve_view())
        self.b_factor_entry.bind("<KeyRelease>", lambda e: self._update_curve_view())

        # --- embedded plot area ---
        self.plan_fig = Figure(figsize=(10, 6), dpi=100)
        self.plan_ax = self.plan_fig.add_subplot(211)
        self.plan_ax_fft = self.plan_fig.add_subplot(212)
        self.plan_canvas = FigureCanvasTkAgg(self.plan_fig, master=tab)
        self.plan_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        # Initial draw (only if both fields have values)
        self.root.after(100, self._update_curve_view)

    def _update_curve_view(self):
        a = safe_float(self.a_factor_entry.get())
        b = safe_float(self.b_factor_entry.get())
        self.orch.profile_factors(a, b)
        f = self.orch.velocity_func()

        x_0 = 0
        x_1 = np.pi / self.orch.motor.get_omega()
        # --- 1. Calculate the definite integral ---
        integral_value, error = quad(f, x_0, x_1)
        print(f"Integral of f(x) from {x_0} to {x_1} = {integral_value:.6f}")
        print(f"Estimated error: {error:.2e}")

        # --- 2. Prepare data for plotting ---
        x = np.linspace(x_0, x_1, 10000)
        y = f(x)

        # --- 3. Clear and redraw the two panels ---
        self.plan_ax.clear()
        self.plan_ax_fft.clear()

        # ----- Top panel: original function -----
        self.plan_ax.plot(x, y, "b-", linewidth=2, label=r"$f(x)$")

        # Fill the actual integrated interval [x_0, x_1]
        self.plan_ax.fill_between(
            x,
            y,
            alpha=0.3,
            color="skyblue",
            label=f"Area = {integral_value:.4f}",
        )

        # Only draw the a/b markers if they fall inside the plotted range —
        # otherwise axvline forces matplotlib to expand the x-axis.
        if x_0 <= a <= x_1:
            self.plan_ax.axvline(a, color="gray", linestyle="--", alpha=0.7)
        if x_0 <= b <= x_1:
            self.plan_ax.axvline(b, color="gray", linestyle="--", alpha=0.7)
        self.plan_ax.axhline(0, color="black", linewidth=0.8)

        self.plan_ax.set_xlabel("x")
        self.plan_ax.set_ylabel("f(x)")
        self.plan_ax.set_title(
            f"Function Plot and Definite Integral on [{x_0:.2f}, {x_1:.2f}]"
        )
        self.plan_ax.legend(loc="upper left")
        self.plan_ax.grid(True, alpha=0.3)

        # Lock the x-axis to the integration interval AFTER all drawing calls,
        # so autoscaling can't rescale it when factors a/b change.
        self.plan_ax.set_xlim(x_0, x_1)

        # ----- Bottom panel: FFT (one-sided amplitude spectrum) -----
        dx = x[1] - x[0]
        N = len(y)

        Y = np.fft.fft(y)
        freqs = np.fft.fftfreq(N, d=dx)

        mask = freqs >= 0
        freqs_pos = freqs[mask]
        amp_pos = (2.0 / N) * np.abs(Y[mask])
        # DC bin is not doubled in a one-sided spectrum
        amp_pos[0] /= 2.0

        self.plan_ax_fft.plot(freqs_pos, amp_pos, "r-", linewidth=1.5)
        self.plan_ax_fft.set_xlabel("Frequency (cycles per unit x)")
        self.plan_ax_fft.set_ylabel("Amplitude")
        self.plan_ax_fft.set_title("FFT of f(x) (one-sided amplitude spectrum)")
        self.plan_ax_fft.grid(True, alpha=0.3)

        if len(amp_pos) > 1:
            peak_idx = np.argmax(amp_pos[1:]) + 1
            peak_f = freqs_pos[peak_idx]
            peak_a = amp_pos[peak_idx]
            self.plan_ax_fft.annotate(
                f"peak ≈ {peak_f:.3f}\n(period ≈ {1 / peak_f:.2f})",
                xy=(peak_f, peak_a),
                xytext=(peak_f + 0.05, peak_a * 0.8),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9,
            )

        self.plan_fig.tight_layout()
        self.plan_canvas.draw_idle()

    def _populate_channels(self):
        # Clear existing channel frames before repopulating
        for widget in getattr(self, "_channel_frames", []):
            widget.destroy()
        self._channel_frames = []

        pad = {"padx": 8, "pady": 4}
        self.channels = self.orch.channels()

        for ch in self.channels:
            ch_frame = ttk.LabelFrame(self.tab_calibration, text=ch["name"])
            ch_frame.pack(fill="x", padx=10, pady=8)
            self._channel_frames.append(ch_frame)

            ttk.Button(
                ch_frame,
                text="Começar Calibração",
                command=lambda n=ch["number"]: self.orch.start_calibration(n),
            ).grid(row=0, column=0, **pad)
            ttk.Button(
                ch_frame,
                text="Terminar Calibração",
                command=lambda: self.orch.commit_calibration(),
            ).grid(row=0, column=1, **pad)

            ttk.Label(ch_frame, text="Valor referência:").grid(
                row=1, column=0, sticky="w", **pad
            )
            ref_entry = ttk.Entry(ch_frame, width=10)
            ref_entry.grid(row=1, column=1, sticky="w", **pad)
            ttk.Button(
                ch_frame,
                text="Capturar Sinal",
                command=lambda e=ref_entry: self.orch.ref_point(int(e.get())),
            ).grid(row=1, column=2, **pad)

    # ------------------------------------------------------------------
    # CHANNEL SELECTOR
    # ------------------------------------------------------------------
    def _refresh_channel_combo(self):
        """Rebuild the dropdown options from the orchestrator's channel list."""
        channels = self.orch.channels()
        self._channel_lookup = {
            f"{ch['number']} - {ch['name']}": ch["number"] for ch in channels
        }
        self.channel_combo["values"] = list(self._channel_lookup.keys())
        if self.channel_combo["values"]:
            if self.channel_var.get() not in self._channel_lookup:
                self.channel_var.set(self.channel_combo["values"][0])
        else:
            self.channel_var.set("")
        if hasattr(self, "ax"):
            self._update_ylabel()

    def _on_channel_selected(self, event=None):
        """Reset the plot buffers when the user switches channels."""
        self.data.clear()
        self.data.extend([0.0] * WINDOW_SIZE)
        self._t_data.clear()
        self._y_data.clear()
        self.line_1.set_data(range(WINDOW_SIZE), list(self.data))
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)
        self._update_ylabel()
        self.canvas.draw_idle()

    def _selected_channel_number(self):
        return self._channel_lookup.get(self.channel_var.get())

    def _update_ylabel(self):
        """Set the y-axis label to the currently selected channel's name."""
        label = "CoF [-]"
        for ch in self.orch.channels():
            if ch["number"] == self._selected_channel_number():
                label = ch["name"]
                break
        self.ax.set_ylabel(label)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # CSV OUTPUT
    # ------------------------------------------------------------------
    def _open_csv(self):
        """Open the output file (fresh) and write the header row with the
        channel names."""
        path = Path(self.output_path.get().strip() or OUT_CSV)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        # header: channel names, in the same order as the frame
        self._csv_writer.writerow([ch["name"] for ch in self.orch.channels()])
        self._csv_file.flush()

    def _write_csv_frame(self, frame):
        """Append one frame as a CSV row."""
        if self._csv_writer is None:
            return
        if isinstance(frame, dict):
            frame = frame.values()
        self._csv_writer.writerow(np.atleast_1d(list(frame)).tolist())
        self._csv_file.flush()

    def _close_csv(self):
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            finally:
                self._csv_file = None
                self._csv_writer = None

    # ------------------------------------------------------------------
    # TIMER
    # ------------------------------------------------------------------
    def _timer_seconds(self):
        """Duração do ensaio em segundos, lida da aba Plano de Ensaio (entrada em minutos)."""
        return max(0.0, safe_float(self.time_entry.get()) * 60)

    def _on_time_changed(self):
        # Só atualiza o display se não houver ensaio em andamento
        if not self._running:
            self.timer_label.config(text=format_mmss(self._timer_seconds()), fg="black")

    def _reset_timer(self):
        self._timer_t0 = None
        self.timer_label.config(text=format_mmss(self._timer_seconds()), fg="black")

    def _update_timer(self):
        if self._timer_t0 is None:
            return
        remaining = self._timer_total - (time.monotonic() - self._timer_t0)
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

        self._close_csv()  # in case of a previous run
        self._open_csv()

        self._t0 = time.time()
        self._t_data.clear()
        self._y_data.clear()
        self.data.clear()
        self.data.extend([0.0] * WINDOW_SIZE)

        self._timer_total = self._timer_seconds()  # congela a duração ao iniciar
        self._running = True
        self._reset_timer()
        self._timer_t0 = time.monotonic()

    def stop(self, reset_timer=True):
        self.orch.stop()
        self._running = False
        self._close_csv()
        if reset_timer:
            pass  # self._reset_timer()
        else:
            self._timer_t0 = None  # freeze the display (at 00:00)

    def velocity_mode(self):
        a = safe_float(self.a_factor_entry.get())
        b = safe_float(self.b_factor_entry.get())
        self.orch.profile_factors(a, b)
        self.orch.commit_profile()

    def add_channel(self):
        result = ask_value(self.root, "nome", "coiso")
        if result and result.get("channel"):
            # Add the channel to the orchestrator, then refresh the UI
            self.orch.add_channel(result["channel"], result["name"])
            self._populate_channels()
            self._refresh_channel_combo()

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
                ch = self._selected_channel_number()
                frame = self.orch.read_enabled()
                self._write_csv_frame(frame)  # append this frame to the CSV
                if ch is not None:
                    val = self.orch.read_channel(ch)
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
    config = parse_config("config.txt")
    # gui = GUI(config)
    # gui.run()
    # exit()
    with HardwareSim(fail_prop=0.5) as rig:
        rig.set_channel(1, 500, 10)
        gui = GUI(rig.make_config())
        gui.run()


if __name__ == "__main__":
    main()
