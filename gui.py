"""Minimal supervisory UI: start/stop the DLG logger over IPC and plot CoF (CH1 / normal force) live."""

import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from orchestrator import Orchestrator

OUT_DIR = Path.home()
FILE_NAME = "data"
OUT_CSV = os.path.join(OUT_DIR, f"{FILE_NAME}.csv")
RATE_HZ = 50

force_n = 1.0
t_data, y_data = [], []


def tail_csv():
    """Follow dlg.csv (idx,t_qpc,t_s,ch1..ch8,atrito,err) and collect t_s and CoF."""
    while running and not os.path.exists(OUT_CSV):
        time.sleep(0.05)
    with open(OUT_CSV, "r", encoding="utf-8") as f:
        f.readline()  # header
        while running:
            pos = f.tell()
            line = f.readline()
            if not line.endswith("\n"):  # nothing new / partial line
                f.seek(pos)
                time.sleep(0.02)
                continue
            p = line.split(",")
            if len(p) < 13 or p[-1].strip() != "0":  # skip short rows and err=1
                continue
            try:
                t_data.append(float(p[2]))
                y_data.append(float(p[3]) / force_n)
            except ValueError:
                pass


class GUI:
    def __init__(self):
        # windowing initialization
        self.root = tk.Tk()
        self.root.title("Minimal supervisory")
        bar = tk.Frame(self.root)
        bar.pack(fill="x", padx=6, pady=6)
        tk.Label(bar, text="Normal force [N]").pack(side="left")
        force_entry = tk.Entry(bar, width=8)
        force_entry.insert(0, "1")
        force_entry.pack(side="left", padx=6)

        tk.Label(bar, text="Output path:").pack(side="left")
        self.output_path = tk.Entry(bar, width=8)
        self.output_path.insert(0, OUT_CSV)
        self.output_path.pack(side="left", padx=6)
        tk.Button(bar, text="Start", command=self.start).pack(side="left", padx=2)
        tk.Button(bar, text="Stop", command=self.stop).pack(side="left", padx=2)
        fig = Figure(figsize=(7, 3.5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.state_canvas = tk.Canvas(
            self.root, width=400, height=160, bg="white", highlightthickness=0
        )
        self.state_canvas.pack(fill="x")
        # Square dimensions
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
        # orchestrator intialization
        self.orch = Orchestrator()
        self.orch.launch()

    def output_file(self):
        path = self.output_path.get()
        f = open(path, "w")
        return f

    def start(self):
        print("starting")
        self.orch.start()

    def stop(self):
        print("stoping")
        self.orch.pause()

    def redraw(self):
        self.ax.clear()
        self.ax.plot(t_data[-2000:], y_data[-2000:])
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("CoF [-]")
        self.ax.grid(alpha=0.5)
        self.canvas.draw_idle()
        state = self.orch.check_state()
        self.state_canvas.itemconfig(self.orch_state_text, text=state)
        if state == "ERROR":
            self.state_canvas.itemconfig(self.orch_state_rect, fill="red")
        else:
            self.state_canvas.itemconfig(self.orch_state_rect, fill="green")
        self.root.after(100, self.redraw)

    def update(self):
        self.orch.update()
        # self.orch.write_data()
        self.root.after(1000, self.update)

    def on_close(self):
        self.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    gui = GUI()
    gui.update()
    gui.redraw()
    gui.run()


if __name__ == "__main__":
    main()
