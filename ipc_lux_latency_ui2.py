#!/usr/bin/env python3
# IPC-only Latency UI: MIC-733R DO4 (gpio304) trigger -> Phidgets LUX1000_0 light
# - Plots latency (ms) and light% (ramp)
# - Logs CSV
# - Buttons: Export CSV/PNG, Toggle Lumens Plot, DO HIGH/LOW

import os, time, csv, threading, collections
from datetime import datetime, timezone
from typing import Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

# =================== Hardware / System Config ===================
GPIO_LINE = "304" # DO4 -> gpio304
GPIO_BASE = f"/sys/class/gpio/gpio{GPIO_LINE}"
GPIO_EXPORT = "/sys/class/gpio/export"
GPIO_VALUE = f"{GPIO_BASE}/value"
GPIO_DIR = f"{GPIO_BASE}/direction"
POLL_S = 0.005 # DO polling 200 Hz

# Phidgets LUX1000_0 sampling
DATA_INTERVAL_MS = 50 # Min data interval for LUX1000_0
RING_SECONDS = 5.0 # Keep last 5s of lux readings

# Detection
DETECT_TIMEOUT_S = 3.0
BASELINE_WINDOW_S = 0.20 # Baseline from last 200ms
REL_RISE_RATIO = 0.20 # +20% over baseline OR ...
ABS_RISE_LUX = 10.0 # ... at least +10 lux

# UI / Logging
REFRESH_MS = 2000
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, datetime.now().strftime("lux_latency_%Y%m%d_%H%M%S.csv"))

plot_lumens = True

# =================== Phidgets imports ===================
from Phidget22.Devices.LightSensor import LightSensor
from Phidget22.PhidgetException import PhidgetException

# =================== Helpers ===================
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()

# =================== Phidgets LUX reader ===================
class LuxReader:
    """Background Phidgets LUX1000_0 reader with timestamped ring buffer."""
    def __init__(self, hub_port: Optional[int] = None, serial: Optional[int] = None):
        self.sensor = LightSensor()
        if serial is not None:
            self.sensor.setDeviceSerialNumber(serial)
        if hub_port is not None:
            self.sensor.setHubPort(hub_port)
        self.sensor.setChannel(0)

        self.lock = threading.Lock()
        self.buf = collections.deque(maxlen=int(RING_SECONDS*1000//DATA_INTERVAL_MS)+10)
        self.running = False

    # IMPORTANT: callback signature must be (lightSensor, illuminance)
    def _on_change(self, lightSensor, lux: float):
        with self.lock:
            self.buf.append((time.time_ns(), float(lux)))

    def start(self):
        try:
            self.sensor.openWaitForAttachment(5000)
            self.sensor.setDataInterval(DATA_INTERVAL_MS)
            self.sensor.setIlluminanceChangeTrigger(0.0)
            self.sensor.setOnIlluminanceChangeHandler(self._on_change)
            self.running = True
            # prime buffer with one manual call
            self._on_change(self.sensor, self.sensor.getIlluminance())
            print(f"[phidgets] LUX attached. DataInterval={self.sensor.getDataInterval()} ms")
        except PhidgetException as e:
            raise RuntimeError(f"Phidgets open failed: {e.details}")

    def stop(self):
        try:
            self.running = False
            self.sensor.close()
        except Exception:
            pass

    def latest(self) -> Tuple[int, float]:
        with self.lock:
            return self.buf[-1] if self.buf else (time.time_ns(), float('nan'))

    def window(self, seconds: float) -> list:
        cutoff = time.time_ns() - int(seconds*1e9)
        with self.lock:
            return [(t, v) for (t, v) in self.buf if t >= cutoff]

# =================== GPIO helpers ===================
def export_gpio():
    if not os.path.exists(GPIO_BASE):
        try:
            with open(GPIO_EXPORT, "w") as f:
                f.write(GPIO_LINE)
            print(f"[gpio] exported gpio{GPIO_LINE}")
        except Exception as e:
            print(f"[gpio] export failed: {e}")
    try:
        with open(GPIO_DIR, "w") as f:
            f.write("in")
    except Exception:
        pass

def read_gpio() -> int:
    try:
        with open(GPIO_VALUE, "r") as f:
            return 1 if f.read().strip() == "1" else 0
    except Exception:
        return 0

def write_gpio(value: int):
    """Manual DO test from UI; may fail if DO is hardware-managed."""
    try:
        with open(GPIO_DIR, "w") as f:
            f.write("out")
    except Exception:
        pass
    try:
        with open(GPIO_VALUE, "w") as f:
            f.write("1" if value else "0")
        print(f"[gpio] DO set -> {value}")
    except Exception as e:
        messagebox.showwarning("GPIO", f"Could not drive DO: {e}")

# =================== Logging ===================
def init_log():
    with open(LOG_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_iso","sequence_id","event_type",
                    "trigger_ns","light_ns","latency_ms",
                    "baseline_lux","threshold_lux","peak_lux","do_state"])
    print(f"[log] writing to {LOG_FILE}")

def log_row(seq:int, event_type:str, trig_ns, light_ns,
            latency_ms, base, thr, peak, do_state):
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            now_iso(), seq, event_type, trig_ns, light_ns,
            "" if latency_ms is None else f"{latency_ms:.3f}",
            "" if base is None else f"{base:.3f}",
            "" if thr is None else f"{thr:.3f}",
            "" if peak is None else f"{peak:.3f}",
            do_state
        ])

# =================== Detection ===================
def detect_latency_on_trigger(lux: LuxReader, trigger_ns: int
                              ) -> Tuple[Optional[float], Optional[int], float, float, float]:
    """
    Returns (latency_ms, light_ns, baseline_lux, threshold_lux, peak_lux).
    latency_ms/light_ns are None if not detected within timeout.
    """
    recent = lux.window(BASELINE_WINDOW_S)
    recent_before = [v for (t, v) in recent if t <= trigger_ns]
    baseline = float(np.median(recent_before)) if recent_before else lux.latest()[1]

    threshold = baseline + max(baseline*REL_RISE_RATIO, ABS_RISE_LUX)

    deadline = trigger_ns + int(DETECT_TIMEOUT_S*1e9)
    light_ns_hit = None
    peak = baseline

    while time.time_ns() < deadline:
        t_ns, v = lux.latest()
        if v > peak:
            peak = v
        if v >= threshold:
            light_ns_hit = t_ns
            break
        time.sleep(0.001)

    if light_ns_hit is None:
        return (None, None, baseline, threshold, peak)
    return ((light_ns_hit - trigger_ns)/1e6, light_ns_hit, baseline, threshold, peak)

# =================== Trigger monitor thread ===================
class TriggerMonitor(threading.Thread):
    def __init__(self, lux: LuxReader):
        super().__init__(daemon=True)
        self.lux = lux
        self.seq = 1
        self.running = True

    def run(self):
        export_gpio()
        prev = read_gpio()
        print(f"[do] monitoring gpio{GPIO_LINE} for rising edges...")
        while self.running:
            cur = read_gpio()
            if cur == 1 and prev == 0:
                trig_ns = time.time_ns()
                log_row(self.seq, "trigger", trig_ns, "", None, None, None, None, 1)
                latency_ms, light_ns, base, thr, peak = detect_latency_on_trigger(self.lux, trig_ns)
                if latency_ms is not None:
                    print(f"#{self.seq} DO↑ latency={latency_ms:.3f} ms base={base:.1f} lux thr={thr:.1f} peak={peak:.1f}")
                    log_row(self.seq, "network", trig_ns, light_ns, latency_ms, None, None, None, 1)
                    log_row(self.seq, "lumens", trig_ns, light_ns, None, base, thr, peak, 1)
                else:
                    print(f"#{self.seq} DO↑ light NOT detected within {DETECT_TIMEOUT_S:.1f}s base={base:.1f} thr={thr:.1f} peak={peak:.1f}")
                    log_row(self.seq, "network", trig_ns, "", None, None, None, None, 1)
                self.seq += 1
            prev = cur
            time.sleep(POLL_S)

# =================== UI helpers ===================
def export_csv():
    dest = filedialog.asksaveasfilename(defaultextension=".csv",
                                        filetypes=[("CSV Files","*.csv")],
                                        title="Save Log As")
    if dest:
        with open(LOG_FILE,"r") as fi, open(dest,"w") as fo:
            fo.write(fi.read())
        print(f"[export] CSV -> {dest}")

def export_png():
    path = filedialog.asksaveasfilename(defaultextension=".png",
                                        filetypes=[("PNG Image","*.png")],
                                        title="Save Plot As PNG")
    if path:
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"[export] PNG -> {path}")

def toggle_lumens():
    global plot_lumens
    plot_lumens = not plot_lumens

def _load_df():
    if not os.path.exists(LOG_FILE):
        return None, None, None
    df = pd.read_csv(LOG_FILE)
    for c in ["latency_ms","baseline_lux","threshold_lux","peak_lux","trigger_ns","light_ns"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp_dt"] = pd.to_datetime(df["timestamp_iso"], errors="coerce")
    df_trig = df[df["event_type"]=="trigger"].copy()
    df_net = df[df["event_type"]=="network"].copy()
    df_lux = df[df["event_type"]=="lumens"].copy()
    return df_trig, df_net, df_lux

def update_deviance_panel():
    _, df_net, _ = _load_df()
    lines, vals = [], []
    if df_net is not None and not df_net.empty:
        recent = df_net.dropna(subset=["latency_ms"]).tail(20)
        for _, r in recent.iterrows():
            ts = r.get("timestamp_iso","")
            lat = r.get("latency_ms", np.nan)
            if pd.notna(lat):
                lines.append(f"{ts} — {lat:.3f} ms"); vals.append(lat)
    if vals:
        lines.append("-"*48)
        lines.append(f"n={len(vals)} avg={np.mean(vals):.3f} ms "
                     f"min={np.min(vals):.3f} ms max={np.max(vals):.3f} ms")
    dev_text.configure(state="normal")
    dev_text.delete("1.0","end")
    dev_text.insert("end", "\n".join(lines) if lines else "No activations with measured light yet.")
    dev_text.configure(state="disabled")
    root.after(REFRESH_MS, update_deviance_panel)

def update_table():
    _, df_net, _ = _load_df()
    text = "Timestamp\t\tLatency (ms)\n"
    if df_net is not None and not df_net.empty:
        for _, r in df_net.dropna(subset=["latency_ms"]).tail(15).iterrows():
            text += f"{r['timestamp_iso']}\t{r['latency_ms']:.3f}\n"
    label.config(text=text)
    root.after(REFRESH_MS, update_table)

def update_graph():
    df_trig, df_net, df_lux = _load_df()
    ax.clear(); ax2.clear()

    if df_net is not None and not df_net.empty:
        tail = df_net.dropna(subset=["timestamp_dt","latency_ms"]).tail(120)
        ax.plot(tail["timestamp_dt"], tail["latency_ms"],
                marker="o", markersize=4, linewidth=1.8, linestyle="-",
                label="Latency", color="blue", zorder=3)
        ax.set_ylabel("Latency (ms)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left")

    if df_trig is not None and not df_trig.empty:
        for _, r in df_trig.dropna(subset=["timestamp_dt"]).tail(120).iterrows():
            ax.axvline(r["timestamp_dt"], color="purple", alpha=0.35, linewidth=1.5, zorder=1)
        ax.plot([], [], color="purple", alpha=0.7, linewidth=1.5, label="Trigger")
        ax.legend(loc="upper left")

    if plot_lumens and df_lux is not None and not df_lux.empty:
        df_r = df_lux.copy()
        recent = df_r.tail(400)
        if not recent.empty and recent["baseline_lux"].notna().any():
            dark = recent["baseline_lux"].dropna().max()
            bright = recent["peak_lux"].dropna().max()
        else:
            dark = bright = None

        if dark is not None and bright is not None and bright > dark:
            df_r["light_pct"] = 100.0 * (df_r["peak_lux"] - dark) / (bright - dark)
        else:
            mx = df_r["peak_lux"].max()
            df_r["light_pct"] = 100.0 * (df_r["peak_lux"] / mx) if pd.notna(mx) and mx>0 else 0.0

        ax2.plot(df_r["timestamp_dt"], df_r["light_pct"],
                 marker=".", markersize=3, linewidth=2.2, linestyle="-",
                 label="Light % (peak)", color="orange", zorder=4)
        ax2.set_ylabel("Light Level (%)"); ax2.set_ylim(0, 100); ax2.legend(loc="upper right")

    ax.set_title("Recent Latency and Light Level")
    ax.set_xlabel("Timestamp")
    fig.autofmt_xdate(); fig.tight_layout()
    canvas.draw()
    root.after(REFRESH_MS, update_graph)

# =================== Tk UI setup ===================
root = tk.Tk()
root.title("Latency Monitor (IPC + Phidgets)")
root.geometry("1250x900")

btn_frame = tk.Frame(root); btn_frame.pack(pady=6)
tk.Button(btn_frame, text="Export Log CSV", command=export_csv).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Toggle Lumens Plot", command=toggle_lumens).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="DO HIGH", command=lambda: write_gpio(1)).grid(row=0, column=2, padx=5)
tk.Button(btn_frame, text="DO LOW", command=lambda: write_gpio(0)).grid(row=0, column=3, padx=5)
tk.Button(btn_frame, text="Export Plot PNG", command=export_png).grid(row=0, column=4, padx=5)

dev_frame = tk.Frame(root); dev_frame.pack(padx=10, pady=4, fill="x")
tk.Label(dev_frame, text="Activation Deviance (Trigger → Light)",
         font=("TkDefaultFont",10,"bold")).pack(anchor="w")
dev_text = tk.Text(dev_frame, height=6, wrap="none")
dev_scroll = tk.Scrollbar(dev_frame, orient="vertical", command=dev_text.yview)
dev_text.configure(yscrollcommand=dev_scroll.set, state="disabled")
dev_text.pack(side="left", fill="x", expand=True); dev_scroll.pack(side="right", fill="y")

label = tk.Label(root, text="", font=("Courier",10), justify="left", anchor="nw")
label.pack(padx=10, pady=6, fill="x")

fig, ax = plt.subplots(figsize=(10.5,5.2)); ax2 = ax.twinx()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(padx=10, pady=10, fill="both", expand=True)

# =================== Main ===================
def main():
    init_log()
    lux = LuxReader(hub_port=None, serial=None) # set hub_port/serial if you want to pin
    lux.start()
    mon = TriggerMonitor(lux); mon.start()

    update_deviance_panel(); update_table(); update_graph()
    try:
        root.mainloop()
    finally:
        lux.stop()

if __name__ == "__main__":
    main()
