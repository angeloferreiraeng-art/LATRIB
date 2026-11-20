#!/usr/bin/env python3
"""hwsim.py - software stand-in for the hardware used by orchestrator/motor/sensors.

Two simulated devices, plus a tiny helper rig that wires them together:

  * ServoDriveSim + ModbusRtuServer  - a Modbus-RTU slave (servo drive) reachable over a serial
    port.  On Linux/macOS the "serial port" is a pseudo-terminal (no extra software needed).
    On Windows pair two virtual COM ports (e.g. com0com: COM5 <-> COM6) and give the simulator
    one end (--modbus-port COM6) while your code opens the other (COM5).
  * DLGSimulator - the UDP datalogger (DLG0221/DLG4K protocol used in sensors.py): answers
    SETCHCFG / ACQSETUP / ACQSTART / ACQSTOP and streams ACQDATA packets at the requested rate.

Standalone:   python hwsim.py                      # prints the port to point MOTOR_PORT at
              python hwsim.py --modbus-port COM6   # Windows with a com0com pair

IMPORTANT - what is real and what is inferred
  * Wire formats (Modbus RTU framing/CRC, the DLG struct layouts) follow the code in sensors.py
    and the Modbus spec, so protocol-level mistakes in your code get caught.
  * The *meaning* of drive registers is inferred from the constants/comments in motor.py (no
    drive manual was available).  Treat motion behaviour as a plausible model, not ground truth.
    Assumptions are collected in ServoDriveSim's docstring.
"""

from __future__ import annotations

import argparse
import math
import os
import select
import socket
import struct
import threading
import time
from collections import deque

import numpy as np

# --------------------------------------------------------------------------
# Modbus helpers
# --------------------------------------------------------------------------


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def with_crc(frame: bytes) -> bytes:
    return frame + struct.pack("<H", crc16(frame))


class ModbusError(Exception):
    """Modbus exception response (code 1 fn, 2 address, 3 value, 4 device failure)."""

    def __init__(self, code: int, msg: str = ""):
        super().__init__(f"modbus exception {code} {msg}".strip())
        self.code = code


# --------------------------------------------------------------------------
# Serial endpoints
# --------------------------------------------------------------------------


class PtyEndpoint:
    """Pseudo-terminal pair (POSIX).  Point the client at `.path`."""

    def __init__(self):
        import pty
        import tty

        self.master, self.slave = pty.openpty()
        tty.setraw(self.master)
        tty.setraw(self.slave)  # no echo, no line editing
        self.path = os.ttyname(self.slave)

    def read(self, timeout: float) -> bytes:
        r, _, _ = select.select([self.master], [], [], timeout)
        return os.read(self.master, 512) if r else b""

    def write(self, data: bytes) -> None:
        os.write(self.master, data)

    def close(self) -> None:
        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass


class SerialEndpoint:
    """Real/virtual COM port through pyserial (use with a com0com style pair)."""

    def __init__(self, port: str, baud: int = 19200):
        import serial

        self.path = port
        self.ser = serial.Serial(
            port, baud, parity="E", stopbits=1, bytesize=8, timeout=0.02
        )

    def read(self, timeout: float) -> bytes:
        self.ser.timeout = timeout
        return self.ser.read(512)

    def write(self, data: bytes) -> None:
        self.ser.write(data)

    def close(self) -> None:
        self.ser.close()


# --------------------------------------------------------------------------
# Servo drive model
# --------------------------------------------------------------------------

# Register map (taken from motor.py)
R_CTRL_MODE = 0x0200
R_SPEED_SRC_A, R_SPEED_SRC_B, R_SPEED_SRC_SEL, R_SPEED_VAL = (
    0x0600,
    0x0601,
    0x0602,
    0x0603,
)
R_VDI_ENABLE = 0x0C09
R_VDI_LEVEL = 0x3100
R_MS_HEAD = 0x1200
R_MS_ACCEL1, R_MS_DECEL1 = 0x1203, 0x1204
R_MS_FIRST = 0x1214
R_ANALOG_SPEED_10V = 0x0333
R_POS_SRC = 0x0500
R_PULSE_PER_ROT = 0x0502
R_MP_HEAD = 0x1100
R_MP_FIRST = 0x110C
R_SPEED_MON = 0x0B00
R_ABS_POS = 0x0B07  # 32 bit, low word first
R_POS_PER_REV = 0x0B09
R_ENC_ANGLE = 0x0B0A

_KNOWN = [
    (0x0200, 0x02FF), (0x0300, 0x03FF), (0x0500, 0x05FF), (0x0600, 0x06FF),
    (0x0B00, 0x0BFF), (0x0C00, 0x0CFF), (0x1100, 0x11FF), (0x1200, 0x12FF),
    (0x1700, 0x17FF), (0x3100, 0x31FF),
]  # fmt: skip
_MONITOR = (0x0B00, 0x0BFF)  # read-only block


def _i16(v: int) -> int:
    return v - 0x10000 if v & 0x8000 else v


class ServoDriveSim:
    """Register file + simple motion model of the servo drive.

    Assumptions inferred from motor.py (all adjustable / easy to change):
      * runs while VDI level register (0x3100) == 0b11, stops on anything else;
      * control mode reg 0x0200: 0 speed, 1 position;
      * speed command source = reg 0x0600 (A) unless reg 0x0602 == 1, then reg 0x0601 (B);
        source code 0 = reg 0x0603 (rpm), 1 = analog input, 5 = multi-segment speed table;
      * multi-speed table: header @0x1200 = [?, n_segments, ?], entries @0x1214 as
        (speed rpm int16, duration, wait); duration unit = `speed_time_unit_s` (0.1 s, which is
        how motor.py's set_vel_profile treats it).  The table loops forever;
      * multi-position table: header @0x1100 = [?, n_segments, ...], entries @0x110C as five
        words (pos low, pos high, speed rpm, accel, wait); positions in command pulses
        (reg 0x0502 pulses/rev); the table loops forever;
      * 0x0B00 block is read-only monitoring: 0x0B00 speed rpm, 0x0B07/08 abs position,
        0x0B09 and 0x0B0A position within a revolution (0..65535).
    Anything suspicious (e.g. speeds above `max_rpm`) is appended to `self.warnings`.
    """

    ENC_RES = 65536

    def __init__(self, speed_time_unit_s: float = 0.1, strict_addresses: bool = True,
                 max_rpm: float = 6000.0, word_low_first: bool = True, clock=time.monotonic):  # fmt: skip
        self._lock = threading.RLock()
        self.clock = clock
        self.unit_s = speed_time_unit_s
        self.strict = strict_addresses
        self.max_rpm = max_rpm
        self.word_low_first = word_low_first
        self.regs = [0] * 0x10000
        self.regs[R_PULSE_PER_ROT] = 10000
        self.regs[R_ANALOG_SPEED_10V] = 3000
        self.running = False
        self.angle_rev = 0.0
        self.speed_rpm = 0.0
        self.analog_v = 0.0  # volts on the analog speed input, settable by tests
        self.warnings: list[str] = []
        self._warned: set[str] = set()
        self._seg = 0
        self._seg_t = 0.0
        self._t = clock()

    # -- helpers -----------------------------------------------------------
    def _warn(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            self.warnings.append(msg)

    def _check_range(self, addr: int, count: int) -> None:
        if addr < 0 or addr + count > 0x10000:
            raise ModbusError(2, "address out of range")
        if self.strict:
            for a in (addr, addr + count - 1):
                if not any(lo <= a <= hi for lo, hi in _KNOWN):
                    raise ModbusError(2, f"unknown register 0x{a:04X}")

    def _ppr(self) -> int:
        return max(1, self.regs[R_PULSE_PER_ROT])

    def _get(self, a: int) -> int:
        if a == R_SPEED_MON:
            return int(round(self.speed_rpm)) & 0xFFFF
        if a in (R_POS_PER_REV, R_ENC_ANGLE):
            return int((self.angle_rev % 1.0) * self.ENC_RES) & 0xFFFF
        if a in (R_ABS_POS, R_ABS_POS + 1):
            p = int(math.floor(self.angle_rev * self._ppr())) & 0xFFFFFFFF
            return (p & 0xFFFF) if a == R_ABS_POS else (p >> 16)
        return self.regs[a]

    # -- register access (called by the Modbus server) ---------------------
    def read(self, addr: int, count: int) -> list[int]:
        with self._lock:
            self._advance()
            self._check_range(addr, count)
            return [self._get(a) for a in range(addr, addr + count)]

    def write(self, addr: int, values: list[int]) -> None:
        with self._lock:
            self._advance()
            self._check_range(addr, len(values))
            for i, v in enumerate(values):
                a = addr + i
                if _MONITOR[0] <= a <= _MONITOR[1]:
                    raise ModbusError(2, f"0x{a:04X} is read-only")
                if a == R_CTRL_MODE and v not in (0, 1, 2):
                    raise ModbusError(3, "invalid control mode")
            for i, v in enumerate(values):
                self.regs[addr + i] = v & 0xFFFF
            if addr <= R_VDI_LEVEL < addr + len(values):
                self._update_run()

    def snapshot(self) -> dict:
        with self._lock:
            self._advance()
            return dict(running=self.running, angle_rev=self.angle_rev,
                        speed_rpm=self.speed_rpm, mode=self.regs[R_CTRL_MODE])  # fmt: skip

    # -- motion ------------------------------------------------------------
    def _update_run(self) -> None:
        new = (self.regs[R_VDI_LEVEL] & 0b11) == 0b11
        if new and not self.running:
            self._seg, self._seg_t = 0, 0.0
        if not new:
            self.speed_rpm = 0.0
        self.running = new

    def _speed_table(self):
        n = min(self.regs[R_MS_HEAD + 1], 16)
        out = []
        for i in range(n):
            b = R_MS_FIRST + 3 * i
            out.append((float(_i16(self.regs[b])), self.regs[b + 1] * self.unit_s))
        return out

    def _pos_table(self):
        n = min(self.regs[R_MP_HEAD + 1], 16)
        out = []
        for i in range(n):
            b = R_MP_FIRST + 5 * i
            lo, hi = (
                (self.regs[b], self.regs[b + 1])
                if self.word_low_first
                else (self.regs[b + 1], self.regs[b])
            )
            v = (hi << 16) | lo
            out.append((v - (1 << 32) if v >= 1 << 31 else v, self.regs[b + 2]))
        return out

    def _advance(self) -> None:
        now = self.clock()
        dt, self._t = now - self._t, now
        if dt <= 0 or not self.running:
            return
        mode = self.regs[R_CTRL_MODE]
        if mode == 0:
            self._adv_speed(dt)
        elif mode == 1:
            self._adv_pos(dt)
        else:
            self.speed_rpm = 0.0

    def _adv_speed(self, dt: float) -> None:
        sel = self.regs[R_SPEED_SRC_SEL]
        src = self.regs[R_SPEED_SRC_B] if sel == 1 else self.regs[R_SPEED_SRC_A]
        if src == 5:
            table = self._speed_table()
            if not table or sum(d for _, d in table) <= 0:
                self.speed_rpm = 0.0
                return
            guard = 0
            while dt > 1e-12 and guard < 1000:
                guard += 1
                self._seg %= len(table)
                rpm, dur = table[self._seg]
                if abs(rpm) > self.max_rpm:
                    self._warn(
                        f"multi-speed segment {self._seg}: {rpm:.0f} rpm exceeds max {self.max_rpm:.0f}"
                    )
                    rpm = math.copysign(self.max_rpm, rpm)
                step = min(dt, max(0.0, dur - self._seg_t))
                self.angle_rev += rpm / 60.0 * step
                self.speed_rpm = rpm
                self._seg_t += step
                dt -= step
                if self._seg_t >= dur - 1e-12:
                    self._seg, self._seg_t = self._seg + 1, 0.0
            return
        if src == 1:
            rpm = self.analog_v / 10.0 * self.regs[R_ANALOG_SPEED_10V]
        elif src == 0:
            rpm = float(_i16(self.regs[R_SPEED_VAL]))
        else:
            rpm = 0.0
        rpm = max(-self.max_rpm, min(self.max_rpm, rpm))
        self.speed_rpm = rpm
        self.angle_rev += rpm / 60.0 * dt

    def _adv_pos(self, dt: float) -> None:
        table = self._pos_table()
        if not table:
            self.speed_rpm = 0.0
            return
        ppr = self._ppr()
        guard = 0
        while dt > 1e-12 and guard < 200:
            guard += 1
            self._seg %= len(table)
            tgt, spd = table[self._seg]
            if spd > self.max_rpm:
                self._warn(f"multi-position segment {self._seg}: speed reg {spd} (rpm) exceeds max "
                           f"{self.max_rpm:.0f} - clamped")  # fmt: skip
                spd = int(self.max_rpm)
            v = max(spd, 1) / 60.0
            err = tgt / ppr - self.angle_rev
            if abs(err) > 1e-9:
                need = abs(err) / v
                step = min(dt, need)
                self.angle_rev += math.copysign(v * step, err)
                self.speed_rpm = math.copysign(max(spd, 1), err)
                dt -= step
                if step < need:
                    return
            self.speed_rpm = 0.0
            self._seg += 1


# --------------------------------------------------------------------------
# Modbus RTU server
# --------------------------------------------------------------------------


class ModbusRtuServer:
    """Serves ServoDriveSim over RTU (functions 0x03, 0x04, 0x06, 0x10).

    unit_id_policy: "strict"  - ignore requests for any other unit id (like a real drive);
                    "lenient" - also answer unit id 0 but record it in `id_mismatches`.
    Fault injection: inject_exception(addr, code, count), drop_next(n).
    """

    def __init__(
        self,
        drive: ServoDriveSim,
        endpoint,
        slave_id: int = 1,
        unit_id_policy: str = "lenient",
    ):
        self.drive, self.ep, self.slave_id, self.policy = (
            drive,
            endpoint,
            slave_id,
            unit_id_policy,
        )
        self.requests: deque = deque(maxlen=5000)  # (unit, fc, addr, count)
        self.id_mismatches: list[int] = []
        self.stats = dict(frames=0, bad_crc=0, ignored=0, exceptions=0, dropped=0)
        self._faults: list[list[int]] = []
        self._drop = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="modbus-sim", daemon=True
        )

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.ep.close()

    def inject_exception(self, addr: int, code: int = 4, count: int = 1) -> None:
        with self._lock:
            self._faults.append([addr, code, count])

    def drop_next(self, n: int = 1) -> None:
        with self._lock:
            self._drop = n

    @staticmethod
    def _expected_len(buf: bytearray):
        if len(buf) < 2:
            return None
        if buf[1] in (15, 16):
            return None if len(buf) < 7 else 9 + buf[6]
        return 8  # every other request we know of (or answer with "illegal function")

    def _run(self):
        buf, last = bytearray(), time.monotonic()
        while not self._stop.is_set():
            try:
                data = self.ep.read(0.02)
            except (OSError, ValueError):
                if self._stop.is_set():
                    break
                time.sleep(0.01)
                continue
            now = time.monotonic()
            if buf and now - last > 0.5:
                buf.clear()  # stale partial frame
            if data:
                buf += data
                last = now
            while buf:
                n = self._expected_len(buf)
                if n is None or len(buf) < n:
                    break
                frame = bytes(buf[:n])
                del buf[:n]
                if not self._on_frame(frame):
                    buf.clear()

    def _on_frame(self, frame: bytes) -> bool:
        if crc16(frame[:-2]) != struct.unpack("<H", frame[-2:])[0]:
            self.stats["bad_crc"] += 1
            return False
        self.stats["frames"] += 1
        unit, fc, body = frame[0], frame[1], frame[2:-2]
        if unit != self.slave_id:
            self.id_mismatches.append(unit)
            if not (unit == 0 and self.policy == "lenient"):
                self.stats["ignored"] += 1
                return True
        with self._lock:
            if self._drop > 0:
                self._drop -= 1
                self.stats["dropped"] += 1
                return True
        try:
            pdu = self._execute(unit, fc, body)
        except ModbusError as e:
            self.stats["exceptions"] += 1
            pdu = bytes([fc | 0x80, e.code])
        self.ep.write(with_crc(bytes([unit]) + pdu))
        return True

    def _fault(self, addr: int, count: int) -> None:
        with self._lock:
            for f in self._faults:
                if f[2] > 0 and addr <= f[0] < addr + count:
                    f[2] -= 1
                    raise ModbusError(f[1], "injected")

    def _execute(self, unit: int, fc: int, body: bytes) -> bytes:
        if fc in (3, 4):
            addr, count = struct.unpack(">HH", body[:4])
            self.requests.append((unit, fc, addr, count))
            if not 1 <= count <= 125:
                raise ModbusError(3, "bad count")
            self._fault(addr, count)
            vals = self.drive.read(addr, count)
            return bytes([fc, 2 * count]) + struct.pack(f">{count}H", *vals)
        if fc == 6:
            addr, val = struct.unpack(">HH", body[:4])
            self.requests.append((unit, fc, addr, 1))
            self._fault(addr, 1)
            self.drive.write(addr, [val])
            return bytes([fc]) + body[:4]
        if fc == 16:
            addr, count, bc = struct.unpack(">HHB", body[:5])
            self.requests.append((unit, fc, addr, count))
            if not 1 <= count <= 123 or bc != 2 * count or len(body) < 5 + bc:
                raise ModbusError(3, "bad count")
            vals = list(struct.unpack(f">{count}H", body[5 : 5 + bc]))
            self._fault(addr, count)
            self.drive.write(addr, vals)
            return bytes([fc]) + struct.pack(">HH", addr, count)
        raise ModbusError(1, "function not supported")


class RtuMaster:
    """Minimal blocking Modbus-RTU master (used by the self-tests and the pymodbus stand-in)."""

    def __init__(self, port: str, timeout: float = 0.3):
        self.timeout = timeout
        self._ser = None
        if os.name == "posix":
            import tty

            self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY)
            tty.setraw(self.fd)
        else:  # pragma: no cover
            import serial

            self._ser = serial.Serial(port, 19200, parity="E", timeout=0.02)

    def close(self):
        if self._ser:
            self._ser.close()
        else:
            os.close(self.fd)

    def _read(self, t: float) -> bytes:
        if self._ser:
            self._ser.timeout = t
            return self._ser.read(256)
        r, _, _ = select.select([self.fd], [], [], t)
        return os.read(self.fd, 256) if r else b""

    @staticmethod
    def _resp_len(buf: bytes):
        if len(buf) < 2:
            return None
        if buf[1] & 0x80:
            return 5
        if buf[1] in (3, 4):
            return None if len(buf) < 3 else 5 + buf[2]
        return 8

    def raw(self, frame: bytes, timeout: float | None = None) -> bytes:
        """Send a raw frame, return the raw response (b'' on timeout)."""
        while self._read(0):  # flush stale input
            pass
        if self._ser:
            self._ser.write(frame)
        else:
            os.write(self.fd, frame)
        deadline = time.monotonic() + (timeout or self.timeout)
        buf = b""
        while time.monotonic() < deadline:
            buf += self._read(0.02)
            n = self._resp_len(buf)
            if n is not None and len(buf) >= n:
                return buf[:n]
        return b""

    def transact(self, unit: int, pdu: bytes) -> bytes:
        resp = self.raw(with_crc(bytes([unit]) + pdu))
        if not resp:
            raise TimeoutError("no response")
        if crc16(resp[:-2]) != struct.unpack("<H", resp[-2:])[0]:
            raise IOError("bad response CRC")
        if resp[0] != unit:
            raise IOError(f"response from unit {resp[0]}, expected {unit}")
        if resp[1] & 0x80:
            raise ModbusError(resp[2])
        return resp[1:-2]

    def read(self, unit: int, addr: int, count: int = 1) -> list[int]:
        p = self.transact(unit, struct.pack(">BHH", 3, addr, count))
        return list(struct.unpack(f">{p[1] // 2}H", p[2:]))

    def write(self, unit: int, addr: int, value: int) -> None:
        self.transact(unit, struct.pack(">BHH", 6, addr, value & 0xFFFF))

    def write_many(self, unit: int, addr: int, values) -> None:
        values = [int(v) & 0xFFFF for v in values]
        self.transact(
            unit,
            struct.pack(">BHHB", 16, addr, len(values), 2 * len(values))
            + struct.pack(f">{len(values)}H", *values),
        )


# --------------------------------------------------------------------------
# DLG datalogger (UDP)
# --------------------------------------------------------------------------
OP_ACQSETUP, OP_ACQSETUP_R = 0x1000, 0x1001
OP_ACQSTART, OP_ACQSTART_R = 0x1002, 0x1003
OP_ACQSTOP, OP_ACQSTOP_R = 0x1006, 0x1007
OP_ACQDATA = 0x100B
OP_SETCHCFG, OP_SETCHCFG_R = 0x2002, 0x2003

_HDR = struct.Struct("<HH")
_RESP = struct.Struct("<HHH")
_ACQ_SETUP = struct.Struct("<HHfHHHHhh8h")
_ACQ_SETUP_R = struct.Struct("<HHHf")
_SET_CH = struct.Struct("<HHH6hHHH")
_DATA_HDR = struct.Struct("<HHiifhhiii")
N_CH = 8
MAX_SAMPLES_PER_PACKET = 720


class SignalModel:
    """Raw int16 counts for the 8 channels.

    ch1 strain gauge: offset + counts_per_unit*load + mains hum + noise   (set `load`)
    ch3 encoder:      follows the simulated drive shaft angle (+-10000 counts per turn)
    ch0, ch2, ch4..7: constant 100*channel + noise  (distinct, so channel mix-ups are visible)
    """

    def __init__(self, drive: ServoDriveSim | None = None, seed: int = 0, mains_hz: float = 60.0,
                 noise_counts: float = 2.0, hum_counts: float = 30.0):  # fmt: skip
        self.drive = drive
        self.rng = np.random.default_rng(seed)
        self.mains_hz, self.noise, self.hum = mains_hz, noise_counts, hum_counts
        self.load = 0.0
        self.strain_counts_per_unit = 100.0
        self.strain_offset = 250.0
        self.overrides: dict[
            int, tuple[float, float]
        ] = {}  # ch -> (value, noise sigma)

    def set_channel(self, ch: int, value: float, noise: float | None = None) -> None:
        """Pin channel `ch` to raw counts around `value` (gaussian noise sigma, default = model noise)."""
        if not 0 <= ch < N_CH:
            raise ValueError(f"channel {ch} out of range 0..{N_CH - 1}")
        self.overrides[ch] = (
            float(value),
            self.noise if noise is None else float(noise),
        )

    def clear_channel(self, ch: int | None = None) -> None:
        """Undo set_channel for one channel, or for all when ch is None."""
        if ch is None:
            self.overrides.clear()
        else:
            self.overrides.pop(ch, None)

    def sample(self, t: float) -> np.ndarray:
        n = self.rng.normal(0.0, self.noise, N_CH)
        out = 100.0 * np.arange(N_CH) + n
        out[1] = (self.strain_offset + self.strain_counts_per_unit * self.load
                  + self.hum * math.sin(2 * math.pi * self.mains_hz * t) + n[1])  # fmt: skip
        if self.drive is not None:
            frac = self.drive.snapshot()["angle_rev"] % 1.0
            out[3] = (frac - 0.5) * 20000.0 + n[3]
        for ch, (value, sigma) in list(self.overrides.items()):
            out[ch] = value + self.rng.normal(0.0, sigma)
        return out


class DLGSimulator:
    """UDP datalogger.  Replies go to whoever sent the request (like the real unit).

    rate_scale      real sample rate = requested * rate_scale (lets tests check that the
                    'real frequency' in the AcqSetup reply is honoured).
    reply_to_config send SETCHCFG_R / ACQSTOP_R acknowledgements (sensors.py's acq_setup_new
                    comment says the hardware does).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 41401, model: SignalModel | None = None,
                 rate_scale: float = 1.0, max_rate_hz: float = 4000.0, reply_to_config: bool = True):  # fmt: skip
        self.model = model or SignalModel()
        self.rate_scale, self.max_rate, self.reply_to_config = (
            rate_scale,
            max_rate_hz,
            reply_to_config,
        )
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.address = self.sock.getsockname()
        self.rx_log: deque = deque(maxlen=20000)  # opcodes received
        self.channel_cfgs: dict[int, tuple] = {}
        self.setup: dict = {}
        self.streaming = False
        self.packets_sent = 0
        self.rate = 0.0
        self.n_bursts, self.n_ch = 1, N_CH
        self._client = None
        self._frame_no = 0
        self._sample_idx = 0
        self._next_tx = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dlg-sim", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.sock.close()

    def _reply(self, payload: bytes, addr) -> None:
        self.sock.sendto(payload, addr)

    def _on_packet(self, pkt: bytes, addr) -> None:
        if len(pkt) < _HDR.size:
            return
        op = struct.unpack_from("<H", pkt)[0]
        self.rx_log.append(op)
        self._client = addr
        if op == OP_SETCHCFG and len(pkt) == _SET_CH.size:
            _, _, _, ch, t, gi, lpf, pwr, bal, dc, ac, ub = _SET_CH.unpack(pkt)
            err = 0
            if not 0 <= ch < N_CH or not 0 <= t <= 5:
                err = 1
            else:
                self.channel_cfgs[ch] = (t, gi, lpf, pwr, bal, dc, ac, ub)
            if self.reply_to_config:
                self._reply(_RESP.pack(OP_SETCHCFG_R, 0, err), addr)
        elif op == OP_ACQSETUP and len(pkt) == _ACQ_SETUP.size:
            f = _ACQ_SETUP.unpack(pkt)
            rate, n_bursts, n_ch = f[2], f[7], f[8]
            if not (0 < rate <= self.max_rate) or n_bursts < 1 or not 1 <= n_ch <= N_CH \
                    or n_bursts * n_ch > MAX_SAMPLES_PER_PACKET:  # fmt: skip
                self._reply(_ACQ_SETUP_R.pack(OP_ACQSETUP_R, 0, 1, 0.0), addr)
                return
            self.rate, self.n_bursts, self.n_ch = rate * self.rate_scale, n_bursts, n_ch
            self.setup = dict(
                requested=rate,
                real=self.rate,
                n_bursts=n_bursts,
                n_ch=n_ch,
                icm=list(f[9:]),
            )
            self._reply(_ACQ_SETUP_R.pack(OP_ACQSETUP_R, 0, 0, float(self.rate)), addr)
        elif op == OP_ACQSTART and len(pkt) == _HDR.size:
            if self.rate > 0:
                self.streaming, self._sample_idx = True, 0
                self._next_tx = time.perf_counter()
            self._reply(_RESP.pack(OP_ACQSTART_R, 0, 0 if self.rate > 0 else 1), addr)
        elif op == OP_ACQSTOP and len(pkt) == _HDR.size:
            self.streaming = False
            if self.reply_to_config:
                self._reply(_RESP.pack(OP_ACQSTOP_R, 0, 0), addr)

    def _send_data(self) -> None:
        n, nch = self.n_bursts, self.n_ch
        rows = np.empty((n, nch), dtype="<i2")
        t = 0.0
        for b in range(n):
            t = self._sample_idx / self.rate
            rows[b] = np.clip(np.round(self.model.sample(t)[:nch]), -32768, 32767)
            self._sample_idx += 1
        hdr = _DATA_HDR.pack(OP_ACQDATA, 0, self._frame_no, int(t * 1000) & 0x7FFFFFFF,
                             float(self.rate), n, nch, 0, 0, 0)  # fmt: skip
        self._frame_no += 1
        self.packets_sent += 1
        self._reply(hdr + rows.tobytes(), self._client)
        self._next_tx += n / self.rate
        if time.perf_counter() - self._next_tx > 0.25:  # fell far behind: resync
            self._next_tx = time.perf_counter()

    def _run(self):
        while not self._stop.is_set():
            wait = (
                max(0.0005, self._next_tx - time.perf_counter())
                if self.streaming
                else 0.05
            )
            self.sock.settimeout(wait)
            try:
                pkt, addr = self.sock.recvfrom(4096)
                self._on_packet(pkt, addr)
            except socket.timeout:
                pass
            except OSError:
                if self._stop.is_set():
                    break
                continue
            if self.streaming and self._client and time.perf_counter() >= self._next_tx:
                try:
                    self._send_data()
                except OSError:
                    pass


# --------------------------------------------------------------------------
# Whole rig
# --------------------------------------------------------------------------
class HardwareSim:
    """Drive + datalogger together.  Use as a context manager.

    modbus_port  None -> pseudo-terminal (POSIX).  Otherwise the simulator's end of a virtual
                 COM pair; then pass `client_port` (the other end) for make_config().
    """

    def __init__(self, udp_host: str = "127.0.0.1", udp_port: int = 41401, modbus_port: str | None = None,
                 client_port: str | None = None, slave_id: int = 1, unit_id_policy: str = "lenient",
                 rate_scale: float = 1.0, reply_to_config: bool = True, seed: int = 0,
                 speed_time_unit_s: float = 0.1, baud: int = 19200):  # fmt: skip
        self.udp_host, self.slave_id, self.baud = udp_host, slave_id, baud
        self.drive = ServoDriveSim(speed_time_unit_s=speed_time_unit_s)
        self.model = SignalModel(self.drive, seed=seed)
        self.dlg = DLGSimulator(
            udp_host, udp_port, self.model, rate_scale, reply_to_config=reply_to_config
        )
        self._ep = (
            PtyEndpoint() if modbus_port is None else SerialEndpoint(modbus_port, baud)
        )
        self.client_port = client_port or self._ep.path
        self.modbus = ModbusRtuServer(self.drive, self._ep, slave_id, unit_id_policy)

    @property
    def udp_port(self) -> int:
        return self.dlg.address[1]

    def start(self):
        self.dlg.start()
        self.modbus.start()
        return self

    def stop(self):
        self.modbus.stop()
        self.dlg.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def set_channel(self, ch: int, value: float, noise: float | None = None) -> None:
        """Make channel `ch` output raw counts around `value`, e.g. rig.set_channel(1, 5000)
        to simulate a sensor reading a known reference.  Takes effect immediately."""
        print(f"SETTING CHANNEL TO {value}")
        self.model.set_channel(ch, value, noise)

    def clear_channel(self, ch: int | None = None) -> None:
        """Back to the default signal for `ch` (or for all channels if None)."""
        self.model.clear_channel(ch)

    def make_config(self, rate_hz: float = 200.0, bind_port: int = 41402, timeout_s: float = 1.0,
                    frames_per_packet: int = 1, cutoff: float = 0.1) -> dict[str, str]:  # fmt: skip
        """Config dict in the format Orchestrator expects (same keys parse_config() produces)."""
        types = [4, 4, 4, 1, 4, 4, 4, 4]  # SensorsConfig defaults
        cfg = {}
        for i in range(N_CH):
            cfg[f"SENSORS_CHANNEL{i}_TYPE"] = str(types[i])
            cfg[f"SENSORS_CHANNEL{i}_CUTOFF"] = str(cutoff)
            cfg[f"SENSORS_CHANNEL{i}_RESONANCE"] = "0.7071"
            cfg[f"SENSORS_CHANNEL{i}_GAIN"] = "1.0"
            cfg[f"SENSORS_CHANNEL{i}_OFFSET"] = "0.0"
        cfg.update(SENSORS_IP=self.udp_host, SENSORS_PORT=str(self.udp_port), SENSORS_BIND_IP="127.0.0.1",
                   SENSORS_BIND_PORT=str(bind_port), SENSORS_RATE=str(rate_hz),
                   SENSORS_FRAME_PER_PACKET=str(frames_per_packet), SENSORS_TIMEOUT=str(timeout_s),
                   MOTOR_PORT=self.client_port, MOTOR_ID=str(self.slave_id), MOTOR_BAUD_RATE=str(self.baud))  # fmt: skip
        return cfg


def main():
    ap = argparse.ArgumentParser(description="Run the hardware simulator standalone.")
    ap.add_argument(
        "--modbus-port",
        help="simulator end of a virtual COM pair (default: pty on Linux/macOS)",
    )
    ap.add_argument(
        "--client-port", help="the other end of the pair, only used to print the config"
    )
    ap.add_argument("--udp-host", default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=41401)
    ap.add_argument("--slave-id", type=int, default=1)
    ap.add_argument(
        "--strict-ids",
        action="store_true",
        help="ignore requests whose unit id != slave id",
    )
    a = ap.parse_args()
    rig = HardwareSim(a.udp_host, a.udp_port, a.modbus_port, a.client_port, a.slave_id,
                      "strict" if a.strict_ids else "lenient")  # fmt: skip
    with rig:
        print(
            f"Modbus RTU slave id {a.slave_id}: point MOTOR_PORT at  {rig.client_port}"
        )
        print(
            f"UDP datalogger listening on {rig.dlg.address[0]}:{rig.dlg.address[1]}  (SENSORS_IP/SENSORS_PORT)"
        )
        print("Ctrl-C to quit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
