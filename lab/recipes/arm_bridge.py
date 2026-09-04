#!/usr/bin/env python3
"""Arm bridge: the process that talks to an SO-ARM101 (six Feetech STS3215 servos on one serial bus).

It runs as an ordinary Cortex run (local executor by default, or on the GPU box if the arm hangs off it) and speaks
the runs protocol. The server never touches the serial port itself: it writes joint targets to <dir>/cmd.json and the
bridge writes what the servos report to <dir>/obs.json, twenty times a second. Without an arm, --sim emulates the
servos with a first-order lag so recording, playback and planning work end to end on any laptop.

    python arm_bridge.py --dir ~/Cortex/arm --sim
    python arm_bridge.py --dir ~/Cortex/arm --port /dev/tty.usbmodem58760431541

Protocol on the wire (Feetech/SCS): FF FF id len instr params checksum. Registers: 42 goal position (2 bytes),
56 present position (2 bytes), 40 torque enable. Positions are 12-bit ticks (0..4095) = 360 degrees; the
calibration in <dir>/calibration.json maps ticks to degrees per joint (offset, sign). LeRobot uses the same bus;
when `lerobot` is installed its FeetechMotorsBus is preferred, otherwise a minimal pyserial driver is used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from common import metric, status, result  # type: ignore
except Exception:
    def _p(tag, **kw): print(tag + " " + json.dumps(kw), flush=True)  # noqa: E306
    def metric(**kw): _p("METRIC", **kw)  # noqa: E306
    def status(**kw): _p("STATUS", **kw)  # noqa: E306
    def result(**kw): _p("RESULT", **kw)  # noqa: E306

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IDS = {j: i + 1 for i, j in enumerate(JOINTS)}
LIMITS = {"shoulder_pan": (-110, 110), "shoulder_lift": (-100, 100), "elbow_flex": (-100, 100), "wrist_flex": (-100, 100), "wrist_roll": (-160, 160), "gripper": (0, 100)}


def clamp(j: str, v: float) -> float:
    lo, hi = LIMITS[j]
    return float(max(lo, min(hi, v)))


class SimServos:
    """Servos that do not exist: each joint moves toward its goal with a time constant of about 120 ms."""
    def __init__(self):
        self.pos = {j: 0.0 for j in JOINTS}
        self.goal = dict(self.pos)
        self.t = time.time()

    def write(self, goals: dict[str, float]) -> None:
        self.goal.update({j: clamp(j, v) for j, v in goals.items() if j in self.pos})

    def read(self) -> dict[str, float]:
        now = time.time()
        dt = min(0.2, now - self.t)
        self.t = now
        k = 1 - pow(2.718281828, -dt / 0.12)
        for j in JOINTS:
            self.pos[j] += (self.goal[j] - self.pos[j]) * k
        return dict(self.pos)


class FeetechSerial:
    """Minimal STS3215 driver over pyserial: goal position writes, present position reads."""
    def __init__(self, port: str, baud: int, calib: dict):
        import serial  # pyserial
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.calib = calib
        for j in JOINTS:
            self._write(IDS[j], 40, bytes([1]))  # torque on

    @staticmethod
    def _pkt(i: int, instr: int, params: bytes) -> bytes:
        body = bytes([i, len(params) + 2, instr]) + params
        return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])

    def _write(self, i: int, reg: int, data: bytes) -> None:
        self.ser.write(self._pkt(i, 3, bytes([reg]) + data))
        self.ser.read(6 + 0)  # swallow the status packet if any

    def _read(self, i: int, reg: int, n: int) -> bytes | None:
        self.ser.reset_input_buffer()
        self.ser.write(self._pkt(i, 2, bytes([reg, n])))
        r = self.ser.read(6 + n)
        if len(r) < 6 + n or r[:2] != b"\xff\xff":
            return None
        return r[5:5 + n]

    def deg_to_ticks(self, j: str, d: float) -> int:
        c = self.calib.get(j, {"offset": 2048, "sign": 1})
        return int(max(0, min(4095, round(c["offset"] + c["sign"] * d * 4096 / 360))))

    def ticks_to_deg(self, j: str, t: int) -> float:
        c = self.calib.get(j, {"offset": 2048, "sign": 1})
        return (t - c["offset"]) * 360 / 4096 * c["sign"]

    def write(self, goals: dict[str, float]) -> None:
        for j, v in goals.items():
            if j in IDS:
                t = self.deg_to_ticks(j, clamp(j, v))
                self._write(IDS[j], 42, bytes([t & 0xFF, (t >> 8) & 0xFF]))

    def read(self) -> dict[str, float]:
        out = {}
        for j in JOINTS:
            b = self._read(IDS[j], 56, 2)
            if b is not None:
                out[j] = self.ticks_to_deg(j, b[0] | (b[1] << 8))
        return out


def open_bus(port: str, baud: int, calib: dict):
    try:  # LeRobot's bus when present (same wire protocol, better error handling)
        try:
            from lerobot.motors.feetech import FeetechMotorsBus  # type: ignore
            from lerobot.motors import Motor, MotorNormMode  # type: ignore
        except Exception:
            from lerobot.common.motors.feetech import FeetechMotorsBus  # type: ignore
            from lerobot.common.motors import Motor, MotorNormMode  # type: ignore
        motors = {j: Motor(IDS[j], "sts3215", MotorNormMode.DEGREES) for j in JOINTS}
        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect()

        class LeRobotServos:
            def write(self, goals):
                for j, v in goals.items():
                    if j in IDS:
                        bus.write("Goal_Position", j, clamp(j, v))

            def read(self):
                return {j: float(bus.read("Present_Position", j)) for j in JOINTS}
        status(phase="bus", msg="lerobot FeetechMotorsBus")
        return LeRobotServos()
    except Exception as e:
        status(phase="bus", msg=f"lerobot not used ({type(e).__name__}); pyserial driver")
    return FeetechSerial(port, baud, calib)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.expanduser(os.environ.get("CORTEX_ARM_DIR", "~/Cortex/arm")))
    ap.add_argument("--port", default="")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--steps", type=int, default=0, help="stop after this many ticks (0 = run until stopped)")
    a = ap.parse_args()
    d = Path(os.path.expanduser(a.dir))
    d.mkdir(parents=True, exist_ok=True)
    calib = {}
    cp = d / "calibration.json"
    if cp.exists():
        try:
            calib = json.loads(cp.read_text())
        except Exception:
            calib = {}
    if a.sim or not a.port:
        servos = SimServos()
        mode = "sim"
        status(phase="start", msg="simulated servos (no --port)")
    else:
        servos = open_bus(a.port, a.baud, calib)
        mode = "hardware"
        status(phase="start", msg=f"serial {a.port} @ {a.baud}")
    cmd_path, obs_path = d / "cmd.json", d / "obs.json"
    last_cmd = None
    n = 0
    t0 = time.time()
    period = 1.0 / a.hz
    while True:
        tick = time.time()
        try:
            raw = cmd_path.read_text() if cmd_path.exists() else ""
            if raw and raw != last_cmd:
                c = json.loads(raw)
                servos.write({j: float(v) for j, v in (c.get("joints") or {}).items()})
                last_cmd = raw
        except Exception as e:
            status(phase="cmd", msg=f"bad cmd: {e}")
        try:
            obs = servos.read()
            tmp = obs_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"t": time.time(), "mode": mode, "joints": obs, "pid": os.getpid()}))
            os.replace(tmp, obs_path)
        except Exception as e:
            status(phase="read", msg=f"read failed: {e}")
            time.sleep(0.5)
        n += 1
        if n % int(a.hz * 5) == 0:
            metric(step=n, hz=n / (time.time() - t0), **{f"j_{k}": round(v, 2) for k, v in obs.items()})
        if a.steps and n >= a.steps:
            break
        time.sleep(max(0.0, period - (time.time() - tick)))
    result(mode=mode, ticks=n, seconds=time.time() - t0)


if __name__ == "__main__":
    main()
