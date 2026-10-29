#!/usr/bin/env python3
"""Smallest ServoJ example: move the flange straight up by --dz and back down, streamed at 125 Hz.

  python3 scripts/servo_line.py                 # dry run: plan and print only (reads the joints)
  python3 scripts/servo_line.py --execute       # move (press Enter to stop smoothly)
  options: --dz 0.05 (m)  --speed 0.03 (m/s)

The current orientation is kept. Keep a hand on the emergency stop; Enter is a software stop only.
"""
import argparse
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairino_fr3 import DEFAULT_IP, FR3, Path as SPath, ServoTrajectory, Stopped  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--dz", type=float, default=0.05)
    ap.add_argument("--speed", type=float, default=0.03)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    if not 0 < a.dz <= 0.15 or not 0 < a.speed <= 0.10:
        sys.exit("keep --dz in (0, 0.15] m and --speed in (0, 0.10] m/s for this example")

    robot = FR3(a.ip)
    q0 = np.radians(robot.q())
    T0 = robot.kin.fk(q0)                      # wrist3_link pose
    T1 = T0.copy()
    T1[2, 3] += a.dz
    # up and down are two trajectories: a single path cannot turn back without stopping
    up = ServoTrajectory(robot.kin, SPath([T0, T1], [a.speed]), q0)
    down = ServoTrajectory(robot.kin, SPath([T1, T0], [a.speed]), up.q[-1])
    for name, t in (("up", up), ("down", down)):
        print(f"{name}: path {t.path.s[-1] * 100:.1f}cm, {len(t.q)} points, about {t.duration:.1f}s, "
              f"max joint change {np.degrees(np.ptp(t.q, axis=0)).max():.1f}deg")
    if not a.execute:
        print("dry run: add --execute to move")
        return
    if not robot.ready():
        sys.exit(f"stop: robot not ready {robot.state()} (run scripts/startup.py)")

    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    print("moving... press Enter to stop smoothly")
    for t in (up, down):
        try:
            t.execute(robot.ip, stop)
        except Stopped:
            print(f"stopped. {t.stats}")
            return
        print(t.stats)
    print("done")


if __name__ == "__main__":
    main()
