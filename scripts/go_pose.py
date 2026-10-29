#!/usr/bin/env python3
"""Move the FR3 to a joint pose with the controller's MoveJ (10 % speed by default).

  python3 scripts/go_pose.py --q j1,j2,j3,j4,j5,j6 [--speed 10] [--check] [--ip ...]
  python3 scripts/go_pose.py --pose ready [--check]

  --check  compute and print only, do not move
Refuses to move when
  - the robot is not in automatic mode / has an error / is not idle
  - any joint origin or the flange of the joint-linear path goes below --min-height (m, base z)
  - the controller's forward kinematics and this package's kinematics disagree by more than 5 mm
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairino_fr3 import DEFAULT_IP, FR3  # noqa: E402

# Joint poses in deg. "ready" is a well-conditioned pose (manipulability ~0.023) used before teleoperation.
POSES = {"ready": [49.3, -90.0, 90.0, -90.0, -90.0, 65.7]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--pose", choices=sorted(POSES), default=None)
    ap.add_argument("--q", default=None, help="target joints in deg, comma separated")
    ap.add_argument("--speed", type=float, default=10.0, help="MoveJ speed %%")
    ap.add_argument("--min-height", type=float, default=0.15)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if (a.pose is None) == (a.q is None):
        sys.exit("give exactly one of --pose or --q")
    target = np.array(POSES[a.pose] if a.pose else [float(v) for v in a.q.split(",")])

    robot = FR3(a.ip)
    st = robot.state()
    if not (robot.ready(st) and st["state"] == 1):
        sys.exit(f"stop: robot state {st} (automatic mode, no error and idle required)")

    cur = robot.q()
    low = min(min([F[2, 3] for F in robot.kin.fk_all(np.radians(cur + (target - cur) * t))[0][2:]]
                  + [robot.flange_T(cur + (target - cur) * t)[2, 3]])
              for t in np.linspace(0, 1, 41))
    if low < a.min_height:
        sys.exit(f"stop: lowest point of the path {low * 1000:.0f}mm < {a.min_height * 1000:.0f}mm. Jog it closer first.")

    fk = robot.rpc().GetForwardKin([float(v) for v in target])
    mine = robot.flange_T(target)[:3, 3] * 1000
    if fk[0] != 0 or np.linalg.norm(np.array(fk[1:4]) - mine) > 5:
        sys.exit(f"stop: controller FK and package FK disagree {fk} vs {np.round(mine, 1)}")

    print(f"current {np.round(cur, 1).tolist()} -> {target.tolist()} (speed {a.speed}%, lowest {low * 1000:.0f}mm, "
          f"max joint change {np.max(np.abs(target - cur)):.1f}deg)")
    if a.check:
        print("--check: not moving")
        return
    t0 = time.time()
    rc = robot.move_j(target, a.speed)
    err = np.max(np.abs(robot.q() - target))
    print(f"MoveJ rc={rc} ({time.time() - t0:.1f}s), final error {err:.2f}deg")


if __name__ == "__main__":
    main()
