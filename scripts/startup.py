#!/usr/bin/env python3
"""Run once after powering the FR3: enable + automatic mode + gripper activation (no teach pendant needed).

  python3 scripts/startup.py [--check] [--reset-error] [--ip 192.168.58.2]
    --check        print the state only, change nothing
    --reset-error  try ResetAllError if an error is active (only after you know the cause)

Order: wait for the controller -> check state (stop on error / e-stop) -> enable -> auto mode -> gripper.
The arm does not move. The gripper fingers sweep once during activation.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairino_fr3 import DEFAULT_IP, FR3  # noqa: E402


def show(st):
    print(f"  mode={'auto' if st['mode'] == 0 else 'manual'} robot_state={st['state']} error={st['main']}/{st['sub']} "
          f"estop={st['estop']} | gripper active={st['g_active']} fault={st['g_fault']}/{st['g_fid']} pos={st['g_pos']}")


def wait_controller(robot, timeout_s=180):
    t0 = time.time()
    while True:
        try:
            ver = robot.rpc().GetSoftwareVersion()
            robot.state()
            return ver
        except Exception as e:  # noqa: BLE001
            if time.time() - t0 > timeout_s:
                sys.exit(f"controller not responding ({robot.ip}): {e}")
            print("waiting for the controller to boot...", flush=True)
            time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reset-error", action="store_true")
    a = ap.parse_args()

    robot = FR3(a.ip)
    ver = wait_controller(robot)
    r = robot.rpc()
    print(f"connected {ver[1:]}  gripper config {r.GetGripperConfig()[1:]}")
    st = robot.state()
    show(st)
    if a.check:
        return

    if st["estop"]:
        sys.exit("stop: emergency stop is pressed. Release it and run again.")
    if st["main"] or st["sub"]:
        if not a.reset_error:
            sys.exit(f"stop: robot error {st['main']}/{st['sub']}. Check the cause, then run with --reset-error.")
        print("ResetAllError", r.ResetAllError())
        time.sleep(1.0)

    print("enable + automatic mode:", robot.enable_auto())
    if not robot.state()["g_active"]:
        print("activating the gripper (fingers move once)")
        robot.activate_gripper()

    st = robot.state()
    show(st)
    ok = robot.ready(st) and st["g_active"]
    print("ready" if ok else "not ready: check the state above (if the gripper config is gone, register it again on the pendant)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
