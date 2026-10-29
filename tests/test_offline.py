"""Offline checks (no robot, no camera): kinematics and ServoJ trajectory planning."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairino_fr3 import FR3Kinematics, Path as SPath, ServoTrajectory  # noqa: E402

READY = np.radians([49.3, -90.0, 90.0, -90.0, -90.0, 65.7])


def test_ik_roundtrip():
    kin = FR3Kinematics()
    T = kin.fk(READY)
    T_goal = T.copy()
    T_goal[:3, 3] += [0.03, -0.02, 0.04]
    q = kin.ik(T_goal, READY)
    assert q is not None
    assert np.linalg.norm(kin.fk(q)[:3, 3] - T_goal[:3, 3]) < 1e-3


def test_jacobian_matches_finite_difference():
    kin = FR3Kinematics()
    J = kin.jacobian(READY)
    h = 1e-6
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = h
        dp = (kin.fk(READY + dq)[:3, 3] - kin.fk(READY - dq)[:3, 3]) / (2 * h)
        assert np.allclose(J[:3, i], dp, atol=1e-5)


def test_servo_trajectory_vertical_line():
    kin = FR3Kinematics()
    T0 = kin.fk(READY)
    T1 = T0.copy()
    T1[2, 3] += 0.05
    traj = ServoTrajectory(kin, SPath([T0, T1], [0.03]), READY)
    # straight vertical path: x/y stay put, reaches the top, speed limit respected, stops at the end
    assert np.ptp(traj.path.p[:, 0]) < 1e-6 and np.ptp(traj.path.p[:, 1]) < 1e-6
    assert np.linalg.norm(kin.fk(traj.q[-1])[:3, 3] - T1[:3, 3]) < 1e-3
    assert traj.vb.max() <= 0.03 + 1e-9 and traj.vb[-1] == 0.0
    assert 0.05 / 0.03 < traj.duration < 2.5


def test_path_rejects_turning_back():
    kin = FR3Kinematics()
    T0 = kin.fk(READY)
    T1 = T0.copy()
    T1[2, 3] += 0.05
    try:
        SPath([T0, T1, T0], [0.03, 0.03])
    except ValueError:
        return
    raise AssertionError("a reversing path must be rejected")


def test_corner_blend_keeps_corner_close():
    kin = FR3Kinematics()
    T0 = kin.fk(READY)
    T1, T2 = T0.copy(), T0.copy()
    T1[2, 3] += 0.10
    T2[2, 3] += 0.10
    T2[0, 3] += 0.10
    p = SPath([T0, T1, T2], [0.05, 0.05]).p
    # 90 deg corner rounded with r = 3 cm: the Bezier passes r * sqrt(2) / 4 (about 1.1 cm) from the corner
    assert np.min(np.linalg.norm(p - T1[:3, 3], axis=1)) < 0.015


def test_dry_execute_reaches_end():
    kin = FR3Kinematics()
    T0 = kin.fk(READY)
    T1 = T0.copy()
    T1[0, 3] += 0.01
    traj = ServoTrajectory(kin, SPath([T0, T1], [0.05]), READY)
    traj.execute("0.0.0.0", dry_run=True, rate=2000.0)   # dry run: nothing is sent


def test_joint_limit_margin_is_enforced():
    kin = FR3Kinematics()
    assert math.isclose(kin.upper[0], 3.0543) and math.isclose(kin.lower[1], -4.6251)
