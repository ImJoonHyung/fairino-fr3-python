"""FAIRINO FR3 forward kinematics, Jacobian and damped-least-squares IK (numpy only).

Joint origins/axes/limits are the values of `fairino3_v6.urdf` (frcobot_ros2), embedded so that no
ROS installation is needed. Set FR3_URDF=/path/to/fairino3_v6.urdf to load them from a URDF instead.
The end frame is `wrist3_link`; the controller's tool flange is wrist3_link + z 101.4 mm (FLANGE_Z).
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

FLANGE_Z = 0.1014  # wrist3_link -> flange face (m), checked against the controller's GetForwardKin
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]

# name: (xyz, rpy, axis, lower, upper, velocity)  -- from fairino3_v6.urdf
_FR3_V6 = {
    "j1": ((0, 0, 0), (0, 0, 0), (0, 0, 1), -3.0543, 3.0543, 3.15),
    "j2": ((0, 0, 0.14), (1.5708, 0, 0), (0, 0, 1), -4.6251, 1.4835, 3.15),
    "j3": ((-0.28, 0, 0), (0, 0, 0), (0, 0, 1), -2.8274, 2.8274, 3.15),
    "j4": ((-0.24001, 0, 0), (0, 0, 0), (0, 0, 1), -4.6251, 1.4835, 3.2),
    "j5": ((0, 0, 0.102), (1.5708, 0, 0), (0, 0, 1), -3.0543, 3.0543, 3.2),
    "j6": ((0, 0, 0.102), (-1.5708, 0, 0), (0, 0, 1), -3.0543, 3.0543, 3.2),
}


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle_R(axis, a):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def rotvec(R):
    """Rotation matrix -> rotation vector (axis * angle)."""
    a = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if abs(math.pi - a) < 1e-6:  # numerical fix near 180 deg
        w = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        return w * a
    return v * (a / (2 * math.sin(a)))


def _load_urdf(path):
    root = ET.parse(str(path)).getroot()
    out = {}
    for name in JOINTS:
        j = root.find(f"joint[@name='{name}']")
        o, lim = j.find("origin"), j.find("limit")
        out[name] = (tuple(float(v) for v in o.get("xyz", "0 0 0").split()),
                     tuple(float(v) for v in o.get("rpy", "0 0 0").split()),
                     tuple(float(v) for v in j.find("axis").get("xyz").split()),
                     float(lim.get("lower")), float(lim.get("upper")), float(lim.get("velocity", "3.14")))
    return out


class FR3Kinematics:
    def __init__(self, urdf_path=None):
        urdf_path = urdf_path or os.environ.get("FR3_URDF")
        params = _load_urdf(urdf_path) if urdf_path else _FR3_V6
        self.origins, self.axes, lower, upper, self.vel = [], [], [], [], []
        for name in JOINTS:
            xyz, rpy, axis, lo, hi, v = params[name]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = rpy_to_R(*rpy), xyz
            self.origins.append(T)
            self.axes.append(np.array(axis, float))
            lower.append(lo)
            upper.append(hi)
            self.vel.append(v)
        self.lower, self.upper = np.array(lower), np.array(upper)

    def fk_all(self, q):
        """Joint frames (before each joint rotation) and the wrist3_link pose, q in rad."""
        T = np.eye(4)
        frames = []
        for i in range(6):
            T = T @ self.origins[i]
            frames.append(T.copy())
            R = np.eye(4)
            R[:3, :3] = axis_angle_R(self.axes[i], q[i])
            T = T @ R
        return frames, T

    def fk(self, q):
        return self.fk_all(q)[1]

    def jacobian(self, q):
        """Geometric Jacobian at the wrist3_link origin, base frame, rows [v; w]."""
        frames, Te = self.fk_all(q)
        pe = Te[:3, 3]
        J = np.zeros((6, 6))
        for i, F in enumerate(frames):
            z = F[:3, :3] @ self.axes[i]
            J[:3, i] = np.cross(z, pe - F[:3, 3])
            J[3:, i] = z
        return J

    def dls_step(self, q, T_goal, lam_max=0.08, w_thresh=0.02, max_pos_err=0.005, max_rot_err=0.03):
        """One damped-least-squares IK step. Returns (dq, manipulability, pos error, rot error).

        The error solved per step is clipped to max_pos_err [m] / max_rot_err [rad] so that the
        linearisation stays valid; damping grows only near singularities (manipulability < w_thresh).
        """
        Te = self.fk(q)
        e_p = T_goal[:3, 3] - Te[:3, 3]
        e_r = rotvec(T_goal[:3, :3] @ Te[:3, :3].T)
        n_p, n_r = np.linalg.norm(e_p), np.linalg.norm(e_r)
        err = np.concatenate([e_p * min(1.0, max_pos_err / max(n_p, 1e-12)),
                              e_r * min(1.0, max_rot_err / max(n_r, 1e-12))])
        J = self.jacobian(q)
        w = math.sqrt(max(np.linalg.det(J @ J.T), 0.0))
        lam = 0.0 if w >= w_thresh else lam_max * (1 - (w / w_thresh) ** 2)
        dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2 + 1e-9) * np.eye(6), err)
        return dq, w, n_p, n_r

    def ik(self, T_goal, q_seed, iters=600, tol_pos=2e-4, tol_rot=2e-3):
        """wrist3_link goal pose -> joints (rad), or None if it does not converge."""
        q = np.asarray(q_seed, float).copy()
        for _ in range(iters):
            dq, _, ep, er = self.dls_step(q, T_goal)
            q = np.clip(q + dq, self.lower, self.upper)
            if ep < tol_pos and er < tol_rot:
                return q
        return None
