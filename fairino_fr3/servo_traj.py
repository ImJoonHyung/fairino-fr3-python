"""Waypoints -> smooth Cartesian path -> joint trajectory -> ServoJ streamed at 125 Hz (FR3, controller v3.8.6).

  - Path: polyline of flange (wrist3_link) positions. Intermediate waypoints are rounded with a
    quadratic Bezier of radius r, so the arm does not stop at them.
  - Orientation: tool pointing down; only the yaw (about base z) is interpolated along the arc length.
  - IK: solved in advance every 2 mm, so reachability / joint limits / IK branch jumps are checked
    before anything is sent.
  - Speed: per-segment tool speed limit + joint speed limit (dq/ds) + acceleration limit, zero at the
    end (trapezoid, backward pass computed in advance).
  - Stop: when stop_event is set the arm decelerates along the path at a_stop -> Stopped exception.
  - Watchdog: a thread compares the measured joints with the recent commands and aborts on deviation.
"""
import math
import threading
import time
import xmlrpc.client
from collections import deque

import numpy as np


class Stopped(Exception):
    pass


def _yaw_R(R0, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R0


class Path:
    """flange_poses: list of 4x4 (wrist3_link targets in base_link), speeds: tool speed limit per segment (m/s)."""

    def __init__(self, flange_poses, speeds, blend_r=0.03, ds=0.002):
        P = [T[:3, 3] for T in flange_poses]
        R0 = flange_poses[0][:3, :3]
        yaws = [math.atan2((T[:3, :3] @ R0.T)[1, 0], (T[:3, :3] @ R0.T)[0, 0]) for T in flange_poses]
        pts, vl = [P[0]], []
        # round the intermediate corners
        for i in range(1, len(P)):
            a, b = P[i - 1], P[i]
            if i < len(P) - 1:
                c = P[i + 1]
                r = min(blend_r, 0.45 * np.linalg.norm(b - a), 0.45 * np.linalg.norm(c - b))
                u1 = (b - a) / max(np.linalg.norm(b - a), 1e-9)
                u2 = (c - b) / max(np.linalg.norm(c - b), 1e-9)
                # the blend cannot stop the arm, and near a reversal it would cut the waypoint off
                if np.dot(u1, u2) < math.cos(math.radians(150)):
                    raise ValueError(f"path turns back at waypoint {i}: split it into separate trajectories")
                p_in, p_out = b - u1 * r, b + u2 * r
                seg = self._line(pts[-1], p_in, ds)
                pts += seg
                vl += [speeds[i - 1]] * len(seg)
                arc = [(1 - t) ** 2 * p_in + 2 * (1 - t) * t * b + t ** 2 * p_out for t in np.linspace(0, 1, 12)[1:]]
                pts += arc
                vl += [min(speeds[i - 1], speeds[i])] * len(arc)
            else:
                seg = self._line(pts[-1], b, ds)
                pts += seg
                vl += [speeds[i - 1]] * len(seg)
        self.p = np.array(pts)
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.p, axis=0), axis=1))]
        self.vlim = np.r_[vl[0] if vl else 0.0, vl]
        # yaw: interpolated by the cumulative distance between the original waypoints
        cum = np.r_[0, np.cumsum([np.linalg.norm(P[i] - P[i - 1]) for i in range(1, len(P))])]
        frac = self.s / max(self.s[-1], 1e-9) * cum[-1]
        self.yaw = np.interp(frac, cum, np.unwrap(yaws))
        self.R0 = R0

    @staticmethod
    def _line(a, b, ds):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / ds)))
        return [a + (b - a) * k / n for k in range(1, n + 1)]

    def pose(self, i):
        T = np.eye(4)
        T[:3, :3] = _yaw_R(self.R0, self.yaw[i])
        T[:3, 3] = self.p[i]
        return T


class ServoTrajectory:
    def __init__(self, kin, path, q0_rad, qdot_max=math.radians(60), a_max=0.3):
        self.kin, self.path = kin, path
        q, qs = np.asarray(q0_rad, float), []
        for i in range(len(path.p)):
            T = path.pose(i)
            for _ in range(60):
                dq, _, ep, er = kin.dls_step(q, T)
                q = q + dq
                if ep < 2e-4 and er < 2e-3:
                    break
            else:
                raise RuntimeError(f"path IK failed (point {i}, {np.round(T[:3, 3], 3).tolist()})")
            if np.any(q < kin.lower + math.radians(2)) or np.any(q > kin.upper - math.radians(2)):
                raise RuntimeError("path too close to a joint limit")
            qs.append(q.copy())
        self.q = np.array(qs)
        dqds = np.max(np.abs(np.diff(self.q, axis=0)), axis=1) / np.maximum(np.diff(path.s), 1e-9)
        if len(dqds) and np.max(np.abs(np.diff(self.q, axis=0))) > math.radians(5):
            raise RuntimeError("joint jump along the path (possible IK branch switch)")
        vj = qdot_max / np.maximum(np.r_[dqds[:1], dqds], 1e-9)
        self.vcap = np.minimum(path.vlim, vj)
        # backward pass so that v = 0 at the end: v(s) <= sqrt(v(s')^2 + 2a(s'-s))
        s = path.s
        vb = self.vcap.copy()
        vb[-1] = 0.0
        for i in range(len(s) - 2, -1, -1):
            vb[i] = min(vb[i], math.sqrt(vb[i + 1] ** 2 + 2 * a_max * (s[i + 1] - s[i])))
        self.vb, self.a_max = vb, a_max
        # approximate duration
        v, t = 0.0, 0.0
        for i in range(1, len(s)):
            v2 = min(math.sqrt(v * v + 2 * a_max * (s[i] - s[i - 1])), vb[i])
            t += (s[i] - s[i - 1]) / max((v + v2) / 2, 1e-3)
            v = v2
        self.duration = t

    def q_at(self, sv):
        s = self.path.s
        i = int(np.clip(np.searchsorted(s, sv) - 1, 0, len(s) - 2))
        f = (sv - s[i]) / max(s[i + 1] - s[i], 1e-9)
        return self.q[i] + (self.q[i + 1] - self.q[i]) * np.clip(f, 0, 1), i

    def execute(self, ip, stop_event=None, dry_run=False, rate=125.0, a_stop=0.8, max_dev_deg=6.0):
        dt = 1.0 / rate
        rpc = xmlrpc.client.ServerProxy(f"http://{ip}:20003/RPC2")
        hist = deque(maxlen=int(0.6 * rate))   # last 0.6 s of commands (tracking lag is not an error)
        hist.append(self.q[0].copy())
        state = {"q_cmd": self.q[0].copy(), "abort": None, "run": True, "max_dev": 0.0, "max_lag": 0.0}

        def monitor():
            mon = xmlrpc.client.ServerProxy(f"http://{ip}:20003/RPC2")
            while state["run"]:
                time.sleep(0.2)
                try:
                    r = mon.GetActualJointPosDegree(0)
                    qa = np.radians(np.array(r[1:7], float))
                    recent = np.array(list(hist))
                    dev = np.min(np.max(np.abs(recent - qa), axis=1))       # distance from the recent command path
                    lag = np.max(np.abs(state["q_cmd"] - qa))                # plain lag (logged only)
                    state["max_dev"], state["max_lag"] = max(state["max_dev"], dev), max(state["max_lag"], lag)
                    if dev > math.radians(max_dev_deg):
                        state["abort"] = (f"measured joints left the command path by {math.degrees(dev):.1f}deg "
                                          f"(lag {math.degrees(lag):.1f}deg)")
                except Exception as e:  # noqa: BLE001
                    state["abort"] = f"state read failed {e}"
        if not dry_run:
            rc = rpc.ServoMoveStart()
            if rc != 0:
                raise RuntimeError(f"ServoMoveStart rc={rc}")
            threading.Thread(target=monitor, daemon=True).start()
        s_end = self.path.s[-1]
        sv, v, stopping, fails = 0.0, 0.0, False, 0
        t_next = time.perf_counter()
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    stopping = True
                if state["abort"]:
                    raise RuntimeError(f"aborted: {state['abort']}")
                _, i = self.q_at(sv)
                if stopping:
                    v = max(0.0, v - a_stop * dt)
                else:
                    v = min(v + self.a_max * dt, self.vb[i] if i < len(self.vb) else 0.0)
                    v = max(v, 0.004 if sv < s_end - 1e-4 else 0.0)   # avoid stalling just before the end (>= 4 mm/s)
                sv = min(s_end, sv + v * dt)
                q, _ = self.q_at(sv)
                state["q_cmd"] = q
                hist.append(q)
                if not dry_run:
                    # ServoJ(joint deg[6], ext axes[4], acc, vel, cmdT, filterT, gain, id) -- 8 arguments on v3.8.6
                    rc = rpc.ServoJ([float(x) for x in np.degrees(q)], [0.0] * 4, 0.0, 0.0, dt, 0.0, 0.0, 0)
                    fails = fails + 1 if rc != 0 else 0
                    if fails >= 5:
                        raise RuntimeError(f"ServoJ failed 5 times in a row rc={rc} (v={v:.3f}m/s)")
                    late = time.perf_counter() - t_next
                    state["late"] = max(state.get("late", 0.0), late)
                if (stopping and v <= 0.0) or sv >= s_end:
                    break
                t_next += dt
                time.sleep(max(0.0, t_next - time.perf_counter()))
            if not dry_run:
                time.sleep(0.15)   # let the last command settle
        finally:
            state["run"] = False
            if not dry_run:
                rpc.ServoMoveEnd()
                self.stats = (f"max lag {math.degrees(state['max_lag']):.1f}deg, path deviation "
                              f"{math.degrees(state['max_dev']):.1f}deg, max loop delay {state.get('late', 0.0) * 1000:.0f}ms")
        if stopping and sv < s_end - 1e-4:
            raise Stopped()
