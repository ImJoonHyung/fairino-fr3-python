#!/usr/bin/env python3
"""Colour pick-and-place with one fixed RealSense camera and ServoJ motion (FR3 + DH AG gripper).

Perception: colour segmentation on the table -> pixel ray intersected with the calibrated table plane
(no depth stream) -> object position in base_link. Motion: waypoints joined into one smooth path and
streamed with ServoJ at 125 Hz (no stop-and-go), see fairino_fr3/servo_traj.py.

  python3 examples/color_pick.py --dry-run --image photo.png --color pink   # offline: no robot, no camera
  python3 examples/color_pick.py --dry-run                                  # live camera, robot is not commanded
  python3 examples/color_pick.py                                            # real run
Keys: 1 red/pink, 2 blue, 3 green, 4 yellow, 0 start pose, q quit, SPACE stop (smooth deceleration)
Options: --place x,y (default: put it back where it was), --fast 0.15 (m/s in the air), --slow 0.05 (m/s near the table)

Before the first run, set the constants in "cell setup" for YOUR table: START_Q, REACH, WORKSPACE and the
gripper lengths. START_Q must be a pose where the camera sees the table without the arm in the way.
"""
import argparse
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fairino_fr3 import DEFAULT_IP, FLANGE_Z, FR3, Path as SPath, ServoTrajectory, Stopped  # noqa: E402

CALIB = ROOT / "calib/T_base_cam.npz"
OUT = ROOT / "examples/out"

# ------------------------------------------------------------------ cell setup (edit for your table)
START_Q = [93.6, -88.9, 107.2, -108.3, -87.5, 0.1]   # deg, gripper down, arm out of the camera's view
REACH = (0.25, 0.50)                                  # m, radius from the base axis for a vertical grasp
WORKSPACE = ((-0.30, 0.60), (-0.75, 0.10))            # m, base_link x/y box searched for objects
# DH AG-140-165 on the FR3 flange: flange face -> fingertip. The fingers move down ~4 cm when closing.
TIP_CLOSED, TIP_OPEN = 0.240, 0.200
GRIP_APPROACH_PCT = 60   # opening before a grasp (%); less opening = less finger travel when closing


def tip_len(open_pct):
    """Flange face -> fingertip length for an opening (%), linear approximation."""
    return TIP_CLOSED - (TIP_CLOSED - TIP_OPEN) * open_pct / 100.0


# ------------------------------------------------------------------ perception
COLOR_BAND = {"red": 0, "pink": 0, "orange": 1, "yellow": 1, "green": 2, "blue": 3, "cyan": 3, "purple": 4}
HUE_BANDS = [((0, 10), (160, 180)), ((10, 35),), ((35, 85),), ((85, 130),), ((130, 160),)]  # OpenCV hue 0..180


def color_blobs(bgr, table_mask):
    """Saturated regions on the table, split by hue band so touching objects of different colours separate."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = (hsv[..., 1] > 70) & (hsv[..., 2] > 60) & (table_mask > 0)
    out = []
    for bi, band in enumerate(HUE_BANDS):
        hm = np.zeros(sat.shape, bool)
        for lo, hi in band:
            hm |= (hsv[..., 0] >= lo) & (hsv[..., 0] < hi)
        m = (sat & hm).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        out += [(lab == i, st[i], cen[i], bi) for i in range(1, n) if 200 < st[i][4] < 40000]
    return out


class Geometry:
    def __init__(self, path=CALIB):
        if not Path(path).exists():
            sys.exit(f"{path} not found: run calib/calib_touch.py first")
        d = np.load(path)
        self.T_bc = d["T_base_cam"]
        self.K, self.dist = d["K"], d["dist"]
        self.table_z = float(d["table_z"])

    def ray_plane(self, uv, z):
        """Pixel -> intersection of its viewing ray with the plane base_z = z."""
        n = cv2.undistortPoints(np.array([[uv]], np.float64), self.K, self.dist).ravel()
        o, d = self.T_bc[:3, 3], self.T_bc[:3, :3] @ np.array([n[0], n[1], 1.0])
        return o + (z - o[2]) / d[2] * d

    def project(self, P):
        pc = np.linalg.inv(self.T_bc) @ np.r_[P, 1]
        return tuple(cv2.projectPoints(pc[None, :3], np.zeros(3), np.zeros(3), self.K, self.dist)[0].ravel().astype(int))

    def table_mask(self, h, w, xy_box):
        (x0, x1), (y0, y1) = xy_box
        poly = np.array([self.project([x, y, self.table_z]) for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]])
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 1)
        return m


def perceive(bgr, color, geo, kin, q_deg, height):
    """Largest object of the requested colour within reach -> (position, yaw, image path)."""
    h, w = bgr.shape[:2]
    T_now = kin.fk(np.radians(q_deg))
    seg = [np.array(geo.project((T_now @ np.r_[0, 0, z, 1])[:3]), float) for z in (FLANGE_Z, FLANGE_Z + TIP_CLOSED)]

    def ok(c_):
        a_, b_ = seg   # skip colour on the gripper itself (pixels near the flange-fingertip segment)
        tt = np.clip(np.dot(c_ - a_, b_ - a_) / max(np.dot(b_ - a_, b_ - a_), 1e-9), 0, 1)
        P_ = geo.ray_plane(c_, geo.table_z + height / 2)
        return REACH[0] <= math.hypot(P_[0], P_[1]) <= REACH[1] and np.linalg.norm(c_ - (a_ + tt * (b_ - a_))) >= 15

    band = COLOR_BAND[color]
    blobs = [b for b in color_blobs(bgr, geo.table_mask(h, w, WORKSPACE)) if ok(np.asarray(b[2], float))]
    same = [b for b in blobs if b[3] == band]
    if not same:
        raise RuntimeError(f"no {color} object within reach ({REACH[0]:.2f}-{REACH[1]:.2f} m from the base)")
    mask, st, cen, _ = max(same, key=lambda b: b[1][4])
    P = geo.ray_plane(cen, geo.table_z + height / 2)
    ys, xs = np.nonzero(mask)
    pts = np.array([geo.ray_plane((x, y), geo.table_z + height) for x, y in zip(xs[::7], ys[::7])])
    yaw = math.radians(cv2.minAreaRect(pts[:, :2].astype(np.float32) * 1000)[2])

    vis = bgr.copy()
    for m_, s_, c_, b_ in blobs:
        col = (0, 0, 255) if m_ is mask else (255, 255, 255)
        cv2.rectangle(vis, (s_[0], s_[1]), (s_[0] + s_[2], s_[1] + s_[3]), col, 2)
    cv2.circle(vis, geo.project(P), 7, (0, 255, 0), -1)
    cv2.putText(vis, f"{color}: ({P[0]:.3f},{P[1]:.3f})m", (20, 40), 0, 1.0, (0, 0, 255), 3)
    OUT.mkdir(exist_ok=True)
    fn = OUT / f"pick_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(fn), vis)
    return P, yaw, fn, vis


# ------------------------------------------------------------------ io
class Camera:
    """Keeps the latest RealSense colour frame (reconnects on USB drop)."""

    def __init__(self):
        import pyrealsense2 as rs
        self.rs, self.img, self.lock = rs, None, threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        rs = self.rs
        while True:
            try:
                p, c = rs.pipeline(), rs.config()
                c.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
                p.start(c)
                while True:
                    f = p.wait_for_frames(3000)
                    with self.lock:
                        self.img = np.asanyarray(f.get_color_frame().get_data()).copy()
            except Exception as e:  # noqa: BLE001
                print(f"[camera] reconnecting: {e}", flush=True)
                time.sleep(2)

    def get(self, timeout=10):
        t = time.time()
        while time.time() - t < timeout:
            with self.lock:
                if self.img is not None:
                    return self.img.copy()
            time.sleep(0.05)
        raise RuntimeError("no camera frame")


class Keys:
    """Single-key terminal input (no Enter). SPACE sets the stop flag immediately."""

    def __init__(self):
        import queue
        import termios
        import tty
        self.q, self.stop = queue.Queue(), threading.Event()
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self._termios = termios
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ch = sys.stdin.read(1)
            if ch == " ":
                self.stop.set()
            self.q.put(ch)

    def wait(self, msg, allowed):
        print(msg, flush=True)
        while True:
            ch = self.q.get()
            if ch in allowed:
                return ch

    def clear(self):
        self.stop.clear()
        while not self.q.empty():
            self.q.get_nowait()

    def restore(self):
        self._termios.tcsetattr(self.fd, self._termios.TCSADRAIN, self.old)


# ------------------------------------------------------------------ motion
def _off(L):
    """wrist3_link -> fingertip (flange face + L)."""
    off = np.eye(4)
    off[2, 3] = FLANGE_Z + L
    return off


class Mover:
    """Fingertip waypoints -> ServoJ trajectory. In dry-run only a virtual joint state is updated."""

    def __init__(self, robot, stop_event, dry_run, q_start_deg):
        self.robot, self.stop, self.dry = robot, stop_event, dry_run
        self.kin = robot.kin
        self.q_virtual = np.radians(q_start_deg)
        self.drawn, self.t_total = [], 0.0

    def q(self):
        return self.q_virtual if self.dry else np.radians(self.robot.q())

    def tip(self, L):
        return self.kin.fk(self.q()) @ _off(L)

    def run(self, tip_targets, speeds, L, name):
        q0 = self.q()
        poses = [self.kin.fk(q0)] + [T @ np.linalg.inv(_off(L)) for T in tip_targets]
        traj = ServoTrajectory(self.kin, SPath(poses, speeds), q0)
        self.t_total += traj.duration
        print(f"   - {name}: path {traj.path.s[-1] * 100:.0f}cm, about {traj.duration:.1f}s", flush=True)
        if self.dry:
            self.drawn.append(np.array([(traj.path.pose(i) @ _off(L))[:3, 3] for i in range(0, len(traj.path.p), 5)]))
            self.q_virtual = traj.q[-1].copy()
            return
        if not self.robot.ready():
            raise RuntimeError(f"robot not ready {self.robot.state()}")
        try:
            traj.execute(self.robot.ip, self.stop)
        finally:
            print(f"     ({getattr(traj, 'stats', '')})", flush=True)

    def gripper(self, pct):
        if self.dry:
            print(f"   - gripper {pct}% (dry)")
            return
        self.robot.gripper(pct, force_pct=40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", default=None, help="x,y in base_link (m); default: put it back where it was")
    ap.add_argument("--height", type=float, default=0.045, help="object height (m)")
    ap.add_argument("--approach", type=float, default=0.10, help="approach height above the grasp (m)")
    ap.add_argument("--fast", type=float, default=0.15, help="tool speed in the air (m/s)")
    ap.add_argument("--slow", type=float, default=0.05, help="tool speed near the table (m/s)")
    ap.add_argument("--dry-run", action="store_true", help="plan only, never command the robot")
    ap.add_argument("--image", default=None, help="offline test on a saved colour image (implies --dry-run)")
    ap.add_argument("--color", default=None, choices=sorted(COLOR_BAND), help="with --image: colour to pick")
    ap.add_argument("--ip", default=DEFAULT_IP)
    a = ap.parse_args()
    place = tuple(map(float, a.place.split(","))) if a.place else None
    offline = a.image is not None
    dry = a.dry_run or offline
    if offline and not a.color:
        sys.exit("--image needs --color")

    geo, robot = Geometry(), FR3(a.ip)
    q_start = START_Q if offline else robot.q()       # offline: assume the arm is at the start pose
    cam = None if offline else Camera()
    keys = None if offline else Keys()
    mv = Mover(robot, keys.stop if keys else threading.Event(), dry, q_start)
    lift_z = geo.table_z + a.approach
    T_start = robot.kin.fk(np.radians(START_Q)) @ _off(TIP_CLOSED)
    holding = [False]
    L_open = tip_len(GRIP_APPROACH_PCT)
    if dry:
        print("[DRY RUN: nothing is sent to the robot]")

    def at_start():
        return np.max(np.abs(np.degrees(mv.q()) - np.array(START_Q))) < 0.5

    def go_home():
        """Up -> above the start pose (lift height) -> open gripper -> straight down to the start pose."""
        if at_start():
            return
        L = L_open
        start_tip = robot.kin.fk(np.radians(START_Q)) @ _off(L)
        cur = mv.tip(L)
        up = cur.copy()
        up[2, 3] = max(cur[2, 3], lift_z)
        above = start_tip.copy()
        above[2, 3] = max(start_tip[2, 3], lift_z)
        mv.run([up, above], [a.slow, a.fast], L, "to above the start pose")
        if holding[0]:
            print("   holding an object: stopping above the start pose")
            return
        mv.gripper(100)   # in our cell the start pose is beside the table edge; open fingers stay above the table
        mv.run([start_tip], [a.slow], L, "down to the start pose")

    def task(color, img):
        t0 = time.time()
        if not offline:
            go_home()
            img = cam.get()
        P, yaw, fn, vis = perceive(img, color, geo, robot.kin, np.degrees(mv.q()), a.height)
        print(f"   {color} -> ({P[0]:.3f}, {P[1]:.3f}) m, yaw {math.degrees(yaw):.0f} deg  [{fn.name}]")
        dyaw = (yaw - math.atan2(T_start[1, 0], T_start[0, 0]) + math.pi / 4) % (math.pi / 2) - math.pi / 4
        Rz = np.array([[math.cos(dyaw), -math.sin(dyaw), 0], [math.sin(dyaw), math.cos(dyaw), 0], [0, 0, 1]])
        # open fingertip at 65 % of the object height; closing moves the fingers down onto the upper half
        grasp_z = geo.table_z + max(0.65 * a.height, 0.028)

        def pose(x, y, z, rot=True):
            T = np.eye(4)
            T[:3, :3] = (Rz if rot else np.eye(3)) @ T_start[:3, :3]
            T[:3, 3] = [x, y, z]
            return T
        px, py = place if place else (P[0], P[1])
        L = L_open
        cur = mv.tip(L)
        mv.gripper(GRIP_APPROACH_PCT)
        mv.run([pose(cur[0, 3], cur[1, 3], lift_z, rot=False), pose(P[0], P[1], grasp_z + a.approach),
                pose(P[0], P[1], grasp_z)], [a.slow, a.fast, a.slow], L, "approach and descend")
        mv.gripper(0)
        holding[0] = True
        mv.run([pose(P[0], P[1], grasp_z + a.approach), pose(px, py, grasp_z + a.approach),
                pose(px, py, grasp_z + 0.005)], [a.slow, a.fast, a.slow], L, "lift and carry")
        mv.gripper(GRIP_APPROACH_PCT)
        holding[0] = False
        go_home()
        print(f"   done: planned motion {mv.t_total:.1f}s" if dry else f"   done: {time.time() - t0:.1f}s")
        if dry:
            colors = [(0, 255, 255), (0, 128, 255), (255, 0, 255), (0, 255, 0)]
            for k, pts in enumerate(mv.drawn):
                uv = np.array([geo.project(p) for p in pts], np.int32)
                cv2.polylines(vis, [uv], False, colors[k % len(colors)], 2)
            out = OUT / f"plan_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(out), vis)
            print(f"   planned path image: {out}")
            mv.drawn = []
            mv.q_virtual = np.radians(q_start)
        mv.t_total = 0.0

    if offline:
        img = cv2.imread(a.image)
        if img is None:
            sys.exit(f"cannot read {a.image}")
        try:
            task(a.color, img)
        except RuntimeError as e:
            sys.exit(f"x {e}")
        return

    menu = "[1] red/pink  [2] blue  [3] green  [4] yellow  [0] start pose  [q] quit   (SPACE = stop)"
    colors = {"1": "pink", "2": "blue", "3": "green", "4": "yellow"}
    try:
        while True:
            keys.clear()
            k = keys.wait(menu, allowed=tuple(colors) + ("0", "q"))
            if k == "q":
                break
            keys.clear()
            try:
                if k == "0":
                    go_home()
                    continue
                print(f"> pick the {colors[k]} object")
                task(colors[k], None)
            except Stopped:
                print(f"stopped (decelerated). joints {np.round(np.degrees(mv.q()), 1).tolist()}  [0] start pose")
            except RuntimeError as e:
                print(f"x {e}")
    finally:
        keys.restore()


if __name__ == "__main__":
    main()
