#!/usr/bin/env python3
"""Fixed third-person RealSense camera <-> FR3 base_link calibration by touching 4 board corners.

Steps
  1) Tape the ChArUco board (calib/make_charuco.py) flat on the table where the robot reaches it
     and the camera sees it whole.
  2) Run this script -> it finds the board pose and saves an image marking P1..P4.
  3) With the gripper CLOSED, jog the fingertip onto P1 with the pendant (or drag mode), press Enter;
     repeat for P2..P4. This script never moves the robot; it only reads the joints.
  4) It fits camera -> robot (T_base_cam) and the table height, prints residuals (mm) and saves them.

  python3 calib/calib_touch.py --tip-mm <flange face -> closed fingertip length> [--square-mm 35] [--ip ...]
Output: calib/T_base_cam.npz, calib/calib_touch_points.png

Accuracy: touch each point so the fingertip just meets the paper; stopping a few mm above it shifts
the table height (and every grasp) by the same amount. Keep the gripper vertical (tilt is printed).
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairino_fr3 import DEFAULT_IP, FLANGE_Z, FR3  # noqa: E402

HERE = Path(__file__).parent
TOUCH_IDX = [0, 5, 23, 18]  # four outer inner-corners of the 6x4 inner-corner grid (P1..P4)


def make_board(square_mm, marker_mm):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    return cv2.aruco.CharucoBoard((7, 5), square_mm / 1000, marker_mm / 1000, d)


def start_camera():
    p, c = rs.pipeline(), rs.config()
    c.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    prof = p.start(c)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    dist = np.array(intr.coeffs[:5], dtype=float)
    for _ in range(30):
        p.wait_for_frames(5000)
    return p, K, dist


def board_pose(p, board, K, dist, n=30):
    """Board pose from n frames (median). Returns (T_cam_board, last image, detected corners)."""
    det = cv2.aruco.CharucoDetector(board)
    rvecs, tvecs, img, ncorner = [], [], None, 0
    for _ in range(n):
        img = np.asanyarray(p.wait_for_frames(5000).get_color_frame().get_data()).copy()
        cc, ci, _, _ = det.detectBoard(img)
        if ci is None or len(ci) < 8:
            continue
        obj, imgp = board.matchImagePoints(cc, ci)
        ok, rv, tv = cv2.solvePnP(obj, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            rvecs.append(rv.ravel())
            tvecs.append(tv.ravel())
            ncorner = len(ci)
    if len(tvecs) < n // 3:
        return None, img, ncorner
    rv, tv = np.median(rvecs, 0), np.median(tvecs, 0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(rv)[0], tv
    return T, img, ncorner


def kabsch(A, B):
    """B ~= R A + t (A, B: Nx3)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cb - R @ ca
    return T


def rms_fit(A, B):
    T = kabsch(A, B)
    return np.sqrt(np.mean(np.linalg.norm((T[:3, :3] @ A.T).T + T[:3, 3] - B, axis=1) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tip-mm", type=float, required=True, help="flange face -> closed fingertip (mm)")
    ap.add_argument("--square-mm", type=float, default=35.0)
    ap.add_argument("--marker-mm", type=float, default=None, help="default: square x 26/35")
    ap.add_argument("--ip", default=DEFAULT_IP)
    a = ap.parse_args()
    marker = a.marker_mm or a.square_mm * 26 / 35
    board = make_board(a.square_mm, marker)
    corners_b = np.asarray(board.getChessboardCorners())  # board frame (m), z = 0
    robot = FR3(a.ip)
    tip = np.eye(4)
    tip[2, 3] = FLANGE_Z + a.tip_mm / 1000

    p, K, dist = start_camera()
    print(f"camera intrinsics fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
    T_cb, img, nc = board_pose(p, board, K, dist)
    if T_cb is None:
        p.stop()
        sys.exit("board not found: is the whole board in view, without glare or shadows?")
    print(f"board: {nc} corners, camera -> board {np.linalg.norm(T_cb[:3, 3]):.3f} m")

    vis = img.copy()
    cv2.drawFrameAxes(vis, K, dist, cv2.Rodrigues(T_cb[:3, :3])[0], T_cb[:3, 3], 0.05)
    pts = [corners_b[i] for i in TOUCH_IDX]
    for k, pb in enumerate(pts):
        pc = (T_cb @ np.r_[pb, 1])[:3]
        uv = cv2.projectPoints(pc[None], np.zeros(3), np.zeros(3), K, dist)[0].ravel().astype(int)
        cv2.circle(vis, tuple(uv), 8, (0, 0, 255), 2)
        cv2.putText(vis, f"P{k + 1}", (uv[0] + 10, uv[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    out_img = HERE / "calib_touch_points.png"
    cv2.imwrite(str(out_img), vis)
    p.stop()
    print(f"P1..P4 marked in: {out_img}")
    print("P = inner corner where two black squares meet. Close the gripper and touch that point with the fingertip.")
    print("Do not move the board from now on.\n")

    touched = []
    for k in range(len(pts)):
        input(f"[P{k + 1}] fingertip on P{k + 1}, then Enter ...")
        T = robot.kin.fk(np.radians(robot.q())) @ tip
        touched.append(T[:3, 3])
        tilt = np.degrees(np.arccos(abs(T[2, 2])))
        print(f"  tip {np.round(T[:3, 3] * 1000, 1).tolist()} mm (gripper {tilt:.1f} deg from vertical)")
        if tilt > 10:
            print("  warning: a tilted gripper turns any --tip-mm error into a position error")

    A, B = np.array(pts), np.array(touched)
    T_bb = kabsch(A, B)                      # base <- board
    best = min(itertools.permutations(range(len(B))), key=lambda pm: rms_fit(A, B[list(pm)]))
    if list(best) != list(range(len(B))):
        print(f"warning: the order P{[i + 1 for i in best]} fits best. Check the touch order.")
    res = np.linalg.norm((T_bb[:3, :3] @ A.T).T + T_bb[:3, 3] - B, axis=1)
    side_b = np.linalg.norm(A[1] - A[0]), np.linalg.norm(A[3] - A[0])
    side_r = np.linalg.norm(B[1] - B[0]), np.linalg.norm(B[3] - B[0])
    T_base_cam = T_bb @ np.linalg.inv(T_cb)
    table_z = float(np.mean(B[:, 2]))
    n = T_bb[:3, 2]
    print(f"\nresiduals {np.round(res * 1000, 1).tolist()} mm (RMS {np.sqrt(np.mean(res ** 2)) * 1000:.1f})")
    print(f"board sides: board {np.round(np.array(side_b) * 1000, 1).tolist()} / robot {np.round(np.array(side_r) * 1000, 1).tolist()} mm")
    print(f"table height (base_link z) {table_z * 1000:.1f} mm, table normal {np.round(n, 3).tolist()}")
    print(f"camera position (base_link) {np.round(T_base_cam[:3, 3], 3).tolist()} m")
    np.savez(HERE / "T_base_cam.npz", T_base_cam=T_base_cam, K=K, dist=dist, table_z=table_z,
             table_normal=n, T_base_board=T_bb, tip_mm=a.tip_mm, square_mm=a.square_mm,
             residual_mm=res * 1000, robot_points=B, board_points=A, stamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"saved: {HERE / 'T_base_cam.npz'}")
    if np.sqrt(np.mean(res ** 2)) > 0.005:
        print("RMS above 5 mm: check --square-mm (real printed size), --tip-mm and the touches, then redo")


if __name__ == "__main__":
    main()
