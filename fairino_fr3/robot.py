"""Minimal FAIRINO FR3 client over the controller's built-in XML-RPC (port 20003) and state stream (20004).

No vendor SDK is needed. Verified on controller software v3.8.6 only: the state-packet byte offsets
and the argument lists of MoveJ / ServoJ / MoveGripper may differ on other versions.

Notes from use
  - Never call the XML-RPC introspection method `system.listMethods`: on v3.8.6 the controller's
    20003/20004 ports closed right after it.
  - Do not probe the controller ports with raw writes (e.g. `echo > /dev/tcp/...`).
  - External motion commands (MoveJ/ServoJ) are accepted only in automatic mode (Mode(0)).
"""
import os
import socket
import struct
import threading
import time
import xmlrpc.client

import numpy as np

from .kinematics import FLANGE_Z, FR3Kinematics

DEFAULT_IP = os.environ.get("FR3_IP", "192.168.58.2")   # factory default of the first LAN port

# State stream (port 20004, ~100 Hz, packets start with 0x5A 0x5A). Offsets from the packet start, v3.8.6.
_OFF_STATUS = 2      # <BHBBiiB : frame cnt, len, program state, robot state, main err, sub err, mode
_OFF_ESTOP = 549
_OFF_GRIP_DONE = 554
_OFF_GRIPPER = 566   # <BHHB : fault id, fault, active (bit0), position (%)


class FR3:
    def __init__(self, ip=DEFAULT_IP):
        self.ip = ip
        self.kin = FR3Kinematics()

    def rpc(self):
        """A new proxy per call: xmlrpc.client.ServerProxy is not thread-safe."""
        return xmlrpc.client.ServerProxy(f"http://{self.ip}:20003/RPC2")

    # ------------------------------------------------------------------ state
    def state(self):
        s = socket.create_connection((self.ip, 20004), timeout=3)
        buf = b""
        while len(buf) < 4000:
            buf += s.recv(65536)
        s.close()
        p = buf[buf.find(b"\x5a\x5a"):]
        _, _, prog, st, main_c, sub_c, mode = struct.unpack_from("<BHBBiiB", p, _OFF_STATUS)
        g_fid, g_fault, g_act, g_pos = struct.unpack_from("<BHHB", p, _OFF_GRIPPER)
        return dict(mode=mode, state=st, main=main_c, sub=sub_c, estop=p[_OFF_ESTOP],
                    g_done=p[_OFF_GRIP_DONE], g_fid=g_fid, g_fault=g_fault, g_active=g_act & 1, g_pos=g_pos)

    def ready(self, st=None):
        """Automatic mode, no error, e-stop released."""
        st = st or self.state()
        return st["mode"] == 0 and not (st["main"] or st["sub"] or st["estop"])

    def q(self):
        """Measured joints (deg)."""
        r = self.rpc().GetActualJointPosDegree(0)
        if r[0] != 0:
            raise RuntimeError(f"GetActualJointPosDegree failed {r}")
        return np.array(r[1:7], dtype=float)

    def flange_T(self, q_deg):
        T = self.kin.fk(np.radians(q_deg))
        off = np.eye(4)
        off[2, 3] = FLANGE_Z
        return T @ off

    # ------------------------------------------------------------------ setup
    def enable_auto(self):
        """RobotEnable(1) + Mode(0). The arm does not move."""
        r = self.rpc()
        r.RobotEnable(1)
        time.sleep(0.5)
        if self.state()["mode"] != 0:
            r.Mode(0)
            time.sleep(1.0)
        return self.state()["mode"] == 0

    # ------------------------------------------------------------------ motion
    def move_j(self, q_deg, speed_pct=10.0, timeout=180):
        """Blocking joint move executed by the controller (its own trajectory). Returns rc (0 = ok)."""
        q_deg = [float(v) for v in q_deg]
        fk = self.rpc().GetForwardKin(q_deg)
        if fk[0] != 0:
            raise RuntimeError(f"GetForwardKin failed {fk}")
        res = {}
        th = threading.Thread(target=lambda: res.__setitem__("rc", self.rpc().MoveJ(
            q_deg, list(fk[1:7]), 0, 0, float(speed_pct), 0.0, 100.0, [0.0] * 4, -1.0, 0, [0.0] * 6)), daemon=True)
        th.start()
        th.join(timeout)
        return res.get("rc")

    # ------------------------------------------------------------------ gripper
    def activate_gripper(self, timeout=10.0):
        """ActGripper reset + activate. The fingers sweep once."""
        r = self.rpc()
        r.ActGripper(1, 0)
        time.sleep(1.0)
        r.ActGripper(1, 1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.5)
            if self.state()["g_active"]:
                return True
        return False

    def gripper(self, pos_pct, force_pct=40, vel_pct=50, block=0):
        """pos 100 = open, 0 = closed. block=0 waits for the motion, 1 returns at once.

        The reported position is unreliable (reads 0 until the first command after power-up and
        does not reflect a held object), so do not use it as a grasp check.
        """
        rc = self.rpc().MoveGripper(1, int(pos_pct), int(vel_pct), int(force_pct), 5000, int(block), 0, 0.0, 0, 0)
        time.sleep(0.3)
        return rc
