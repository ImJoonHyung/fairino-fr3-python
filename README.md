# fairino-fr3-python

**FAIRINO FR3 cobot + DH AG gripper, controlled from plain Python: no vendor SDK, no ROS.**
Startup, state monitoring, gripper control and smooth Cartesian motion streamed with `ServoJ` at 125 Hz,
using only the controller's built-in XML-RPC server and state stream, plus numpy.

**FAIRINO FR3 협동로봇과 DH AG 그리퍼를 순수 Python으로 제어합니다. 제조사 SDK도 ROS도 필요 없습니다.**
컨트롤러에 내장된 XML-RPC 서버와 상태 스트림, numpy만으로 시작, 상태 감시, 그리퍼 제어,
그리고 `ServoJ` 125 Hz 스트리밍으로 끊김 없는 직교 경로 이동을 합니다.

> This is the **FAIRINO (法奥) FR3**, not the Franka Research 3.
> Franka Research 3가 아니라 **FAIRINO FR3**용입니다.

## Who is this for / 누구를 위한 것인가

- You have a FAIRINO arm and want it moving from Python **in minutes**, without installing ROS 2 or
  matching an SDK build to your controller version.
  ROS 2 설치나 컨트롤러 버전에 맞는 SDK 빌드 없이 Python에서 바로 움직이고 싶은 분
- You want **continuous, smooth motion** (straight descents, rounded corners, no stop at every waypoint)
  or your own real-time loop (teleoperation, visual servoing) via `ServoJ`.
  경유점마다 멈추지 않는 매끄러운 동작이나, `ServoJ`로 직접 실시간 루프(텔레옵 등)를 짜고 싶은 분
- You need the **undocumented details**: state-stream byte layout, exact argument lists, pitfalls (see
  [Controller notes](#controller-notes-v386--컨트롤러-메모)).
  상태 패킷 구조, 정확한 함수 인자, 함정 등 문서에 없는 정보가 필요한 분

Use the official SDK / ROS 2 driver instead if you need MoveIt planning with collision checking, or
long-term vendor support across controller versions.
충돌 검사가 있는 MoveIt 계획이나 여러 버전에 걸친 공식 지원이 필요하면 공식 SDK / ROS 2 드라이버를 쓰세요.

**What is the core and what is an example / 핵심과 예제**

| | |
|---|---|
| **Core library** `fairino_fr3/` | robot client, kinematics, ServoJ trajectories. Depends on numpy only. |
| Tools `scripts/` | startup, move to a pose, smallest ServoJ example |
| Examples `calib/`, `examples/` | camera calibration and a colour pick-and-place demo built on the core (needs OpenCV + RealSense) |

<!-- TODO: add a short GIF of servo_line.py or color_pick.py here / 동작 GIF 자리 -->

## Install / 설치

```bash
git clone https://github.com/ImJoonHyung/fairino-fr3-python && cd fairino-fr3-python
python3 -m venv .venv && . .venv/bin/activate
pip install -e .                 # core: numpy only
pip install -e ".[examples]"     # + OpenCV / RealSense / Pillow for calib/ and examples/
pip install -e ".[test]" && python -m pytest -q tests   # offline, no robot needed
export FR3_IP=192.168.58.2       # your controller IP (default 192.168.58.2)
```

> ⚠️ **Safety / 안전** — This code moves a real robot. Keep a hand on the hardware emergency stop,
> start with `--dry-run` / `--check` and low speeds, and clear the workspace. Software stops (SPACE / Enter)
> are not a substitute for the e-stop. Use at your own risk (MIT, no warranty).
> 실제 로봇을 움직이는 코드입니다. 항상 하드웨어 비상정지 버튼을 손에 들고, `--dry-run` / `--check`와
> 낮은 속도로 먼저 확인하세요. 소프트웨어 정지는 비상정지를 대신하지 않습니다.

## Tested with / 검증 환경

| | version |
|---|---|
| Robot | FAIRINO FR3 (v6 kinematics) |
| Controller software | **v3.8.6** (state-packet offsets and argument lists are version specific) |
| Gripper | DH Robotics AG-140-165, registered on the pendant as **DH PGI-140** (same Modbus map) |
| Camera (calibration / example only) | Intel RealSense D455, colour stream 1280x720 |
| OS / Python | Ubuntu 24.04, Python 3.12 |
| Packages | numpy 2.5.2, opencv-contrib-python 4.10.0, pyrealsense2 2.59.1, pillow 10.2 |

Other controller versions may work but are untested: check the state-stream offsets in `fairino_fr3/robot.py`
and the `MoveJ` / `ServoJ` / `MoveGripper` argument lists first.
다른 컨트롤러 버전은 검증하지 않았습니다. `fairino_fr3/robot.py`의 상태 패킷 위치와 함수 인자 개수를 먼저 확인하세요.

## How it works / 동작 방식

```
                        your PC (Python)                                  FR3 controller
 ┌──────────────────────────────────────────────────────┐
 │ waypoints (fingertip poses)                          │
 │   → Path: straight segments, corners rounded (3 cm)  │
 │   → IK every 2 mm (numpy, checked before moving)     │   XML-RPC :20003
 │   → speed profile: tool m/s, joint 60°/s, 0.3 m/s²   │ ──────────────────►  ServoJ (8 ms)  → motors
 │   → stream ServoJ at 125 Hz                          │
 │ watchdog: measured joints vs recent commands         │ ◄──────────────────  state stream :20004 (~100 Hz)
 └──────────────────────────────────────────────────────┘   GetActualJointPosDegree
```

- **No SDK**: the C++/Python SDKs are thin wrappers around the controller's XML-RPC server. This repo calls
  it directly with `xmlrpc.client` and parses the binary state stream (port 20004) itself.
  SDK는 컨트롤러 XML-RPC를 감싼 얇은 포장입니다. 여기서는 `xmlrpc.client`로 직접 호출합니다.
- **ServoJ vs MoveJ**: `MoveJ` lets the controller plan each move (it stops at every waypoint and the tool
  path between waypoints is curved). Here the PC plans one continuous path and streams joint targets,
  so vertical descents are exactly vertical and the arm does not stop between waypoints.
  `MoveJ`는 경유점마다 멈추고 그 사이 손끝 경로가 휘지만, 여기서는 PC가 한 번에 계획해서 직선 그대로 멈추지 않고 이동합니다.
- **Smooth stop**: pressing SPACE (example) decelerates along the path instead of halting abruptly.
- **Not MoveIt**: there is no collision checking beyond simple height checks. For cluttered scenes, plan
  with MoveIt and stream the result with `ServoTrajectory`.
  MoveIt을 쓰지 않으므로 충돌 검사는 높이 검사뿐입니다.

## Layout / 구성

```
fairino_fr3/      core library (numpy only)
  robot.py        XML-RPC client: state stream parsing, enable/auto mode, MoveJ, gripper
  kinematics.py   FK / Jacobian / damped-least-squares IK (FR3 v6 parameters embedded)
  servo_traj.py   Path (corner blending) + ServoTrajectory (IK, speed profile, 125 Hz ServoJ, smooth stop, watchdog)
scripts/
  startup.py      after power-on: enable + automatic mode + gripper activation (arm does not move)
  go_pose.py      MoveJ to a joint pose with path-height and FK cross-checks
  servo_line.py   smallest ServoJ example: flange up and down in a straight line
calib/
  make_charuco.py print the ChArUco board (A4)
  calib_touch.py  fixed camera ↔ robot calibration by touching 4 board corners with the fingertip
examples/
  color_pick.py   colour pick-and-place: one RGB camera + table plane (no depth) + ServoJ motion
tests/
  test_offline.py kinematics and trajectory checks without robot or camera
```

## Network / 네트워크

Network: put the PC on the controller's subnet (e.g. `192.168.58.20/24`). The control box has two LAN
ports; if the teach pendant uses one, connect the PC to the other.
네트워크: PC를 컨트롤러와 같은 대역에 둡니다. 컨트롤 박스의 LAN 포트가 둘이면 펜던트가 쓰지 않는 쪽에 연결합니다.

## Usage / 사용법

```bash
python3 scripts/startup.py --check       # read-only state
python3 scripts/startup.py               # enable + auto mode + gripper activation
python3 scripts/go_pose.py --pose ready --check
python3 scripts/go_pose.py --pose ready  # MoveJ at 10 %
python3 scripts/servo_line.py            # plan only
python3 scripts/servo_line.py --execute  # 5 cm up and down with ServoJ (Enter = smooth stop)
```

As a library / 라이브러리로:

```python
import numpy as np
from fairino_fr3 import FR3, Path, ServoTrajectory

robot = FR3("192.168.58.2")
q0 = np.radians(robot.q())
T0 = robot.kin.fk(q0)                      # wrist3_link pose (flange = +101.4 mm along z)
T1 = T0.copy(); T1[0, 3] += 0.10           # 10 cm along base x
T2 = T1.copy(); T2[2, 3] -= 0.05           # then 5 cm down, corner rounded
traj = ServoTrajectory(robot.kin, Path([T0, T1, T2], speeds=[0.05, 0.03]), q0)
print(f"{traj.duration:.1f} s")            # IK and limits are checked here, before anything moves
traj.execute(robot.ip)                     # streams ServoJ at 125 Hz
robot.gripper(0)                           # close (100 = open)
```

A path must not turn back on itself (e.g. up then down): split it into separate trajectories.
경로가 되돌아가면(올라갔다 내려오기) 궤적을 두 개로 나누세요.

### Camera calibration and colour pick / 카메라 보정과 색 물체 집기

```bash
python3 calib/make_charuco.py                       # print at 100 %, measure a square
python3 calib/calib_touch.py --tip-mm 240 --square-mm 35
python3 examples/color_pick.py --dry-run            # live camera, robot not commanded, saves the planned path image
python3 examples/color_pick.py                      # keys: 1 pink 2 blue 3 green 4 yellow, 0 start pose, SPACE stop
```

Edit the "cell setup" constants in `examples/color_pick.py` (start pose, reach, workspace, gripper lengths)
for your table before running it. Our start pose is specific to our table and may collide in yours.
`examples/color_pick.py`의 "cell setup" 상수(시작 자세, 작업 반경, 그리퍼 길이)를 반드시 자기 환경에 맞게 고치세요.

## Controller notes (v3.8.6) / 컨트롤러 메모

| item | value |
|---|---|
| XML-RPC | `http://<ip>:20003/RPC2` |
| State stream | TCP 20004, ~100 Hz, packets start with `0x5A 0x5A` |
| status (offset 2) | `<BHBBiiB`: frame count, length, program state, robot state, main error, sub error, mode (0 = auto) |
| e-stop | byte 549 |
| gripper motion done | byte 554 |
| gripper (offset 566) | `<BHHB`: fault id, fault, active (bit 0), position % |
| `MoveJ` | `MoveJ(q_deg[6], desc_pose[6], tool, user, vel%, acc, ovl, epos[4], blendT, offset_flag, offset_pos[6])`, `desc_pose` from `GetForwardKin(q)[1:7]` |
| `ServoJ` | `ServoJ(q_deg[6], epos[4], acc, vel, cmdT, filterT, gain, id)` inside `ServoMoveStart()` / `ServoMoveEnd()` |
| `MoveGripper` | `MoveGripper(index, pos%, vel%, force%, maxtime_ms, block, type, rotNum, rotVel, rotTorque)`, 100 = open |
| gripper activation | `ActGripper(1, 0)` then `ActGripper(1, 1)` |

Pitfalls we hit / 겪은 문제:
- **Do not call `system.listMethods`** over XML-RPC: the controller's ports 20003/20004 closed right after it.
  Do not probe the ports with raw writes either.
  `system.listMethods`를 호출하지 마세요. 직후 컨트롤러 포트가 닫혔습니다.
- External motion (`MoveJ`, `ServoJ`) works only in automatic mode.
- `ServoCart` (controller-side IK) raised joint over-speed errors (1/25) near singular poses; IK on the PC
  with joint-speed limits avoided it.
- The gripper position read back is unreliable: it reads 0 until the first command after power-up and does
  not show whether an object is held. Do not use it as a grasp check.
- With the D455, colour + depth together dropped off USB on our cable/port unless the IR emitter was off,
  so the example uses colour only and intersects pixel rays with the calibrated table plane.
- The DH AG fingers move about 4 cm down while closing (flange face → fingertip: 200 mm open, 240 mm closed).

## Limitations / 한계

- Verified on one FR3 with controller v3.8.6 only.
- Collision checking is limited to height checks; no obstacle model.
- The colour example assumes objects of known height on a flat table, a vertical top grasp and a single
  fixed camera (height is not observable from one view).
- Moving the camera requires recalibration unless you re-detect the board and reuse `T_base_board`.

## Acknowledgements

FAIRINO and DH Robotics are trademarks of their owners; this project is not affiliated with them.
The FR3 joint parameters in `fairino_fr3/kinematics.py` are taken from `fairino3_v6.urdf` in
[frcobot_ros2](https://github.com/FAIR-INNOVATION/frcobot_ros2).

## License

MIT
