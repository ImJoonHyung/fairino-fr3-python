"""FAIRINO FR3 + DH gripper control in plain Python (XML-RPC + ServoJ), no vendor SDK or ROS."""
from .kinematics import FLANGE_Z, FR3Kinematics
from .robot import DEFAULT_IP, FR3
from .servo_traj import Path, ServoTrajectory, Stopped

__all__ = ["FLANGE_Z", "FR3Kinematics", "DEFAULT_IP", "FR3", "Path", "ServoTrajectory", "Stopped"]
