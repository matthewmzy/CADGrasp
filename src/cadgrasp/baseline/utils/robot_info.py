import numpy as np
import torch

from cadgrasp.baseline.utils.config import to_dot_dict

GRIPPER_HEIGHT = 0.02
GRIPPER_FINGER_WIDTH = 0.01
GRIPPER_TAIL_LENGTH = 0.04
GRIPPER_DEPTH_BASE = 0.02
GRIPPER_NEW_DEPTH = 0.04
GRIPPER_SPACE_LEFT = 0.01
GRIPPER_CENTER_SHIFT = 0.02
GRIPPER_MAX_WIDTH = 0.1

UR5_START_QPOS = np.deg2rad([15, -95, 75, -75, -90, -80])
