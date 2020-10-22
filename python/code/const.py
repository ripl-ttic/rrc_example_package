#!/usr/bin/env python3
from trifinger_simulation.tasks import move_cube
from trifinger_simulation.trifinger_platform import TriFingerPlatform
from .env.cube_env import INIT_JOINT_CONF

COLLISION_TOLERANCE = 3.5 * 1e-03
MU = 0.5
VIRTUAL_CUBE_HALFWIDTH = 0.0395
CUBE_WIDTH = move_cube._CUBE_WIDTH
MIN_HEIGHT = move_cube._min_height
MAX_HEIGHT = move_cube._max_height
ARENA_RADUIS = move_cube._ARENA_RADIUS
# INIT_JOINT_CONF = TriFingerPlatform.spaces.robot_position.default

TMP_VIDEO_DIR = '/tmp/rrc_videos'

EXCEP_MSSG = "================= captured exception =================\n" + \
    "{message}\n" + "{error}\n" + '=================================='

# colors
TRANSLU_CYAN = (0, 1, 1, 0.4)
TRANSLU_YELLOW = (1, 1, 0, 0.4)
TRANSLU_BLUE = (0, 0, 1, 0.4)
TRANSLU_RED = (1, 0, 0, 0.4)
