#!/usr/bin/env python3
from code.utils import apply_transform
from code.const import *
from code.align_rotation import pitch_rotation_axis_and_angle, project_cube_xy_plane
from scipy.spatial.transform import Rotation as R
import numpy as np
from code import utils

class ScriptedActions:
    def __init__(self, env, cube_tip_positions, orientation=None, vis_markers=None):
        self.env = env
        self.ik_utils = utils.IKUtils(env)
        self.cube_tip_positions = cube_tip_positions
        self.orientation = orientation
        self.tip_positions_list = []
        self.vis_markers = vis_markers
        self._markers = set()

    def _update_markers(self, target_tip_positions, marker_name, color=TRANSLU_CYAN):
        if self.vis_markers is not None:
            if marker_name in self._markers:
                self.vis_markers.remove()
            self.vis_markers.add(target_tip_positions, color=color)
            self._markers.add(marker_name)

    def add_grasp(self, obs, coef=0.9):
        grasp_target_cube_positions = self.cube_tip_positions * coef
        target_tip_positions = apply_transform(obs['object_position'], obs['object_orientation'], grasp_target_cube_positions)
        self._update_markers(target_tip_positions, 'grasp')
        tip_pos_sequence = utils.complete_keypoints(self.ik_utils, self.get_last_tippos(obs), target_tip_positions, unit_length=0.004)
        self.tip_positions_list += tip_pos_sequence

    def add_liftup(self, obs, height=0.0425, coef=0.6):
        grasp_target_cube_positions = self.cube_tip_positions * coef
        target_tip_positions = apply_transform(
            obs['object_position'] + np.array([0, 0, height]),
            obs['object_orientation'], grasp_target_cube_positions)
        self._update_markers(target_tip_positions, 'liftup')
        tip_pos_sequence = utils.complete_keypoints(self.ik_utils, self.get_last_tippos(obs), target_tip_positions, unit_length=0.004)
        self.tip_positions_list += tip_pos_sequence

    def add_pitch_rotation(self, obs, rotate_axis, rotate_angle, coef=0.6):
        grasp_target_cube_positions = self.cube_tip_positions * coef
        self.orientation = (R.from_quat(obs['object_orientation']) * R.from_euler(rotate_axis, rotate_angle)).as_quat()
        target_tip_positions = apply_transform(
            obs['object_position'] + np.array([0, 0, 0.0425]),
            self.orientation, grasp_target_cube_positions)
        self._update_markers(target_tip_positions, 'pitch')
        tip_pos_sequence = utils.complete_keypoints(self.ik_utils, self.get_last_tippos(obs), target_tip_positions, unit_length=0.004)
        self.tip_positions_list += tip_pos_sequence

    def add_yaw_rotation(self, obs, step_angle=np.pi/3):
        grasp_target_cube_positions = self.cube_tip_positions * 0.9
        angle = self._get_yaw_diff(obs)
        angle_clipped = np.clip(angle, -step_angle, step_angle)
        self.orientation = ( R.from_euler('Z', angle_clipped) * R.from_quat(obs['object_orientation'])).as_quat()
        target_tip_positions = apply_transform(obs['object_position'], self.orientation, grasp_target_cube_positions)
        self._update_markers(target_tip_positions, 'yaw')
        tip_pos_sequence = utils.complete_keypoints(self.ik_utils, self.get_last_tippos(obs), target_tip_positions, unit_length=0.002)
        self.tip_positions_list += tip_pos_sequence
        return angle_clipped

    def add_place_cube(self, obs, coef=0.6):
        assert self.orientation is not None
        grasp_target_cube_positions = self.cube_tip_positions * coef
        target_tip_positions = apply_transform(obs['object_position'],
                                            self.orientation,
                                            grasp_target_cube_positions)
        tip_pos_sequence = utils.complete_keypoints(self.ik_utils, self.get_last_tippos(obs), target_tip_positions, unit_length=0.004)
        self.tip_positions_list += tip_pos_sequence

    def add_release(self, obs, coef=2.0, min_allowed_height=0.02):
        assert self.orientation is not None
        grasp_target_cube_positions = self.cube_tip_positions * coef
        target_tip_positions = apply_transform(obs['object_position'], self.orientation, grasp_target_cube_positions)
        min_height = target_tip_positions[:, 2].min()
        if min_height < min_allowed_height:
            target_tip_positions[:, 2] += (min_allowed_height - min_height)
        tip_pos_sequence = utils.complete_keypoints(self.ik_utils, self.get_last_tippos(obs), target_tip_positions, unit_length=0.004)
        self.tip_positions_list += tip_pos_sequence

    def get_last_tippos(self, obs):
        if not self.tip_positions_list:
            return obs['robot_tip_positions']
        return self.tip_positions_list[-1]

    def get_tip_sequence(self):
        return self.tip_positions_list

    def _get_yaw_diff(self, obs):
        projected_goal_orientation = project_cube_xy_plane(obs['goal_object_orientation'])
        rot_diff = (R.from_quat(obs['object_orientation']) * R.from_quat(projected_goal_orientation).inv())
        yaw_diff = -rot_diff.as_euler('ZXY')[0]

        return yaw_diff
