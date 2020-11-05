#!/usr/bin/env python3

from code.make_env import make_training_env
from pybullet_planning.interfaces.robots.collision import get_collision_fn
from code.grasping import Transform, CoulombFriction, Cube
from code.utils import sample_from_normal_cube, set_seed
import argparse
import itertools
import numpy as np
import pybullet as p
import time
from code.const import COLLISION_TOLERANCE


def sample(n, cube_halfwidth, cube_ori, shrink_region=0.6):
    points_world_frame = np.array([sample_from_normal_cube(cube_halfwidth,
                                                           shrink_region=shrink_region,
                                                           avoid_top=True)
                                  for _ in range(n)])
    R_base_to_cube = Transform(np.zeros(3), cube_ori).inverse()
    points_cube_frame = R_base_to_cube(points_world_frame)
    faces = []
    for point in points_cube_frame:
        axis = np.argmax(np.abs(point))
        sign = np.sign(point[axis])
        if axis == 2:
            faces.append(-1 if sign == 1 else -2)
        elif axis == 1:
            faces.append(0 if sign == 1 else 2)
        elif axis == 0:
            faces.append(1 if sign == 1 else 3)
        else:
            raise ValueError("SOMETHING WENT WRONG")

    points = np.array([sample_from_normal_cube(cube_halfwidth,
                                               shrink_region=shrink_region,
                                               face=face,
                                               sample_from_all_faces=True)
                       for face in faces])

    return points


def get_heurisic_grasps(cube_halfwidth, cube_ori):
    points = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0]
    ])
    R_base_to_cube = Transform(np.zeros(3), cube_ori).inverse()
    points = R_base_to_cube(points)
    faces = []
    for point in points:
        axis = np.argmax(np.abs(point))
        sign = np.sign(point[axis])
        if axis == 2:
            faces.append(-1 if sign == 1 else -2)
        elif axis == 1:
            faces.append(0 if sign == 1 else 2)
        elif axis == 0:
            faces.append(1 if sign == 1 else 3)
        else:
            raise ValueError("SOMETHING WENT WRONG")
    # get face centers in cube frame
    points = np.array([sample_from_normal_cube(cube_halfwidth,
                                               shrink_region=0.0,
                                               face=face,
                                               sample_from_all_faces=True)
                       for face in faces])
    grasps = []
    for ind in range(4):
        grasps.append(points[np.array([x for x in range(4) if x != ind])])
    return grasps


class GraspSampler(object):
    def __init__(self, env, obs, mu=1.0, slacky_collision=False):
        from code.const import INIT_JOINT_CONF
        from code.utils import IKUtils
        self.cube = Cube(0.0325, CoulombFriction(mu=mu))
        self.cube_ori = obs['object_orientation']
        self.ik = env.pinocchio_utils.inverse_kinematics
        self.id = env.platform.simfinger.finger_id
        self.tip_ids = env.platform.simfinger.pybullet_tip_link_indices
        self.link_ids = env.platform.simfinger.pybullet_link_indices
        self.T_cube_to_base = Transform(obs['object_position'],
                                        obs['object_orientation'])
        self.q_init = obs['robot_position']
        self.v_init = obs['robot_velocity']
        self.tips_init = obs['robot_tip_positions']
        self.env = env
        self.ik_utils = IKUtils(env)
        self.slacky_collision = slacky_collision
        self._org_tips_init = np.array(self.env.platform.forward_kinematics(INIT_JOINT_CONF))

    def _reject(self, points):
        contacts = [self.cube.contact_from_tip_position(point)
                    for point in points]
        if not self.cube.force_closure_test(contacts):
            print("GRASPING: Not in Force Closure.")
            return True, None
        points_base = self.T_cube_to_base(points)
        print("GRASP POINTS:")
        print(points_base)
        qs = self.ik_utils.sample_no_collision_ik(points_base, sort_tips=False,
                                                  slacky_collision=self.slacky_collision)
        if len(qs) == 0:
            print("GRASPING: No IK Solution.")
            return True, None
        return False, qs[0]

    def _assign_positions_to_fingers(self, tips):

        min_cost = 1000000
        opt_tips = []
        opt_inds = [0, 1, 2]
        for v in itertools.permutations([0, 1, 2]):
            sorted_tips = tips[v, :]
            cost = np.linalg.norm(sorted_tips - self._org_tips_init) #  self.tips_init)
            if min_cost > cost:
                min_cost = cost
                opt_tips = sorted_tips
                opt_inds = v

        return opt_tips, opt_inds

    def __call__(self, cube_halfwidth, shrink_region=0.6, max_retries=300):
        retry = 0
        while retry < max_retries:
            if cube_halfwidth < 0.0325:
                raise ValueError('cube_halwidth must be larger than 0.0325')
            points = sample(3, cube_halfwidth, self.cube_ori,
                            shrink_region=shrink_region)
            tips = self.T_cube_to_base(points)
            tips, inds = self._assign_positions_to_fingers(tips)
            points = points[inds, :]
            should_reject, q = self._reject(points)
            if not should_reject:
                self.env.platform.simfinger.reset_finger_positions_and_velocities(self.q_init, self.v_init)  # TEMP: this line lacks somewhere in this class..
                return points, tips, q
            retry += 1
        raise RuntimeError('No feasible grasp is found.')

    def get_heurisic_grasps(self, cube_halwidth):
        grasps = get_heurisic_grasps(cube_halwidth, self.cube_ori)
        valid_grasps = []
        for points in grasps:
            tips = self.T_cube_to_base(points)
            for inds in itertools.permutations([0, 1, 2]):
                sorted_tips = tips[inds, :]
                sorted_points = points[inds, :]
                should_reject, q = self._reject(sorted_points)
                if not should_reject:
                    self.env.platform.simfinger.reset_finger_positions_and_velocities(self.q_init, self.v_init)  # TEMP: this line lacks somewhere in this class..
                    valid_grasps.append([sorted_points, sorted_tips, q])
        print("YAYAYAYYYA")
        print(len(valid_grasps))
        return valid_grasps
    # def get_heurisic_grasps(self, cube_halwidth):
    #     grasps = get_heurisic_grasps(cube_halwidth, self.cube_ori)
    #     valid_grasps = []
    #     for points in grasps:
    #         tips = self.T_cube_to_base(points)
    #         tips, inds = self._assign_positions_to_fingers(tips)
    #         points = points[inds, :]
    #         should_reject, q = self._reject(points)
    #         if not should_reject:
    #             self.env.platform.simfinger.reset_finger_positions_and_velocities(self.q_init, self.v_init)  # TEMP: this line lacks somewhere in this class..
    #             valid_grasps.append([points, tips, q])
    #     return valid_grasps




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--halfwidth", type=float, default=0.0425, help="cube half width (default: 0.0425)")
    args = parser.parse_args()
    reward_fn = 'task2_reward'
    termination_fn = 'position_close_to_goal'
    initializer = 'task2_init'
    set_seed(args.seed)

    env = make_training_env(reward_fn, termination_fn, initializer,
                            action_space='torque',
                            init_joint_conf=False,
                            visualization=True,
                            rank=args.seed)
    env = env.env

    obs = env.reset()
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                 cameraPitch=-40,
                                 cameraTargetPosition=[0, 0, 0])

    sampler = GraspSampler(env, obs)
    _, tips, q = sampler(cube_halfwidth=args.halfwidth, shrink_region=0.35)

    while (p.isConnected()):
        env.platform.simfinger.reset_finger_positions_and_velocities(q)
