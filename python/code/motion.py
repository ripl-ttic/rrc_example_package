#!/usr/bin/env python3
import numpy as np
from code.const import ARENA_RADIUS, INIT_JOINT_CONF
from code.utils import IKUtils, repeat

class Motion:
    def __init__(self, env):
        self.env = env
        self.vis_markers = None
        self.ik_utils = IKUtils(env)
        if self.env.visualization:
            from code.utils import VisualMarkers
            self.vis_markers = VisualMarkers()

    def move_onto_floor(self):
        tip_positions = np.array([0.1, 0, 0.04, -0.1, 0, 0.04, 0, 0.1, 0.04]).reshape(3, 3)
        if self.env.visualization:
            self.vis_markers.add(tip_positions)

        # sols = self.ik_utils.sample_no_collision_ik(tip_positions, sort_tips=True)
        sols = self.ik_utils.sample_ik(tip_positions, sort_tips=True)
        # actions = self.tip_positions_to_actions([tip_positions])
        # print('actions', actions)
        # obs = self.run_actions(repeat(actions, 1000))
        obs = self.run_actions(repeat([sols[0]], 1000))
        return obs

    def sample_edge(self):
        sols = []
        while len(sols) == 0:
            thetas = np.random.random(3) * 2 * np.pi
            xs = ARENA_RADIUS * np.cos(thetas) * 0.9
            ys = ARENA_RADIUS * np.sin(thetas) * 0.9
            zs = np.ones(xs.shape) * 0.02
            tip_positions = np.concatenate((xs, ys, zs)).reshape(3,3).T
            self.maybe_add_markers(tip_positions, color=(1, 0, 0, 0.5))
            sols = self.ik_utils.sample_ik(tip_positions, sort_tips=True)
            import pdb; pdb.set_trace()
        return sols[0]

    def get_edge_motion(self):
        resolution = 20
        actions = []
        for i in range(resolution):
            theta = i / resolution * (2 * np.pi) / 3
            thetas = np.array([theta, theta + (2 * np.pi) / 3, theta - (2 * np.pi) / 3])

            xs = ARENA_RADIUS * np.cos(thetas) * 0.9
            ys = ARENA_RADIUS * np.sin(thetas) * 0.9
            zs = np.ones(xs.shape) * 0.06
            tip_positions = np.concatenate((xs, ys, zs)).reshape(3,3).T
            self.maybe_add_markers(tip_positions, color=(1, 0, 0, 0.5))
            sols = self.ik_utils.sample_ik(tip_positions, sort_tips=True)
            if len(sols) > 0:
                actions.append(sols[0])
        print('len(actions)', len(actions))
        return actions

    def move_around_workspace_edge(self):
        actions = self.get_edge_motion()
        for action in actions:
            obs = self.run_actions(repeat([action], 40))
        return obs

    def move_to_workspace_edge(self, num_times=5):
        # print('high', self.env.action_space.high)
        # print('low', self.env.action_space.low)
        # print(sols[0])
        # print('contained?', self.env.action_space.contains(sols[0]))
        # print('actions', actions)
        for i in range(num_times):
            joint_conf = self.sample_edge()
            obs = self.run_actions(repeat([joint_conf], 1000))
        return obs

    def tip_positions_to_actions(self, tip_positions_list):
        ik = self.env.pinocchio_utils.inverse_kinematics

        actions = []
        for tip_positions in tip_positions_list:
            target_joint_conf = []
            for i in range(3):
                target_joint = ik(i, tip_positions[i], INIT_JOINT_CONF)
                try:
                    target_joint_conf.append(target_joint[3*i:3*(i+1)])
                except TypeError:
                    return actions
            action = np.concatenate(target_joint_conf)
            actions.append(action)

        return actions

    def maybe_add_markers(self, tip_positions, color=None):
        if self.env.visualization:
            self.vis_markers.remove()
            self.vis_markers.add(tip_positions, color=color)

    def run_actions(self, action_seq):
        for action in action_seq:
            print(action)
            action = np.asarray(action)
            obs, reward, done, info = self.env.step(action)
        return obs
