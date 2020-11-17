#!/usr/bin/env python3
import pybullet as p
import numpy as np
import time
from code.const import INIT_JOINT_CONF, ARENA_RADIUS
from tracking.leap_pybullet_integration import LeapPyBullet, TransformAndScale

FINGER_1_SIDE = 'right'
FINGER_2_SIDE = 'right'
FINGER_3_SIDE = 'left'

FINGER_1_IND = 1
FINGER_2_IND = 0
FINGER_3_IND = 1


class FingerImitation(object):
    def __init__(self, env, obs, use_hand=False):
        self.env = env
        self.ik = self.env.pinocchio_utils.inverse_kinematics
        self.fk = self.env.pinocchio_utils.forward_kinematics
        self.init_obs = obs
        self.tips = np.array(self.fk(INIT_JOINT_CONF))
        self.T = TransformAndScale(pos=np.array([0, 0, -100.]),
                                   ori=np.array([1, 0, 0, 1]) * np.sqrt(2) / 2,
                                   x_scale=0.0016,
                                   y_scale=0.0016,
                                   z_scale=0.001)
        self.leap = LeapPyBullet(self.T, viz=True)

        self.robot_pos = obs['robot_position']
        self.hands = {'left': None, 'right': None}

    def get_tips(self):
        hands = self.leap.detect()
        for hand in hands:
            if hand['is_left']:
                self.hands['left'] = hand
            else:
                self.hands['right'] = hand
        if self.hands['left'] is None or self.hands['right'] is None:
            return None
        else:
            return np.stack([
                self.hands[FINGER_1_SIDE]['fingers'][FINGER_1_IND][-1],
                self.hands[FINGER_2_SIDE]['fingers'][FINGER_2_IND][-1],
                self.hands[FINGER_3_SIDE]['fingers'][FINGER_3_IND][-1]
            ])

    def _clip(self, tips):
        tips[:, -1] = np.maximum(tips[:, -1], 0.01)  # z >= 0
        return tips

    def get_action(self):
        tip_pos = self.get_tips()
        if tip_pos is None:
            return self.robot_pos.copy()
        tip_pos = self._clip(tip_pos)

        for i, tip in enumerate(tip_pos):
            if np.any(np.isnan(tip)):
                # no wrist detected
                continue
            tol, j = 0.001, None
            while j is None:
                j = self.ik(i, tip, INIT_JOINT_CONF, tol=tol)
                tol *= 2
            self.robot_pos[3*i:3*(i+1)] = j[3*i:3*(i+1)]
        return self.robot_pos.copy()


if __name__ == '__main__':
    from code.make_env import make_training_env
    from trifinger_simulation.tasks import move_cube
    env = make_training_env(move_cube.sample_goal(3).to_dict(), 3,
                            action_space='position',
                            frameskip=3,
                            sim=True,
                            visualization=True,
                            reward_fn='competition_reward',
                            termination_fn='position_close_to_goal',
                            initializer='training_init',
                            episode_length=3750,
                            residual=False,
                            randomize=False,
                            skip_motions=True).env

    obs = env.reset()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=0,
                                 cameraPitch=-89.99,
                                 cameraTargetPosition=[0, 0, 0])

    actor = FingerImitation(env, obs, use_hand=False)

    time.sleep(1)

    for _ in range(10):
        obs = env.reset()
        actor.leap.reset()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=0,
                                     cameraPitch=-89.99,
                                     cameraTargetPosition=[0, 0, 0])
        done = False
        while not done:
            action = actor.get_action()
            _, _, done, _ = env.step(action)
            time.sleep(0.025)
