#!/usr/bin/env python3
import pybullet as p
import numpy as np
import time
import os
import torch
from scipy.spatial.transform import Rotation

from env import CubeEnv
from finger_imitation import FingerImitation


class EpisodeManager(object):
    def __init__(self, datadir):
        self.datadir = datadir
        if os.path.exists(self.datadir):
            files = [os.path.join(datadir, f) for f in os.listdir(datadir)]
            files = [f for f in files if os.path.isfile(f)]
            inds = [os.path.basename(f).split('.')[0] for f in files]
            if len(inds) == 0:
                self._ind = 0
            else:
                self._ind = max([int(ind) for ind in inds]) + 1
        else:
            os.makedirs(self.datadir, exist_ok=True)
            self._ind = 0

    def save(self, episode):
        path = os.path.join(self.datadir, f'{self._ind}.pt')
        torch.save(episode, path)
        self._ind += 1

    def load(self, ind=None):
        if self._ind == 0:
            raise ValueError("No files exist.")
        if ind is None:
            ind = np.random.randint(0, self._ind)

        else:
            if ind >= self._ind or ind < 0:
                raise ValueError("Invalid file index. Should be between 0 and {self.ind}. Got {ind}.")
        return torch.load(os.path.join(self.datadir, f"{ind}.pt"))

    def get_neps(self):
        return self._ind


class DataCollection(object):
    def __init__(self, datadir, neps, difficulty=3, frameskip=3, step_limit=500):
        self.episode_manager = EpisodeManager(os.path.join(datadir, f'{difficulty}'))
        self.difficulty = difficulty
        self.frameskip = frameskip
        self.env = CubeEnv(difficulty, visualization=True, frameskip=frameskip)
        obs = self.env.reset()
        self.actor = FingerImitation(self.env, obs)
        self.completed_episodes = 0
        self.min_time_between_actions = frameskip * self.env.platform.simfinger.time_step_s
        self.step_limit = step_limit
        self.neps = neps

    def _position_at_goal(self, obs):
        dist_to_goal = np.linalg.norm(
            obs["desired_goal"]["position"]
            - obs["achieved_goal"]["position"]
        )
        return dist_to_goal < 0.01

    def _pose_at_goal(self, obs):
        goal_rot = Rotation.from_quat(obs['desired_goal']['orientation'])
        actual_rot = Rotation.from_quat(obs['achieved_goal']['orientation'])
        error_rot = goal_rot.inv() * actual_rot
        rot_error_deg = error_rot.magnitude() / np.pi
        return rot_error_deg < 5 and self._position_at_goal(obs)

    def at_goal(self, obs):
        if self.difficulty == 4:
            return self._pose_at_goal(obs)
        else:
            return self._position_at_goal(obs)

    def run_episode(self):
        episode_data = {'observations': [], 'actions': [], 'rewards': []}
        obs = self.env.reset()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=0,
                                     cameraPitch=-89.99,
                                     cameraTargetPosition=[0, 0, 0])
        done = False
        # open window
        self.actor.get_action()
        print("Sleeping 5 seconds!")
        time.sleep(5)
        step = 0
        while not done and step < self.step_limit:
            t1 = time.time()
            action = self.actor.get_action()
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            obs, reward, done, _ = self.env.step(action)
            episode_data['rewards'].append(reward)
            done = done or self.at_goal(obs)
            t2 = time.time()
            time.sleep(max(0, self.min_time_between_actions - (t2 - t1)))
            step += 1

        ans = None
        while ans not in ['y', 'n']:
            print("Keep demonstration? type 'y' or 'n'")
            ans = input()
        if ans == 'y':
            self.episode_manager.save(episode_data)
            self.completed_episodes += 1

    def run_episodes(self):
        while self.completed_episodes < self.neps:
            self.run_episode()


if __name__ == '__main__':
    dc = DataCollection('./data', 25, 1)
    dc.run_episodes()
