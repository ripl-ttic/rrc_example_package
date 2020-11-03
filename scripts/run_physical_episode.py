#!/usr/bin/env python3

import sys
import json
import numpy as np
from code.make_env import make_training_env
from trifinger_simulation.tasks import move_cube
from code.env.cube_env import ActionType

from code.make_env import get_reward_fn, get_initializer, get_termination_fn
from code.env.cube_env import RealRobotCubeEnv, ActionType
from code.wrappers import *

def make_pure_env(cube_goal_pose, goal_difficulty, action_space, frameskip=1,
                      sim=False, visualization=False, reward_fn=None,
                      termination_fn=None, initializer=None, episode_length=120000,
                      residual=False, rank=0, monitor=False):
    is_level_4 = goal_difficulty == 4
    reward_fn = get_reward_fn(reward_fn)
    initializer = get_initializer(initializer)
    termination_fn = get_termination_fn(termination_fn)
    if action_space not in ['torque', 'position', 'torque_and_position', 'position_and_torque']:
        raise ValueError(f"Unknown action space: {action_space}.")
    if action_space == 'torque':
        action_type = ActionType.TORQUE
    elif action_space in ['torque_and_position', 'position_and_torque']:
        action_type = ActionType.TORQUE_AND_POSITION
    else:
        action_type = ActionType.POSITION
    env = RealRobotCubeEnv(cube_goal_pose,
                           goal_difficulty,
                           action_type=action_type,
                           frameskip=frameskip,
                           sim=sim,
                           visualization=visualization,
                           reward_fn=reward_fn,
                           termination_fn=termination_fn,
                           initializer=initializer,
                           episode_length=episode_length)
    env.seed(seed=rank)
    env.action_space.seed(seed=rank)

def main(difficulty, goal_dict):
    if difficulty is None and goal_dict is None:
        difficulty = 3
        goal = move_cube.sample_goal(difficulty)
        goal_dict = {
            'position': goal.position,
            'orientation': goal.orientation
        }
    print('difficulty:', difficulty)
    print('goal_dict:', goal_dict)

    eval_config = {
        'action_space': 'torque',
        'frameskip': 3,
        'residual': True,
        'reward_fn': 'competition_reward',
        'termination_fn': 'no_termination',
        'initializer': 'random_init',
        'monitor': False,
        'rank': 0,
        'episode_length': 100000
    }
    if difficulty in [1, 2, 3]:
        eval_config['action_space'] = 'torque'
    else:
        eval_config['action_space'] = 'torque_and_position'
    env = make_training_env(goal_dict, difficulty, sim=False, visualization=False,
                            **eval_config)
    # env = make_pure_env(goal_dict, difficulty, sim=False, visualization=False,
    #                     **eval_config)

    obs = env.reset()
    done = False
    accumulated_reward = 0
    if env.action_type == ActionType.TORQUE_AND_POSITION:
        zero_action = {
            'torque': (env.action_space['torque'].sample() * 0).astype(np.float64),
            'position': (env.action_space['position'].sample() * 0).astype(np.float64)
        }
        # assert zero_action['torque'].dtype == np.float64
        # assert zero_action['position'].dtype == np.float64
    else:
        zero_action = np.array(env.action_space.sample() * 0).astype(np.float64)
        # assert zero_action.dtype == np.float64

    import random
    import time
    counter = 0
    while not done:
        obs, reward, done, info = env.step(zero_action)
        accumulated_reward += reward
        counter += 1
    print("Accumulated reward: {}".format(accumulated_reward))


if __name__ == '__main__':
    difficulty = None
    goal = None
    if len(sys.argv) > 2:
        difficulty = int(sys.argv[1])
        goal_pose_json = sys.argv[2]
        goal = json.loads(goal_pose_json)
    main(difficulty, goal)
