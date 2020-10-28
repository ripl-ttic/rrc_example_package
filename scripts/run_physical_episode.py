#!/usr/bin/env python3

import sys
import json
import numpy as np
from code.make_env import make_training_env
from trifinger_simulation.tasks import move_cube
from code.env.cube_env import ActionType


def main(difficulty, goal_dict):
    if difficulty is None and goal_dict is None:
        goal = move_cube.sample_goal(3)
        goal_dict = {
            'position': goal.position,
            'orientation': goal.orientation
        }
    print('difficulty:', difficulty)
    print('goal_dict:', goal_dict)

    difficulty = 3  # TEMP
    eval_config = {
        'action_space': 'torque_and_position',
        'frameskip': 3,
        'residual': True,
        'reward_fn': f'task{difficulty}_competition_reward',
        'termination_fn': 'no_termination',
        'initializer': f'task{difficulty}_init',
        'monitor': False,
        'rank': 0,
        'episode_length': 20000
    }
    env = make_training_env(goal_dict, difficulty, sim=True, visualization=True,
                            **eval_config)

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

    while not done:
        obs, reward, done, info = env.step(zero_action)
        accumulated_reward += reward
    print("Accumulated reward: {}".format(accumulated_reward))


if __name__ == '__main__':
    difficulty = None
    goal = None
    if len(sys.argv) > 2:
        difficulty = int(sys.argv[1])
        goal_pose_json = sys.argv[2]
        goal = json.loads(goal_pose_json)
    main(difficulty, goal)
