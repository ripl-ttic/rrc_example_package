#!/usr/bin/env python

import numpy as np
from trifinger_env.simulation.code.training_env import make_training_env
from trifinger_env.simulation.gym_wrapper.envs import cube_env
from trifinger_env.simulation.gym_wrapper.envs.cube_env import ActionType
from trifinger_env.simulation.tasks import move_cube

# env = make_training_env(visualization=False, **eval_config)
# env.unwrapped.initializer = initializer

# eval_config = {
#         'action_space': 'torque_and_position',
#         'frameskip': 3,
#         'residual': True,
#         'reward_fn': f'task{difficulty}_competition_reward',
#         'termination_fn': 'no_termination',
#         'initializer': f'task{difficulty}_init',
#         'monitor': False,
#         'rank': 0
# }

def main():
    goal = move_cube.sample_goal(3)
    goal_dict = {
    'position': goal.position,
    'orientation': goal.orientation
    }

    # env = cube_env.RealRobotCubeEnv(goal_dict, 3)
    difficulty = 3
    eval_config = {
            'action_space': 'torque_and_position',
            'frameskip': 3,
            'residual': True,
            'reward_fn': f'task{difficulty}_competition_reward',
            'termination_fn': 'no_termination',
            'initializer': f'task{difficulty}_init',
            'monitor': False,
            'rank': 0
    }
    env = make_training_env(visualization=True, **eval_config)

    obs = env.reset()
    done = False
    accumulated_reward = 0
    if env.action_type == ActionType.TORQUE_AND_POSITION:
            zero_action = {
            'torque': (env.action_space['torque'].sample() * 0).astype(np.float64),
            'position': (env.action_space['position'].sample() * 0).astype(np.float64)
            }
            assert zero_action['torque'].dtype == np.float64
            assert zero_action['position'].dtype == np.float64
    else:
            zero_action = np.array(env.action_space.sample() * 0).astype(np.float64)
            assert zero_action.dtype == np.float64

    while not done:
            obs, reward, done, info = env.step(zero_action)
            accumulated_reward += reward
    print("Accumulated reward: {}".format(accumulated_reward))
if __name__ == '__main__':
    main()
