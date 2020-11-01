#!/usr/bin/env python3

import numpy as np
from code.make_env import make_training_env
from trifinger_simulation.tasks import move_cube
from code.env.cube_env import ActionType
from code.motion import Motion

def main():
    goal = move_cube.sample_goal(3)
    goal_dict = {
        'position': goal.position,
        'orientation': goal.orientation
    }

    difficulty = 3
    eval_config = {
        'action_space': 'position',
        'frameskip': 3,
        'residual': False,
        'reward_fn': f'task{difficulty}_competition_reward',
        'termination_fn': 'no_termination',
        'initializer': f'task{difficulty}_init',
        'monitor': False,
        'rank': 0,
        'episode_length': 100000
    }
    env = make_training_env(goal_dict, difficulty, sim=False, visualization=False,
                            **eval_config)
    env = env.env.env  # HACK to remove FlatObservationWrapper and InitStayHoldWrapper
    print('env', env)

    obs = env.reset()
    env.register_custom_log('init_cube_pos', obs['object_position'])
    env.register_custom_log('init_cube_ori', obs['object_orientation'])
    env.register_custom_log('goal_pos', obs['goal_object_position'])
    env.register_custom_log('goal_ori', obs['goal_object_orientation'])
    env.save_custom_logs()  # manually save the log
    motion = Motion(env)
    motion.test_inv_dynamics()

if __name__ == '__main__':
    main()
