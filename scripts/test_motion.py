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

    obs = env.reset()
    motion = Motion(env)
    # motion.move_onto_floor()
    # motion.move_to_workspace_edge(5)
    motion.move_around_workspace_edge()

if __name__ == '__main__':
    main()
