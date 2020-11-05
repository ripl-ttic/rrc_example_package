#!/usr/bin/env python3
"""Example evaluation script to evaluate a policy.

This is an example evaluation script which loads a policy trained with PPO. If
this script were moved into the top rrc_simulation folder (since this is where
we will execute the rrc_evaluate command), it would consistute a valid
submission (naturally, imports below would have to be adjusted accordingly).

This script will be executed in an automated procedure.  For this to work, make
sure you do not change the overall structure of the script!

This script expects the following arguments in the given order:
 - Difficulty level (needed for reward computation)
 - initial pose of the cube (as JSON string)
 - goal pose of the cube (as JSON string)
 - file to which the action log is written

It is then expected to initialize the environment with the given initial pose
and execute exactly one episode with the policy that is to be evaluated.

When finished, the action log, which is created by the TriFingerPlatform class,
is written to the specified file.  This log file is crucial as it is used to
evaluate the actual performance of the policy.
"""
import os
import sys
import json

from code.env import cube_env
from trifinger_simulation.tasks import move_cube

from code.make_env import make_training_env
import dl
from dl import nest
from dl.rl import set_env_to_eval_mode
import torch
from code.residual_ppo import ResidualPPO2


def _init_env_and_policy(goal_pose_json, difficulty):
    if difficulty in [1, 2, 3, 4]:
        # HACK to get the path to the root directory
        root_dir = os.path.dirname(os.path.realpath(__file__))
        if difficulty == 4:
            expdir = os.path.join(root_dir, f'../models/mpfc_level_{difficulty}')
        else:
            expdir = os.path.join(root_dir, f'../models/fc_level_{difficulty}')
        bindings = [
            f'make_pybullet_env.goal_pose={goal_pose_json}',
            f'make_pybullet_env.goal_difficulty={difficulty}',
            f'make_pybullet_env.reward_fn="task{difficulty}_competition_reward"',
            'make_pybullet_env.termination_fn="no_termination"',
            f'make_pybullet_env.initializer="random_init"',
            'make_pybullet_env.visualization=False',
            'make_pybullet_env.monitor=False',
            'make_pybullet_env.sim=False',
        ]
        from code.utils import set_seed
        set_seed(0)
        dl.load_config(expdir + '/config.gin', bindings)
        ppo = ResidualPPO2(expdir, nenv=1)
        ppo.load()
        env = ppo.env
        set_env_to_eval_mode(env)
        # override env with fixed initializer
        # env.unwrapped.envs[0].unwrapped.initializer = initializer
        return env, ppo

    else:
        eval_config = {
            'action_space': 'torque_and_position',
            'frameskip': 3,
            'residual': True,
            'reward_fn': f'task{difficulty}_competition_reward',
            'termination_fn': 'no_termination',
            'initializer': 'random_init',
            'monitor': False,
            'rank': 0
        }

        from code.utils import set_seed
        set_seed(0)
        goal_pose_dict = json.loads(goal_pose_json)
        env = make_training_env(goal_pose_dict, difficulty, sim=False, visualization=False, **eval_config)
        # override env with fixed initializer
        # env.unwrapped.initializer = initializer
        return env, None


def main():
    difficulty = int(sys.argv[1])
    if len(sys.argv) == 3:
        goal_pose_json = sys.argv[2]
        # goal_pose = move_cube.Pose.from_json(goal_pose_json)
    else:
        goal_pose = move_cube.sample_goal(difficulty)
        goal_pose_json = json.dumps({
            'position': goal_pose.position.tolist(),
            'orientation': goal_pose.orientation.tolist()
        })

    # the poses are passed as JSON strings, so they need to be converted first
    # initial_pose = move_cube.sample_goal(-1)  # whatever. It'll be overwritten anyway

    # create a FixedInitializer with the given values
    # initializer = cube_env.FixedInitializer(
    #     difficulty, initial_pose, goal_pose
    # )

    env, ppo = _init_env_and_policy(goal_pose_json, difficulty)
    if difficulty in [1, 2, 3, 4]:
        # run residual policy
        obs = env.reset()
        done = False
        accumulated_reward = 0
        while not done:
            obs = torch.from_numpy(obs).float().to(ppo.device)
            with torch.no_grad():
                action = ppo.pi(obs, deterministic=True).action
                action = nest.map_structure(lambda x: x.cpu().numpy(), action)
            obs, reward, done, info = env.step(action)
            accumulated_reward += reward
        print("Accumulated reward: {}".format(accumulated_reward))
        # env.unwrapped.envs[0].unwrapped.platform.store_action_log(output_file)

    else:
        # run base policy (using zero residual action)
        obs = env.reset()
        done = False
        accumulated_reward = 0
        import numpy as np
        if env.action_type == cube_env.ActionType.TORQUE_AND_POSITION:
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
        # env.unwrapped.platform.store_action_log(output_file)


if __name__ == "__main__":
    main()
