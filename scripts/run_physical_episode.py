#!/usr/bin/env python3

import numpy as np
from trifinger_env.simulation.code.training_env import make_training_env
from trifinger_env.simulation.code.training_env.wrappers import PyBulletClearGUIWrapper, ResidualLearningFCWrapper,ResidualLearningFCWrapper,IKActionWrapper,ResidualLearningMotionPlanningFCWrapper, JointConfInitializationWrapper,CubeRotationAlignWrapper,FlatObservationWrapper
from trifinger_env.simulation.tasks import move_cube
from trifinger_env.cube_env import RealRobotCubeEnv, ActionType
from trifinger_env.reward_fns import competition_reward

def get_initializer(name):
    from trifinger_env.simulation.code.training_env import initializers
    if hasattr(initializers, name):
        return getattr(initializers, name)
    else:
        raise ValueError("Can't find initializer: {}".format(name))


def get_reward_fn(name):
    from trifinger_env.simulation.code.training_env import reward_fns
    if hasattr(reward_fns, name):
        return getattr(reward_fns, name)
    else:
        raise ValueError(f"Can't find reward function: {name}")


def get_termination_fn(name):
    from trifinger_env.simulation.code.training_env import termination_fns
    if hasattr(termination_fns, name):
        return getattr(termination_fns, name)
    elif hasattr(termination_fns, "generate_" + name):
        return getattr(termination_fns, "generate_" + name)()
    else:
        raise ValueError(f"Can't find termination function: {name}")

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

def make_physical_env(goal_dict, reward_fn, termination_fn, initializer, action_space,
                      init_joint_conf=False, residual=False, kp_coef=None,
                      kd_coef=None, frameskip=1, rank=0, visualization=False,
                      grasp='pinch', monitor=False):
    is_level_4 = 'task4' in reward_fn
    # reward_fn = get_reward_fn(reward_fn)
    reward_fn = competition_reward
    initializer = get_initializer(initializer)
    termination_fn = get_termination_fn(termination_fn)
    if action_space not in ['torque', 'position', 'ik', 'torque_and_position', 'position_and_torque']:
        raise ValueError("Action Space must be one of: 'torque', 'position', 'ik'.")
    if action_space == 'torque':
        action_type = ActionType.TORQUE
    elif action_space in ['torque_and_position', 'position_and_torque']:
        action_type = ActionType.TORQUE_AND_POSITION
    else:
        action_type = ActionType.POSITION
    # env = TrainingEnv(reward_fn=reward_fn,
    #                   termination_fn=termination_fn,
    #                   initializer=initializer,
    #                   kp_coef=kp_coef,
    #                   kd_coef=kd_coef,
    #                   frameskip=frameskip,
    #                   action_type=action_type,
    #                   visualization=visualization,
    #                   is_level_4=is_level_4)
    goal_difficulty = 3
    env = RealRobotCubeEnv(goal_dict, goal_difficulty, action_type,
                           frameskip=1, sim=False, visualization=visualization,
                           reward_fn=reward_fn, termination_fn=termination_fn,
                           initializer=initializer)
    env.seed(seed=rank)
    env.action_space.seed(seed=rank)
    if visualization:
        env = PyBulletClearGUIWrapper(env)
    if monitor:
        from gym.wrappers import Monitor
        from trifinger_env.simulation.code.training_env.wrappers import RenderWrapper
        from trifinger_env.simulation.code.const import TMP_VIDEO_DIR
        env = Monitor(
            RenderWrapper(env),
            TMP_VIDEO_DIR,
            video_callable=lambda episode_id: True,
            mode='evaluation'
        )

    if action_space == 'ik':
        env = IKActionWrapper(env)
    if residual:
        if action_space == 'torque':
            # env = JointConfInitializationWrapper(env, heuristic=grasp)
            env = ResidualLearningFCWrapper(env, apply_torques=is_level_4,
                                            is_level_4=is_level_4)
        elif action_space == 'torque_and_position':
            env = ResidualLearningMotionPlanningFCWrapper(env, apply_torques=is_level_4,
                                                          action_repeat=2, align_goal_ori=is_level_4,
                                                          use_rrt=is_level_4,
                                                          init_cube_manip='flip_and_grasp' if is_level_4 else 'grasp',
                                                          evaluation=False)
        else:
            raise ValueError(f"Can't do residual learning with {action_space}")
    else:
        if init_joint_conf:
            env = JointConfInitializationWrapper(env, heuristic=grasp)
            if is_level_4:
                env = CubeRotationAlignWrapper(env)
    env = FlatObservationWrapper(env)
    return env


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
    # env = make_training_env(visualization=True, **eval_config)
    env = make_physical_env(goal_dict, visualization=False, **eval_config)

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
