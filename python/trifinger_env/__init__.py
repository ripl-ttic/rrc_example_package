from trifinger_env.cube_env import RealRobotCubeEnv, ActionType
from trifinger_env.wrappers import *


def get_initializer(name):
    from trifinger_env import initializers
    if hasattr(initializers, name):
        return getattr(initializers, name)
    else:
        raise ValueError(f"Can't find initializer: {name}")


def get_reward_fn(name):
    from trifinger_env import reward_fns
    if hasattr(reward_fns, name):
        return getattr(reward_fns, name)
    else:
        raise ValueError(f"Can't find reward function: {name}")


def get_termination_fn(name):
    from trifinger_env import termination_fns
    if hasattr(termination_fns, name):
        return getattr(termination_fns, name)
    elif hasattr(termination_fns, "generate_" + name):
        return getattr(termination_fns, "generate_" + name)()
    else:
        raise ValueError(f"Can't find termination function: {name}")


def make_training_env(cube_goal_pose, goal_difficulty, action_space, frameskip=1,
                      sim=False, visualization=False, reward_fn=None,
                      termination_fn=None, initializer=None,
                      residual=False, rank=0, monitor=False):
    is_level_4 = 'task4' in reward_fn
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
                           initializer=initializer)
    env.seed(seed=rank)
    env.action_space.seed(seed=rank)
    if visualization:
        env = PyBulletClearGUIWrapper(env)
    if monitor:
        from gym.wrappers import Monitor
        from code.const import TMP_VIDEO_DIR
        env = Monitor(
            RenderWrapper(env),
            TMP_VIDEO_DIR,
            video_callable=lambda episode_id: True,
            mode='evaluation'
        )
    if residual:
        if action_space == 'torque':
            # env = JointConfInitializationWrapper(env, heuristic=grasp)
            env = ResidualLearningFCWrapper(env, apply_torques=is_level_4,
                                            is_level_4=is_level_4)
        elif action_space == 'torque_and_position':
            env = ResidualLearningMotionPlanningFCWrapper(env, apply_torques=is_level_4,
                                                          action_repeat=2,
                                                          align_goal_ori=is_level_4,
                                                          use_rrt=is_level_4,
                                                          init_cube_manip='flip_and_grasp' if is_level_4 else 'grasp',
                                                          evaluation=False)
        else:
            raise ValueError(f"Can't do residual learning with {action_space}")
    env = FlatObservationWrapper(env)
    return env
