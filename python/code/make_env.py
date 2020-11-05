from code.env.cube_env import RealRobotCubeEnv, ActionType
from code import wrappers


def get_initializer(name):
    from code.env import initializers
    if name is None:
        return None
    if hasattr(initializers, name):
        return getattr(initializers, name)
    else:
        raise ValueError(f"Can't find initializer: {name}")


def get_reward_fn(name):
    from code.env import reward_fns
    if name is None:
        return reward_fns.competition_reward
    if hasattr(reward_fns, name):
        return getattr(reward_fns, name)
    else:
        raise ValueError(f"Can't find reward function: {name}")


def get_termination_fn(name):
    from code.env import termination_fns
    if name is None:
        return None
    if hasattr(termination_fns, name):
        return getattr(termination_fns, name)
    elif hasattr(termination_fns, "generate_" + name):
        return getattr(termination_fns, "generate_" + name)()
    else:
        raise ValueError(f"Can't find termination function: {name}")


def make_training_env(cube_goal_pose, goal_difficulty, action_space, frameskip=1,
                      sim=False, visualization=False, reward_fn=None,
                      termination_fn=None, initializer=None, episode_length=120000,
                      residual=False, rank=0, monitor=False, randomize=False):
    is_level_4 = goal_difficulty == 4
    reward_fn = get_reward_fn(reward_fn)
    initializer = get_initializer(initializer)(goal_difficulty)
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
    env = wrappers.NewToOldObsWrapper(env)
    env = wrappers.InitStayHoldWrapper(env)
    if not sim:
        env = wrappers.AlignedInitCubeWrapper(env)
    if visualization:
        env = wrappers.PyBulletClearGUIWrapper(env)
    if monitor:
        from gym.wrappers import Monitor
        from code.const import TMP_VIDEO_DIR
        env = Monitor(
            wrappers.RenderWrapper(env),
            TMP_VIDEO_DIR,
            video_callable=lambda episode_id: True,
            mode='evaluation'
        )
    if randomize and sim:
        env = wrappers.RandomizedEnvWrapper(env, visualize=visualization)
    if residual:
        if action_space == 'torque':
            # env = JointConfInitializationWrapper(env, heuristic=grasp)
            env = wrappers.ResidualLearningFCWrapper(env, apply_torques=is_level_4,
                                                     is_level_4=is_level_4)
        elif action_space == 'torque_and_position':
            env = wrappers.ResidualLearningMotionPlanningFCWrapper(
                env,
                apply_torques=is_level_4,
                action_repeat=2,
                align_goal_ori=is_level_4,
                use_rrt=is_level_4,
                init_cube_manip='flip_and_grasp' if is_level_4 else 'grasp',
                evaluation=False
            )
        else:
            raise ValueError(f"Can't do residual learning with {action_space}")
    env = wrappers.FlatObservationWrapper(env)
    return env
