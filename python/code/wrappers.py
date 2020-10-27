"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import pybullet as p
import numpy as np
import gym
from trifinger_simulation import TriFingerPlatform
from trifinger_simulation import trifingerpro_limits
from trifinger_simulation import camera
from code.env.cube_env import ActionType
import cv2
from dl import nest
from code.const import INIT_JOINT_CONF

EXCEP_MSSG = "================= captured exception =================\n" + \
    "{message}\n" + "{error}\n" + '=================================='


class NewToOldObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_names = [
            "robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
            "goal_object_orientation",
            "tip_force",
        ]

        self.observation_space = gym.spaces.Dict(
            {
                "robot_position": env.observation_space['robot']['position'],
                "robot_velocity": env.observation_space['robot']['velocity'],
                "robot_tip_positions": env.observation_space['robot']['tip_positions'],
                "object_position": env.observation_space["achieved_goal"]["position"],
                "object_orientation": env.observation_space["achieved_goal"]["orientation"],
                "goal_object_position": env.observation_space["desired_goal"]["position"],
                "goal_object_orientation": env.observation_space["desired_goal"]["orientation"],
                "tip_force": env.observation_space["robot"]["tip_force"],
            }
        )

    def observation(self, obs):
        old_obs = {
            "robot_position": obs['robot']['position'],
            "robot_velocity": obs['robot']['velocity'],
            "robot_tip_positions": obs['robot']['tip_positions'],
            "tip_force": obs['robot']['tip_force'],
            "object_position": obs['achieved_goal']['position'],
            "object_orientation": obs['achieved_goal']['orientation'],
            "goal_object_position": obs['desired_goal']['position'],
            "goal_object_orientation": obs['desired_goal']['orientation'],
        }
        return old_obs

class InitStayHoldWrapper(gym.Wrapper):
    '''
    Oftentimes the initial pose of the robot is quite off from the robot_position.default.
    And that causes annoying issues.
    This wrapper forces to apply env.initial_action for the first "hold_steps" steps to properly reset the robot pose.
    '''
    def __init__(self, env, hold_steps=400):
        super().__init__(env)
        self.hold_steps = hold_steps

    def reset(self):
        obs = self.env.reset()

        # step environment for "hold_steps" steps
        counter = 0
        done = False
        initial_position = obs['robot_position']
        while not done and counter < self.hold_steps:
            desired_position = np.copy(initial_position)
            # close the bottom joint (joint 2) first, and then close other joints together (joint 0, joint 1)
            if counter < self.hold_steps / 2:
                # close joint 2
                for i in range(3):
                    desired_position[3 * i + 2] = -2.4   # joint 2 (default: -1.7, min: -2.7, max: 0.0)
                    desired_position[3 * i + 1] = 1.4   # joint 1 (default: 0.9, min: 0.0, max: 1.57)
            else:
                desired_position = INIT_JOINT_CONF

            if self.env.action_type == ActionType.TORQUE_AND_POSITION:
                action = {
                    'position': desired_position,
                    'torque': self.env.initial_action['torque']
                }
            elif self.env.action_type == ActionType.TORQUE:
                action = self.env.initial_action['torque']
            else:
                action = desired_position
            obs, reward, done, info = self.env.step(action)
            counter += 1

        return obs

class FlatObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = [
            self.observation_space[name].low.flatten()
            for name in self.observation_names
        ]

        high = [
            self.observation_space[name].high.flatten()
            for name in self.observation_names
        ]

        self.observation_space = gym.spaces.Box(
            low=np.concatenate(low), high=np.concatenate(high)
        )

    def observation(self, obs):
        observation = [obs[name].flatten() for name in self.observation_names]

        observation = np.concatenate(observation)
        return observation


class ResidualLearningFCWrapper(gym.Wrapper):
    '''
    Wrapper to perform residual policy learning on top of the scripted
    force control policy.
    Need JointConfInitializationWrapper under this wrapper.
    '''

    def __init__(self, env, apply_torques, evaluation=False, is_level_4=False):
        super().__init__(env)
        from code.cube_manipulator import CubeManipulator
        assert self.env.action_type == ActionType.TORQUE
        self.action_space = TriFingerPlatform.spaces.robot_torque.gym
        spaces = TriFingerPlatform.spaces
        self.observation_space = gym.spaces.Dict(
            {
                "robot_position": spaces.robot_position.gym,
                "robot_velocity": spaces.robot_velocity.gym,
                "robot_tip_positions": gym.spaces.Box(
                    low=np.array([spaces.object_position.low] * 3),
                    high=np.array([spaces.object_position.high] * 3),
                ),
                "object_position": spaces.object_position.gym,
                "object_orientation": spaces.object_orientation.gym,
                "goal_object_position": spaces.object_position.gym,
                "goal_object_orientation": spaces.object_orientation.gym,
                "tip_force": gym.spaces.Box(
                    low=np.zeros(3),
                    high=np.ones(3),
                ),
                "torque_action": spaces.robot_torque.gym,
            }
        )
        self.observation_names.append("torque_action")
        from code.fc_force_control import ForceControlPolicy
        self.pi = ForceControlPolicy(self.env, apply_torques=apply_torques)
        self.cube_manipulator = CubeManipulator(env)
        self.is_level_4 = is_level_4
        self.__evaluation = evaluation

    def _norm_actions(self, action):
        ts = TriFingerPlatform.spaces.robot_torque.gym
        return 2 * ((action - ts.low) / (ts.high - ts.low)) - 1

    def _add_action_to_obs(self, obs, ac=None):
        ts = TriFingerPlatform.spaces.robot_torque.gym
        if ac is None:
            obs['torque_action'] = np.zeros(ts.shape)
        else:
            obs['torque_action'] = self._norm_actions(ac)
        return obs

    def reset(self):
        obs = self.env.reset()

        # flip the cube
        if self.is_level_4:
            try:
                obs = self.cube_manipulator.align_rotation(obs)
            except Exception as e:
                print(EXCEP_MSSG.format(message='cube flipping seemed to fail...', error=str(e)))
                # NOTE: THIS MAY FAIL if the original env rejects calling reset() before "done" Hasn't checked it.
                # NOTE: Also, this is not allowed for evaluation.
                if not self.__evaluation:
                    if 'Monitor' in str(self.env):
                        self.env.stats_recorder.save_complete()
                        self.env.stats_recorder.done = True
                    return self.reset()
                else:
                    # TODO: run bare force control if planning fails.
                    # self._run_backup_fc_sequence(obs)
                    pass

        # approach a grasp pose
        try:
            obs = self._grasp_approach(obs)
        except Exception as e:
            print(EXCEP_MSSG.format(message='planning to grasp the cube seeemed to fail...', error=str(e)))
            # NOTE: THIS MAY FAIL if the original env rejects calling reset() before "done" Hasn't checked it.
            # NOTE: Also, this is not allowed for evaluation.
            if not self.__evaluation:
                if 'Monitor' in str(self.env):
                    self.env.stats_recorder.save_complete()
                    self.env.stats_recorder.done = True
                return self.reset()
            else:
                # TODO: ?
                # self._run_backup_fc_sequence(obs)
                pass

        obs = self._tighten_grasp(obs)  # NOTE: this steps the environment!!
        self.scripted_action = self.pi(obs)
        return self._add_action_to_obs(obs, self.scripted_action)

    def step(self, torq_action):
        action = self.scripted_action + torq_action
        action = np.clip(action, self.action_space.low,
                         self.action_space.high)
        obs, reward, done, info = self.env.step(action)
        self.scripted_action = self.pi(obs)
        return self._add_action_to_obs(obs, self.scripted_action), reward, done, info

    def _tighten_grasp(self, obs, grasp_force=0.8):
        from code.fc_force_control import grasp_force_control
        obs = grasp_force_control(self.env, obs, self.pi, grasp_force=grasp_force)
        return obs

    def _grasp_approach(self, obs):
        obs = self.cube_manipulator.grasp_approach(
            obs,
            margin_coef=1.3,
            n_trials=1)
        return obs


class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cameras = camera.TriFingerCameras(image_size=(360, 270))
        self.metadata = {"render.modes": ["rgb_array"]}
        self._initial_reset = True
        self._accum_reward = 0
        self._reward_at_step = 0

    def reset(self):
        import pybullet as p
        obs = self.env.reset()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])
        self._accum_reward = 0
        self._reward_at_step = 0
        if self._initial_reset:
            self._episode_idx = 0
            self._initial_reset = False
        else:
            self._episode_idx += 1
        return obs

    def step(self, action):
        observation, reward, is_done, info = self.env.step(action)
        self._accum_reward += reward
        self._reward_at_step = reward
        return observation, reward, is_done, info

    def render(self, mode='rgb_array', **kwargs):
        assert mode == 'rgb_array', 'RenderWrapper Only supports rgb_array mode'
        images = self.cameras.cameras[0].get_image(), self.cameras.cameras[1].get_image()
        height = images[0].shape[1]
        two_views = np.concatenate((images[0], images[1]), axis=1)
        two_views = cv2.putText(two_views, 'step_count: {:06d}'.format(self.env.unwrapped.step_count), (10, 40),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 0),
                    thickness=1, lineType=cv2.LINE_AA)

        two_views = cv2.putText(two_views, 'episode: {}'.format(self._episode_idx), (10, 70),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 0),
                    thickness=1, lineType=cv2.LINE_AA)

        two_views = cv2.putText(two_views, 'reward: {:.2f}'.format(self._reward_at_step), (10, height - 130),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)

        two_views = cv2.putText(two_views, 'acc_reward: {:.2f}'.format(self._accum_reward), (10, height - 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)

        return two_views


class ResidualLearningMotionPlanningFCWrapper(gym.Wrapper):
    '''
        Wrapper to perform residual learning on top of motion planning and force control.
    '''
    def __init__(self, env, apply_torques, action_repeat=2, align_goal_ori=True,
                 init_cube_manip='auto', use_rrt=False, use_incremental_rrt=False,
                 evaluation=True, is_level_4=False):
        super().__init__(env)
        from code.fc_force_control import ForceControlPolicy, Viz
        from code.cube_manipulator import CubeManipulator
        from code.const import MU

        spaces = TriFingerPlatform.spaces
        self.observation_space = gym.spaces.Dict(
            {
                "robot_position": spaces.robot_position.gym,
                "robot_velocity": spaces.robot_velocity.gym,
                "robot_tip_positions": gym.spaces.Box(
                    low=np.array([spaces.object_position.low] * 3),
                    high=np.array([spaces.object_position.high] * 3),
                ),
                "object_position": spaces.object_position.gym,
                "object_orientation": spaces.object_orientation.gym,
                "goal_object_position": spaces.object_position.gym,
                "goal_object_orientation": spaces.object_orientation.gym,
                "tip_force": gym.spaces.Box(
                    low=np.zeros(3),
                    high=np.ones(3),
                ),
                "torque_action": spaces.robot_torque.gym,
                "position_action": spaces.robot_position.gym,
            }
        )
        self.observation_names.append("torque_action")
        self.observation_names.append("position_action")

        assert self.env.action_type == ActionType.TORQUE_AND_POSITION
        # self.action_type = ActionType.TORQUE
        # self.action_space = TriFingerPlatform.spaces.robot_torque.gym
        self.viz = Viz() if self.visualization else None
        fc_policy = ForceControlPolicy(env, apply_torques=apply_torques, mu=MU, grasp_force=0.0,
                                       viz=self.viz, use_inv_dynamics=True)
        self.fc_policy = fc_policy
        self.cube_manipulator = CubeManipulator(env)
        self.action_repeat = action_repeat
        self.align_goal_ori = align_goal_ori

        self.init_cube_manip = init_cube_manip
        self.use_rrt = use_rrt
        self.use_incremental_rrt = use_incremental_rrt
        self.is_level_4 = is_level_4
        self._prev_obs = None
        self._timestep = None
        self.__evaluation = evaluation

    def _norm_actions(self, action):
        ts = TriFingerPlatform.spaces.robot_torque.gym
        ps = TriFingerPlatform.spaces.robot_position.gym
        t, p = action['torque'], action['position']
        return {
            'torque': 2 * ((t - ts.low) / (ts.high - ts.low)) - 1,
            'position': 2 * ((p - ts.low) / (ps.high - ps.low)) - 1
        }

    def _add_action_to_obs(self, obs, ac=None):
        ts = TriFingerPlatform.spaces.robot_torque.gym
        ps = TriFingerPlatform.spaces.robot_position.gym
        if ac is None:
            obs['torque_action'] = np.zeros(ts.shape)
            obs['position_action'] = np.zeros(ps.shape)
        else:
            ac = self._norm_actions(ac)
            obs['torque_action'] = ac['torque']
            obs['position_action'] = ac['position']
        return obs

    def reset(self):
        obs = self.env.reset()
        init_cube_manip = self._choose_init_cube_manip(obs)

        # flip the cube
        if init_cube_manip == 'flip_and_grasp':
            try:
                obs = self.cube_manipulator.align_rotation(obs)
            except Exception as e:
                print(EXCEP_MSSG.format(message='cube flipping seemed to fail...', error=str(e)))
                # NOTE: THIS MAY FAIL if the original env rejects calling reset() before "done" Hasn't checked it.
                # NOTE: Also, this is not allowed for evaluation.
                raise e
                if not self.__evaluation:
                    if 'Monitor' in str(self.env):
                        self.env.stats_recorder.save_complete()
                        self.env.stats_recorder.done = True
                    return self.reset()
                else:
                    # TODO: run bare force control if planning fails.
                    # self._run_backup_fc_sequence(obs)
                    pass

        # wholebody motion planning
        try:
            self.env.register_custom_log('init_cube_pos', obs['object_position'])
            self.env.register_custom_log('init_cube_ori', obs['object_orientation'])
            self.env.register_custom_log('goal_pos', obs['goal_object_position'])
            self.env.register_custom_log('goal_ori', obs['goal_object_orientation'])
            # This does planning inside
            self.planning_fc_policy = self._instantiate_planning_fc_policy(obs)
        except Exception as e:
            print(EXCEP_MSSG.format(message='wholebody_planning seeemed to fail...', error=str(e)))
            # NOTE: THIS MAY FAIL if the original env rejects calling reset() before "done" Hasn't checked it.
            # NOTE: Also, this is not allowed for evaluation.
            raise e
            if not self.__evaluation:
                if 'Monitor' in str(self.env):
                    self.env.stats_recorder.save_complete()
                    self.env.stats_recorder.done = True
                return self.reset()
            else:
                # TODO: run bare force control if planning fails.
                # self._run_backup_fc_sequence(obs)
                pass

        # approach a grasp pose
        if init_cube_manip in ['grasp', 'flip_and_grasp']:
            try:
                obs = self._grasp_approach(obs)
            except Exception as e:
                print(EXCEP_MSSG.format(message='planning to grasp the cube seeemed to fail...', error=str(e)))
                # NOTE: THIS MAY FAIL if the original env rejects calling reset() before "done" Hasn't checked it.
                # NOTE: Also, this is not allowed for evaluation.
                raise e
                if not self.__evaluation:
                    if 'Monitor' in str(self.env):
                        self.env.stats_recorder.save_complete()
                        self.env.stats_recorder.done = True
                    return self.reset()
                else:
                    # TODO: ?
                    # self._run_backup_fc_sequence(obs)
                    pass

        if init_cube_manip == 'skip':
            assert not self.__evaluation, 'init_cube_manip == "skip" is not allowed at evaluation!!'
            obs = self.planning_fc_policy._initialize_joint_poses(obs)

        obs = self._tighten_grasp(obs)  # NOTE: this steps the environment!!
        self._timestep = 0
        self._maybe_reset_viz(obs)
        self._base_action = self.planning_fc_policy.get_action(obs, self._timestep)
        self.env.register_custom_log('wholebody_planning.cube', self.planning_fc_policy.path.cube)
        self.env.register_custom_log('wholebody_planning.joint', self.planning_fc_policy.path.joint_conf)
        return self._add_action_to_obs(obs, self._base_action)

    def step(self, res_action):
        torq_action_space = TriFingerPlatform.spaces.robot_torque.gym
        position_action_space = TriFingerPlatform.spaces.robot_position.gym
        torq_action = self._base_action['torque'] + res_action['torque']
        torq_action = np.clip(torq_action, torq_action_space.low,
                              torq_action_space.high)
        position_action = self._base_action['position'] + res_action['position']
        position_action = np.clip(position_action, position_action_space.low,
                                  position_action_space.high)

        action = {'torque': torq_action, 'position': position_action}
        obs, reward, done, info = self.env.step(action)
        # if not self.is_level_4 and self.planning_fc_policy.get_steps_past_sequence(self._timestep) > 6:
        #     with action_type_to(ActionType.TORQUE, self.env):
        #         # print('cube_sequence ended. discard positional action and use torque only')
        #         obs, reward, done, info = self.env.step(action['torque'])
        # else:
        #     obs, reward, done, info = self.env.step(action)
        self._timestep += 1
        self._maybe_update_cube_ori_viz(obs)
        self._base_action = self.planning_fc_policy.get_action(obs, self._timestep)
        return self._add_action_to_obs(obs, self._base_action), reward, done, info

    def _choose_init_cube_manip(self, obs):
        if self.init_cube_manip == 'auto':
            # whatever
            # TEMP:
            # init_cube_manip = 'flip_and_grasp'
            # init_cube_manip = 'grasp'
            init_cube_manip = 'skip'
            return init_cube_manip

        else:
            return self.init_cube_manip

    def _instantiate_planning_fc_policy(self, obs):
        from code.fc_planned_motion import PlanningAndForceControlPolicy
        planning_fc_policy = PlanningAndForceControlPolicy(
            self.env, obs, self.fc_policy,
            align_goal_ori=self.align_goal_ori, use_rrt=self.use_rrt,
            use_incremental_rrt=self.use_incremental_rrt
        )
        return planning_fc_policy


    def _grasp_approach(self, obs):
        from code.utils import repeat, ease_out
        action_seq = self.cube_manipulator.get_grasp_approach_actions(
            obs,
            cube_tip_pos=self.planning_fc_policy.get_cube_tip_pos(),
            cube_pose=self.planning_fc_policy.get_init_cube_pose(),
            margin_coef=1.3,
            n_trials=1)
        self.env.register_custom_log('grasp_motion', action_seq)
        # self.env.register_custom_log('grasp_tip_pos', self.planning_fc_policy.get_cube_tip_pos())  # FIXME: gett_cube_tip_pos returns tip positions on the "normal" aligned cube.

        act_seq = ease_out(action_seq, in_rep=3 * 5, out_rep=8 * 5)
        obs = self.cube_manipulator._run_planned_actions(obs, act_seq, ActionType.POSITION, frameskip=1)
        return obs

    def _tighten_grasp(self, obs, grasp_force=0.8):
        from code.fc_force_control import grasp_force_control
        obs = grasp_force_control(self.env, obs, self.fc_policy, grasp_force=grasp_force)
        return obs

    def _maybe_reset_viz(self, obs):
        if self.viz is not None:
            self.viz.reset(obs)

    def _maybe_update_cube_ori_viz(self, obs):
        if self.viz is not None:
            self.viz.update_cube_orientation(obs)


class PyBulletClearGUIWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])
        return obs
