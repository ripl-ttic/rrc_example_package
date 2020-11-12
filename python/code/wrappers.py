"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import pybullet as p
import numpy as np
import gym
from trifinger_simulation import TriFingerPlatform
from trifinger_simulation import camera
from trifinger_simulation.visual_objects import CubeMarker
from code.env.cube_env import ActionType
import cv2
from code.const import INIT_JOINT_CONF
from scipy.spatial.transform import Rotation as R
from code.domain_randomization import TriFingerRandomizer

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
                "robot_torque": env.observation_space['robot']['torque'],
                "robot_tip_positions": env.observation_space['robot']['tip_positions'],
                "object_position": env.observation_space["achieved_goal"]["position"],
                "object_orientation": env.observation_space["achieved_goal"]["orientation"],
                "goal_object_position": env.observation_space["desired_goal"]["position"],
                "goal_object_orientation": env.observation_space["desired_goal"]["orientation"],
                "tip_force": env.observation_space["robot"]["tip_force"],
                "action_torque": env.observation_space['robot']['torque'],
                "action_position": env.observation_space['robot']['position'],
            }
        )

    def observation(self, obs):
        old_obs = {
            "robot_position": obs['robot']['position'],
            "robot_velocity": obs['robot']['velocity'],
            "robot_torque": obs['robot']['torque'],
            "robot_tip_positions": obs['robot']['tip_positions'],
            "tip_force": obs['robot']['tip_force'],
            "object_position": obs['achieved_goal']['position'],
            "object_orientation": obs['achieved_goal']['orientation'],
            "goal_object_position": obs['desired_goal']['position'],
            "goal_object_orientation": obs['desired_goal']['orientation'],
        }
        if self.action_space == self.observation_space['robot_position']:
            old_obs['action_torque'] = np.zeros_like(obs['action'])
            old_obs['action_position'] = obs['action']
        elif self.action_space == self.observation_space['robot_torque']:
            old_obs['action_torque'] = obs['action']
            old_obs['action_position'] = np.zeros_like(obs['action'])
        else:
            old_obs['action_torque'] = obs['action']['torque']
            old_obs['action_position'] = obs['action']['position']
        return old_obs

class InitStayHoldWrapper(gym.Wrapper):
    '''
    Oftentimes the initial pose of the robot is quite off from the robot_position.default.
    And that causes annoying issues.
    This wrapper forces to apply env.initial_action for the first "hold_steps" steps to properly reset the robot pose.
    '''
    def __init__(self, env, hold_steps=300):
        super().__init__(env)
        self.hold_steps = hold_steps
        self.env = env

    def reset(self):
        from code.utils import action_type_to
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

            with action_type_to(ActionType.POSITION, self.env):
                obs, reward, done, info = self.env.step(desired_position)
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

    def __init__(self, env, apply_torques, evaluation=False, is_level_4=False, skip_motions=False):
        super().__init__(env)
        from code.cube_manipulator import CubeManipulator
        assert self.env.action_type == ActionType.TORQUE
        self.action_space = TriFingerPlatform.spaces.robot_torque.gym
        spaces = TriFingerPlatform.spaces
        ob_space = dict(self.observation_space.spaces)
        ob_space['base_action_torque'] = spaces.robot_torque.gym
        self.observation_space = gym.spaces.Dict(ob_space)
        self.observation_names.append("base_action_torque")
        from code.fc_force_control import ForceControlPolicy, Viz
        from code.const import MU
        self.viz = Viz() if self.visualization else None
        self.pi = ForceControlPolicy(self.env, apply_torques=apply_torques, mu=MU, viz=self.viz)
        self.cube_manipulator = CubeManipulator(env)
        self.is_level_4 = is_level_4
        self.__evaluation = evaluation
        self.action_log = {'residual_torque': [], 'base_torque': [], 'clipped_torque': []}
        self.skip_motions = skip_motions

    def _norm_actions(self, action):
        ts = TriFingerPlatform.spaces.robot_torque.gym
        return 2 * ((action - ts.low) / (ts.high - ts.low)) - 1

    def _add_action_to_obs(self, obs, ac=None):
        ts = TriFingerPlatform.spaces.robot_torque.gym
        if ac is None:
            obs['base_action_torque'] = np.zeros(ts.shape)
        else:
            obs['base_action_torque'] = self._norm_actions(ac)
        return obs

    def reset(self):
        obs = self.env.reset()
        self.env.register_custom_log('init_cube_pos', obs['object_position'])
        self.env.register_custom_log('init_cube_ori', obs['object_orientation'])
        self.env.register_custom_log('goal_pos', obs['goal_object_position'])
        self.env.register_custom_log('goal_ori', obs['goal_object_orientation'])
        print('init_cube_pos', obs['object_position'])
        print('init_cube_ori', obs['object_orientation'])
        print('goal_pos', obs['goal_object_position'])
        print('goal_ori', obs['goal_object_orientation'])
        self.env.save_custom_logs()

        # flip the cube
        if self.is_level_4:
            try:
                skip = bool(self.env.simulation and self.skip_motions)
                obs = self.cube_manipulator.align_rotation(obs, skip=skip)
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
            skip = bool(self.env.simulation and self.skip_motions)
            obs = self._grasp_approach(obs, skip=skip)
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
        self.action_log['residual_torque'].append(torq_action)
        self.action_log['base_torque'].append(self.scripted_action)
        self.action_log['clipped_torque'].append(action)
        obs, reward, done, info = self.env.step(action)

        # save action logs
        if done:
            self.env.register_custom_log('action_log', self.action_log)
            self.env.save_custom_logs()
        self.scripted_action = self.pi(obs)
        return self._add_action_to_obs(obs, self.scripted_action), reward, done, info

    def _tighten_grasp(self, obs, grasp_force=0.8):
        from code.fc_force_control import grasp_force_control
        obs = grasp_force_control(self.env, obs, self.pi, grasp_force=grasp_force)
        return obs

    def _grasp_approach(self, obs, skip=False):
        # obs = self.cube_manipulator.grasp_approach(
        #     obs,
        #     margin_coef=2.0,
        #     n_trials=1)
        obs = self.cube_manipulator.heuristic_grasp_approach(obs, skip=skip)
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
                 init_cube_manip='grasp', use_rrt=False, use_incremental_rrt=False,
                 evaluation=True, is_level_4=False, skip_motions=False,
                 adjust_tip=True):
        super().__init__(env)
        from code.fc_force_control import ForceControlPolicy, Viz
        from code.cube_manipulator import CubeManipulator
        from code.const import MU

        spaces = TriFingerPlatform.spaces
        ob_space = dict(self.observation_space.spaces)
        ob_space['base_action_torque'] = spaces.robot_torque.gym
        ob_space['base_action_position'] = spaces.robot_position.gym
        self.observation_space = gym.spaces.Dict(ob_space)
        self.observation_names.append("base_action_torque")
        self.observation_names.append("base_action_position")

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
        self.action_log = {'residual_torque': [], 'base_torque': [], 'clipped_torque': [], 'base_position': [], 'residual_position': [], 'clipped_position': []}
        self.skip_motions = skip_motions
        self.adjust_tip = adjust_tip

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
            obs['base_action_torque'] = np.zeros(ts.shape)
            obs['base_action_position'] = np.zeros(ps.shape)
        else:
            ac = self._norm_actions(ac)
            obs['base_action_torque'] = ac['torque']
            obs['base_action_position'] = ac['position']
        return obs

    def reset(self):
        print('reset is called')
        obs = self.env.reset()
        obs = self.run_initial_manipulations(obs)
        return obs

    def run_initial_manipulations(self, obs, retry=False):
        print("doing initial manipulation")
        self.env.register_custom_log('init_cube_pos', obs['object_position'])
        self.env.register_custom_log('init_cube_ori', obs['object_orientation'])
        self.env.register_custom_log('goal_pos', obs['goal_object_position'])
        self.env.register_custom_log('goal_ori', obs['goal_object_orientation'])
        print('init_cube_pos', obs['object_position'])
        print('init_cube_ori', obs['object_orientation'])
        print('goal_pos', obs['goal_object_position'])
        print('goal_ori', obs['goal_object_orientation'])
        self.env.save_custom_logs()

        if retry:
            try:
                print("Something failed. Recentering cube and retrying...")
                obs = self.cube_manipulator.move_to_center(obs, force_control=False)
                num_steps = 10 if self.simulation else 100
                obs = self.cube_manipulator.wait_for(obs, num_steps=num_steps)
            except Exception as e:
                print(EXCEP_MSSG.format(message='cube recentering seems to fail...', error=str(e)))
                self.run_initial_manipulations(obs, retry=True)

        # flip the cube
        if self.init_cube_manip == 'flip_and_grasp':
            try:
                skip = bool(self.env.simulation and self.skip_motions)
                obs = self.cube_manipulator.align_rotation(obs, skip=skip)
            except Exception as e:
                print(EXCEP_MSSG.format(message='cube flipping seemed to fail...', error=str(e)))
                self.run_initial_manipulations(obs, retry=True)

        # wholebody motion planning
        try:
            # This does planning inside
            self.planning_fc_policy = self._instantiate_planning_fc_policy(obs)
        except Exception as e:
            print(EXCEP_MSSG.format(message='wholebody_planning seeemed to fail...', error=str(e)))
            self.run_initial_manipulations(obs, retry=True)

        # approach a grasp pose
        if self.init_cube_manip in ['grasp', 'flip_and_grasp']:
            try:
                skip = bool(self.env.simulation and self.skip_motions)
                obs = self._grasp_approach(obs, skip=skip)
            except Exception as e:
                print(EXCEP_MSSG.format(message='planning to grasp the cube seeemed to fail...', error=str(e)))
                self.run_initial_manipulations(obs, retry=True)

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
        self.action_log['base_torque'].append(self._base_action['torque'])
        self.action_log['residual_torque'].append(res_action['torque'])
        self.action_log['clipped_torque'].append(torq_action)
        position_action = self._base_action['position'] + res_action['position']
        position_action = np.clip(position_action, position_action_space.low,
                                  position_action_space.high)
        self.action_log['base_position'].append(self._base_action['position'])
        self.action_log['residual_position'].append(res_action['position'])
        self.action_log['clipped_position'].append(position_action)

        action = {'torque': torq_action, 'position': position_action}
        obs, reward, done, info = self.env.step(action)
        # if not self.is_level_4 and self.planning_fc_policy.get_steps_past_sequence(self._timestep) > 6:
        #     with action_type_to(ActionType.TORQUE, self.env):
        #         # print('cube_sequence ended. discard positional action and use torque only')
        #         obs, reward, done, info = self.env.step(action['torque'])
        # else:
        #     obs, reward, done, info = self.env.step(action)

        # save action logs
        if done:
            self.env.register_custom_log('action_log', self.action_log)
            self.env.save_custom_logs()
        self._timestep += 1
        self._maybe_update_cube_ori_viz(obs)
        self._base_action = self.planning_fc_policy.get_action(obs, self._timestep)
        return self._add_action_to_obs(obs, self._base_action), reward, done, info

    def _instantiate_planning_fc_policy(self, obs):
        from code.fc_planned_motion import PlanningAndForceControlPolicy
        planning_fc_policy = PlanningAndForceControlPolicy(
            self.env, obs, self.fc_policy,
            align_goal_ori=self.align_goal_ori, use_rrt=self.use_rrt,
            use_incremental_rrt=self.use_incremental_rrt,
            adjust_tip=self.adjust_tip
        )
        return planning_fc_policy

    def _grasp_approach(self, obs, skip=False):
        # obs = self.cube_manipulator.grasp_approach(
        #     obs,
        #     cube_tip_pos=self.planning_fc_policy.get_cube_tip_pos(),
        #     cube_pose=self.planning_fc_policy.get_init_cube_pose(),
        #     margin_coef=2.0,
        #     n_trials=1)
        obs = self.cube_manipulator.heuristic_grasp_approach(
            obs,
            cube_tip_positions=self.planning_fc_policy.get_cube_tip_pos(),
            skip=skip
        )
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


class RandomizedEnvWrapper(gym.Wrapper):
    def __init__(self, env, camera_fps=10.6, visualize=False):
        super().__init__(env)
        self.first_run = True
        self.randomizer = TriFingerRandomizer()
        self.steps_per_camera_frame = int((1.0 / camera_fps) / 0.004)
        self.visualize = visualize
        self.marker = None

        self.step_count = 0
        self.param = None
        spaces = env.observation_space.spaces.copy()
        spaces['clean'] = env.observation_space
        spaces['params'] = self.randomizer.get_parameter_space()
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self):
        if self.marker:
            del self.marker
            self.marker = None
        obs = self.env.reset()
        if self.first_run:
            self.finger_id = self.env.platform.simfinger.finger_id
            self.joint_indices = self.env.platform.simfinger.pybullet_joint_indices
            self.link_indices = self.env.platform.simfinger.pybullet_link_indices
            self.client_id = self.env.platform.simfinger._pybullet_client_id
            self.cube_id = self.env.platform.cube.block
            self.first_run = False

        self.randomize_param()
        self.step_count = 0
        self.noisy_cube_pose = self.sample_noisy_cube(obs)
        if self.visualize:
            self.marker = CubeMarker(
                width=0.065,
                position=self.noisy_cube_pose['position'],
                orientation=self.noisy_cube_pose['orientation'],
                physicsClientId=self.platform.simfinger._pybullet_client_id,
            )

        return self.randomize_obs(obs)

    def step(self, action):
        action = self.randomize_action(action)
        obs, reward, is_done, info = self.env.step(action)

        self.step_count += self.unwrapped.frameskip
        if self.step_count >= self.steps_per_camera_frame:
            self.noisy_cube_pose = self.sample_noisy_cube(obs)
            self.step_count -= self.steps_per_camera_frame
        obs = self.randomize_obs(obs)

        if self.visualize:
            self.marker.set_state(position=obs['object_position'],
                                  orientation=obs['object_orientation'])
        return obs, reward, is_done, info

    def randomize_action(self, action):
        noise = self.randomizer.sample_action_noise()
        action_type = self.env.action_type
        if action_type == ActionType.POSITION:
            return np.clip(action + noise['action_position'],
                           self.action_space.low,
                           self.action_space.high)

        elif action_type == ActionType.TORQUE:
            return np.clip(action + noise['action_torque'],
                           self.action_space.low,
                           self.action_space.high)

        elif action_type == ActionType.TORQUE_AND_POSITION:
            pos_action = np.clip(action['position'] + noise['action_position'],
                                 self.action_space['position'].low,
                                 self.action_space['position'].high)
            torq_action = np.clip(action['torque'] + noise['action_torque'],
                                  self.action_space['torque'].low,
                                  self.action_space['torque'].high)

            return {'torque': torq_action, 'position': pos_action}
        else:
            raise ValueError("Can't add noise to actions. Unknown action type.")

    def randomize_obs(self, obs):
        from copy import deepcopy

        clean_obs = deepcopy(obs)

        noise = self.randomizer.sample_robot_noise()

        # add noise to robot_position
        ob_space = self.env.observation_space['robot_position']
        obs['robot_position'] = np.clip(obs['robot_position'] + noise['robot_position'],
                                        ob_space.low, ob_space.high)
        obs['robot_tip_positions'] = np.array(self.unwrapped.platform.forward_kinematics(obs['robot_position']))
        # add noise to robot_velocity
        ob_space = self.env.observation_space['robot_velocity']
        obs['robot_velocity'] = np.clip(obs['robot_velocity'] + noise['robot_velocity'],
                                        ob_space.low, ob_space.high)
        # add noise to robot_torque
        ob_space = self.env.observation_space['robot_torque']
        obs['robot_torque'] = np.clip(obs['robot_torque'] + noise['robot_torque'],
                                      ob_space.low, ob_space.high)
        # add noise to tip_force
        ob_space = self.env.observation_space['tip_force']
        obs['tip_force'] = np.clip(obs['tip_force'] + noise['tip_force'],
                                   ob_space.low, ob_space.high)

        # use saved noisy object observation
        obs['object_position'] = self.noisy_cube_pose['position']
        obs['object_orientation'] = self.noisy_cube_pose['orientation']

        obs['clean'] = clean_obs
        obs['params'] = np.concatenate(
            [v.flatten() for v in self.params.values()]
        )
        return obs

    def sample_noisy_cube(self, obs):
        noise = self.randomizer.sample_cube_noise()
        q_obj = R.from_quat(obs['object_orientation'])
        q_noise = R.from_euler('ZYX', noise['cube_ori'], degrees=False)
        return {
            'position': obs['object_position'] + noise['cube_pos'],
            'orientation': (q_obj * q_noise).as_quat()
        }

    def randomize_param(self):
        self.params = self.randomizer.sample_dynamics()
        p.changeDynamics(bodyUniqueId=self.cube_id, linkIndex=-1,
                         physicsClientId=self.client_id,
                         mass=self.params['cube_mass'])

        robot_params = {k: v for k, v in self.params.items() if 'cube' not in k}
        self.set_robot_params(**robot_params)

    def set_default(self):
        cube_params = {"mass": self.cube_default_mass}
        self.set_cube_params(**cube_params)
        self.set_params(**self.default_params)

        self.current_param = self.default_params
        self.current_param["cube_mass"] = self.cube_default_mass

    def set_robot_params(self, **kwargs):
        # set params by passing kw dictionary
        # all values of dict should be list which length is 3 or 9 for different params or float/int for the same param

        self.check_robot_param_dict(kwargs)
        for i, link_id in enumerate(self.link_indices):
            joint_kwargs = self.get_robot_param_dict(kwargs, i)
            p.changeDynamics(bodyUniqueId=self.finger_id, linkIndex=link_id,
                             physicsClientId=self.client_id, **joint_kwargs)

    def check_robot_param_dict(self, dic):
        for v in dic.values():
            if len(v.shape) > 0:
                assert len(v) in [3, 9]

    def get_robot_param_dict(self, dic, i):
        ret_dic = {}
        for k in dic.keys():
            if len(dic[k].shape) == 0:
                ret_dic[k] = dic[k]
            elif len(dic[k]) == 3:
                ret_dic[k] = dic[k][i % 3]
            elif len(dic[k]) == 9:
                ret_dic[k] = dic[k][i]
            else:
                raise ValueError("Weird param shape.")

        return ret_dic


class AlignedInitCubeWrapper(gym.ObservationWrapper):
    '''
    The scripted cube-flipping assumes that the z-face of the initial cube is always facing up.
    It's not always the case in the real-robot environment.
    To fix the issue, this wrapper converts the orientation
    '''
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.rot = None
        self.visuals = []

    def reset(self, **kwargs):
        from code.utils import VisualCubeOrientation, VisualMarkers
        obs = self.env.reset()
        init_cube_ori = obs['object_orientation']
        # self.org_vis = VisualCubeOrientation(obs['object_position'], init_cube_ori)
        # self.vis = VisualCubeOrientation(obs['object_position'] + 0.05, init_cube_ori)
        # self.visuals.append(
        #     VisualCubeOrientation(obs['object_position'], init_cube_ori)
        # )
        base_z = np.array([0, 0, 1])
        rot_cube_to_base = R.from_quat(init_cube_ori)

        # tmp = 0
        # marker = VisualMarkers()
        rotations = [R.from_euler('x', i * 90, degrees=True) for i in range(4)]
        rotations += [R.from_euler('y', i * 90, degrees=True) for i in range(4)]
        for rotation in rotations:
            rot_base_z = (rot_cube_to_base * rotation).apply(base_z)
            # marker.add(rot_base_z * 0.05, color=(1, 0, 0, 0.5))

            # tmp += 0.03
            # self.visuals.append(
            #     VisualCubeOrientation(obs['object_position'] + tmp, (R.from_quat(init_cube_ori) * rotation).as_quat())
            # )

            # print('============== rot_base_z =================')
            # print(rot_base_z)
            if rot_base_z[2] > 0.7:
                self.rot = rotation
                break

        if self.rot is None:
            raise RuntimeError('something is wrong with the initial cube orientation')

        # self.visuals.append(
        #     VisualCubeOrientation(obs['object_position'] + 0.1, self._rotate(init_cube_ori))
        # )
        self.goal_ori = self._rotate(obs['goal_object_orientation'])
        cube_ori = self._rotate(obs['object_orientation'])

        # overwrite the values on cube_env
        # self.env.unwrapped.goal['orientation'] = self.goal_ori
        # self.env.unwrapped.platform.cube.set_state(obs['object_position'], cube_ori)
        obs['goal_object_orientation'] = self.goal_ori
        obs['object_orientation'] = cube_ori
        self.unwrapped.platform.cube.set_state(
            obs['object_position'],
            obs['object_orientation']
        )

        return obs

    def _rotate(self, cube_quat):
        return (R.from_quat(cube_quat) * self.rot).as_quat()

    def observation(self, obs):
        # self.org_vis.set_state(obs['object_position'], obs['object_orientation'])
        obs['object_orientation'] = self._rotate(obs['object_orientation'])
        obs['goal_object_orientation'] = self.goal_ori
        # self.vis.set_state(obs['object_position'] + 0.05, obs['object_orientation'])
        self.unwrapped.platform.cube.set_state(
            obs['object_position'],
            obs['object_orientation']
        )
        return obs
