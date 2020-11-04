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
import itertools
from code.grasping import Transform
from scipy.spatial.transform import Rotation as R

EXCEP_MSSG = "================= captured exception =================\n" + \
    "{message}\n" + "{error}\n" + '=================================='


class NewToOldObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_names = [
            "robot_position",
            "robot_velocity",
            "robot_torque",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
            "goal_object_orientation",
            "tip_force",
            "action_torque",
            "action_position"
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
        self.hold_steps = env.episode_length
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

    def __init__(self, env, apply_torques, evaluation=False, is_level_4=False):
        super().__init__(env)
        from code.cube_manipulator import CubeManipulator
        assert self.env.action_type == ActionType.TORQUE
        self.action_space = TriFingerPlatform.spaces.robot_torque.gym
        spaces = TriFingerPlatform.spaces
        ob_space = dict(self.observation_space.spaces)
        ob_space['base_action_torque'] = spaces.robot_torque.gym
        self.observation_space = gym.spaces.Dict(ob_space)
        self.observation_names.append("base_action_torque")
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
        self.env.save_custom_logs()

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
        # obs = self.cube_manipulator.grasp_approach(
        #     obs,
        #     margin_coef=2.0,
        #     n_trials=1)
        obs = self.cube_manipulator.heuristic_grasp_approach(obs)
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
        self.env.register_custom_log('init_cube_pos', obs['object_position'])
        self.env.register_custom_log('init_cube_ori', obs['object_orientation'])
        self.env.register_custom_log('goal_pos', obs['goal_object_position'])
        self.env.register_custom_log('goal_ori', obs['goal_object_orientation'])
        print('init_cube_pos', obs['object_position'])
        print('init_cube_ori', obs['object_orientation'])
        self.env.save_custom_logs()
        init_cube_manip = self._choose_init_cube_manip(obs)

        # flip the cube
        if init_cube_manip == 'flip_and_grasp':
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
        # obs = self.cube_manipulator.grasp_approach(
        #     obs,
        #     cube_tip_pos=self.planning_fc_policy.get_cube_tip_pos(),
        #     cube_pose=self.planning_fc_policy.get_init_cube_pose(),
        #     margin_coef=2.0,
        #     n_trials=1)
        obs = self.cube_manipulator.heuristic_grasp_approach(
            obs,
            cube_tip_positions=self.planning_fc_policy.get_cube_tip_pos()
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
    def __init__(self, env,
                 cube_rot_var=0.05, cube_pos_var=0.001, cube_randomize_step=3, cube_weight_scale=3,
                 action_noise_scale=0.05,
                 robot_position_noise_scale=0.01,
                 visualize=False):
        super().__init__(env)
        self.first_run = True

        #copied from __set_pybullet_params in simfinger.py
        self.default_params = {
            'mass': [0.26, 0.25, 0.021], #setting mass and maxJointVelocity at the same makes simfinger falling down????
            #'maxJointVelocity':10,
            'restitution':0.8,
            'jointDamping':0.0,
            'lateralFriction':0.1,
            'spinningFriction':0.1,
            'rollingFriction':0.1,
            'linearDamping':0.5,
            'angularDamping':0.5,
            'contactStiffness':0.1,
            'contactDamping':0.05
        }

        self.visualize = visualize
        self.cube_randomize_step = cube_randomize_step
        self.cube_rot_var = cube_rot_var
        self.cube_pos_var = cube_pos_var
        self.cube_weight_scale=cube_weight_scale
        self.action_noise_scale=action_noise_scale
        self.robot_position_noise_scale=robot_position_noise_scale

        self.step_count = 0
        self.cube_pos_noise = None
        self.cube_rot_noise = None
        self.param = None

    def reset(self):
        obs = self.env.reset()
        if self.first_run:

            self.finger_id = self.env.platform.simfinger.finger_id
            self.joint_indices = self.env.platform.simfinger.pybullet_joint_indices
            self.link_indices = self.env.platform.simfinger.pybullet_link_indices
            self.client_id = self.env.platform.simfinger._pybullet_client_id

            self.cube_id = self.env.platform.cube.block
            self.cube_default_mass = p.getDynamicsInfo(bodyUniqueId=self.cube_id, linkIndex=-1, physicsClientId=self.client_id)[0]

            #TODO:
            #figure out why the trifinger falling down to ground when mass and maxJointVelocity are set at the same time
            #add adoptive randomizer that randomize env depending on policy performance

            #cannot get all information by getDynamicsInfo. this returns parts of parameter values set
            # for ind in self.link_indices[:3]:
            #     print(p.getDynamicsInfo(bodyUniqueId=self.finger_id, linkIndex=ind, physicsClientId=self.client_id))

            self.first_run=False

        self.randomize_param()
        #self.set_default()
        #self.set_params(**{'mass':100})

        self.step_count = 0
        self.sampleCubePosNoise()
        self.sampleCubeRotNoise()

        return obs

    def step(self, action):
        action = self.randomize_action(action)
        observation, reward, is_done, info = self.env.step(action)
        observation = self.randomize_obs(observation)

        self.step_count += 1
        if self.step_count > self.cube_randomize_step:
            self.sampleCubePosNoise()
            self.sampleCubeRotNoise()
            self.step_count = 0

        if self.visualize:
            self.marker.set_state(position=observation['object_position'], orientation=observation['object_orientation'])
        return observation, reward, is_done, info

    def randomize_action(self, action):
        action_type = self.env.action_type
        position_action_space = TriFingerPlatform.spaces.robot_position.gym
        torq_action_space = TriFingerPlatform.spaces.robot_torque.gym

        if action_type == ActionType.POSITION:
            noise = position_action_space.sample() * self.action_noise_scale
            action = np.clip(action+noise, position_action_space.low, position_action_space.high)
        elif action_type == ActionType.TORQUE:
            noise = torq_action_space.sample() * self.action_noise_scale
            action = np.clip(action+noise, torq_action_space.low, torq_action_space.high)
        elif action_type == ActionType.TORQUE_AND_POSITION:
            pos_action =  action['position']
            torq_action = action['torque']

            pos_noise  = position_action_space.sample() * self.action_noise_scale
            torq_noise = torq_action_space.sample() * self.action_noise_scale

            pos_action  = np.clip(pos_action+pos_noise, position_action_space.low, position_action_space.high)
            torq_action = np.clip(torq_action+torq_noise, torq_action_space.low, torq_action_space.high)

            action = {'torque':torq_action, 'position':pos_action}
        return action

    def randomize_obs(self, obs):
        from copy import deepcopy

        clean_obs = deepcopy(obs)

        quat_noised = self.addNoiseCubeRot(obs['object_orientation'])
        obs['object_orientation'] = quat_noised

        pos_noised = self.addNoiseCubePos(obs['object_position'])
        obs['object_position'] = pos_noised

        robot_pos_noised = self.addNoiseRobotPos(obs['robot_position'])
        obs['robot_position'] = robot_pos_noised

        obs['noisy_obs'] = obs
        obs['clean_obs'] = clean_obs
        obs['params'] = self.current_param

        return obs

    def addNoiseCubePos(self, pos):
        return pos + self.cube_pos_noise

    def addNoiseCubeRot(self, quat):
        from scipy.spatial.transform import Rotation as R
        rot = R.from_quat(quat) * self.cube_rot_noise
        return rot.as_quat()

    def addNoiseRobotPos(self, pos):
        noise = self.env.observation_space['robot_position'].sample() * self.robot_position_noise_scale
        high = self.env.observation_space['robot_position'].high
        low  = self.env.observation_space['robot_position'].low
        pos = noise + pos
        return np.clip(pos, low, high)

    def sampleCubePosNoise(self):
        self.cube_pos_noise = np.random.normal(0, scale=self.cube_pos_var, size=3)

    def sampleCubeRotNoise(self):
        self.cube_rot_noise = self.randGaussRotation(self.cube_rot_var)

    def randGaussRotation(self, var, degrees=False):
        from scipy.spatial.transform import Rotation as R
        order = 'ZYX'
        euler = np.random.normal(0, scale=var, size=3)
        return R.from_euler(order, euler, degrees=degrees)

    def randomize_param(self):
        cube_mass = np.random.uniform(low=1, high=self.cube_weight_scale) * self.cube_default_mass
        cube_params = {"mass": cube_mass}
        self.set_cube_params(**cube_params)

        params = {}
        dic = self.default_params
        for k in dic.keys():
            if type(dic[k]) in [float, int]:
                params[k] = dic[k] * np.random.uniform(low=0.9, high=1.1)
            elif type(dic[k]) in [list, np.ndarray]:
                params[k] = np.array(dic[k]) * np.random.uniform(low=0.9, high=1.1, size=len(dic[k]))
            else:
                raise(ValueError)
        self.set_robot_params(**params)

        self.current_param = params
        self.current_param["cube_mass"] = cube_mass

    def set_default(self):
        cube_params = {"mass":self.cube_default_mass}
        self.set_cube_params(**cube_params)
        self.set_params(**self.default_params)

        self.current_param = self.default_params
        self.current_param["cube_mass"] = self.cube_default_mass

    def set_cube_params(self, **kwargs):
        p.changeDynamics(bodyUniqueId=self.cube_id, linkIndex=-1, physicsClientId=self.client_id,
                         **kwargs)

    def set_robot_params(self, **kwargs):
        # set params by passing kw dictionary
        # all values of dict should be list which length is 3 or 9 for different params or float/int for the same param

        self.check_robot_param_dict(kwargs)
        for i, link_id in enumerate(self.link_indices):
            joint_kwargs = self.get_robot_param_dict(kwargs, i)
            #print(link_id, joint_kwargs)
            p.changeDynamics(bodyUniqueId=self.finger_id, linkIndex=link_id, physicsClientId=self.client_id,
                             **joint_kwargs)


    def check_robot_param_dict(self, dic):
        for v in dic.values():
            assert (type(v) in [list, np.ndarray] and len(v) in [3, 9]) or type(v) in [float, int]


    def get_robot_param_dict(self, dic, i):
        ret_dic = {}
        for k in dic.keys():
            if type(dic[k]) in [float, int]:
                ret_dic[k] = dic[k]
            elif type(dic[k]) in [list, np.ndarray] and len(dic[k]) == 3:
                ret_dic[k] = dic[k][i%3]
            elif type(dic[k]) in [list, np.ndarray] and len(dic[k]) == 9:
                ret_dic[k] = dic[k][i]
            else:
                raise(ValueError)

        return ret_dic


class AlignedInitCubeWrapper(gym.Wrapper):
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
        self.env.unwrapped.goal['orientation'] = self.goal_ori
        self.env.unwrapped.platform.cube.set_state(obs['object_position'], cube_ori)
        obs['goal_object_orientation'] = self.goal_ori
        obs['object_orientation'] = cube_ori

        return obs

    def _rotate(self, cube_quat):
        return (R.from_quat(cube_quat) * self.rot).as_quat()

    # def observation(self, obs):
    #     # self.org_vis.set_state(obs['object_position'], obs['object_orientation'])
    #     obs['object_orientation'] = self._rotate(obs['object_orientation'])
    #     obs['goal_object_orientation'] = self.goal_ori
    #     # self.vis.set_state(obs['object_position'] + 0.05, obs['object_orientation'])
    #     if 'clean_obs' in obs:
    #         cube_pos = obs['clean_obs']['object_position']
    #         cube_ori = obs['clean_obs']['object_orientation']
    #     else:
    #         cube_pos = obs['object_position']
    #         cube_ori = obs['object_orientation']

    #     # self.unwrapped.platform.cube.set_state(cube_pos, cube_ori)
    #     return obs
