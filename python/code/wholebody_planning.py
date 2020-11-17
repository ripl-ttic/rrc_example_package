#!/usr/bin/env python3
import pybullet as p
import numpy as np
import time
from code.utils import apply_transform, VisualCubeOrientation
from pybullet_planning import plan_wholebody_motion
from collections import namedtuple
from trifinger_simulation.tasks.move_cube import _ARENA_RADIUS, _min_height, _max_height
from code.grasp_sampling import GraspSampler
from code.const import COLLISION_TOLERANCE, VIRTUAL_CUBOID_HALF_SIZE

dummy_links = [-2, -3, -4, -100, -101, -102]  # link indices <=100 denotes circular joints
custom_limits = {
    -2: (-_ARENA_RADIUS, _ARENA_RADIUS),
    -3: (-_ARENA_RADIUS, _ARENA_RADIUS),
    -4: (_min_height, _max_height),
    -100: (-np.pi, np.pi),
    -101: (-np.pi, np.pi),
    -102: (-np.pi, np.pi)
}

Path = namedtuple('Path', ['cube', 'joint_conf', 'tip_path', 'cube_tip_pos'])


def get_joint_states(robot_id, link_indices):
    joint_states = [joint_state[0] for joint_state in p.getJointStates(robot_id, link_indices)]
    return np.asarray(joint_states)


def disable_tip_collisions(env):
    disabled_collisions = set()
    for tip_link in env.platform.simfinger.pybullet_tip_link_indices:
        disabled_collisions.add(((env.platform.cube.block, -1), (env.platform.simfinger.finger_id, tip_link)))
    return disabled_collisions


class WholeBodyPlanner:
    def __init__(self, env):
        self.env = env

        # disable collision check for tip
        self._disabled_collisions = disable_tip_collisions(self.env)

    def _get_disabled_colilsions(self):
        disabled_collisions = set()
        for tip_link in self.env.platform.simfinger.pybullet_tip_link_indices:
            disabled_collisions.add(((self.env.platform.cube.block, -1), (self.env.platform.simfinger.finger_id, tip_link)))
        return disabled_collisions

    def _get_tip_path(self, cube_tip_positions, cube_path):
        def get_quat(euler):
            return p.getQuaternionFromEuler(euler)
        return [apply_transform(cube_pose[:3], get_quat(cube_pose[3:]), cube_tip_positions) for cube_pose in cube_path]

    def get_tighter_path(self, path, coef=0.9):
        from code.utils import IKUtils, filter_none_elements
        ik_utils = IKUtils(self.env)
        cube_tip_pos = path.cube_tip_pos * coef
        tip_path = self._get_tip_path(cube_tip_pos, path.cube)

        print('wholebody planning path length:', len(tip_path))
        jconf_sequence = ik_utils.sample_iks(tip_path, sort_tips=False)
        inds, joint_conf = filter_none_elements(jconf_sequence)

        # if two or more tip positions are invalid (no ik solution), just use the original grasp
        num_no_iksols = len(jconf_sequence) - len(joint_conf)
        if num_no_iksols > 0:
            print(f'warning: {num_no_iksols} IK solutions are not found in WholebodyPlanning.get_tighter_path')
        if num_no_iksols > 1:
            print(f'warning: num_no_iksols > 1 --> not using tighter grasp path')
            return path

        cube_path = [path.cube[idx] for idx in inds]
        tip_path = [tip_path[idx] for idx in inds]

        return Path(cube_path, joint_conf, tip_path, cube_tip_pos)

    def plan(self, obs, goal_pos=None, goal_quat=None, retry_grasp=10, mu=1.0,
             halfsize=VIRTUAL_CUBOID_HALF_SIZE, use_rrt=False,
             use_incremental_rrt=False, min_goal_threshold=0.01,
             max_goal_threshold=0.8, use_ori=False):
        goal_pos = obs['goal_object_position'] if goal_pos is None else goal_pos
        goal_quat = obs['goal_object_orientation'] if goal_quat is None else goal_quat
        resolutions = 0.03 * np.array([0.3, 0.3, 0.3, 1, 1, 1])  # roughly equiv to the lengths of one step.

        goal_ori = p.getEulerFromQuaternion(goal_quat)
        target_pose = np.concatenate([goal_pos, goal_ori])
        grasp_sampler = GraspSampler(self.env, obs, mu=mu, slacky_collision=True)
        grasps = grasp_sampler.get_heurisic_grasps(halfsize)
        org_joint_conf = obs['robot_position']
        org_joint_vel = obs['robot_velocity']

        # if self.env.visualization:
        #     vis_cubeori = VisualCubeOrientation(obs['object_position'], obs['object_orientation'])
        #     vis_goalori = VisualCubeOrientation(goal_pos, goal_quat)
        # else:
        #     vis_cubeori = None

        print("WHOLEBODY PLANNING")
        print(f"NUM GRASPS: {len(grasps)}")
        counter = -1
        cube_path = None
        if not use_ori:
            print("CHANGING GOAL ORIENTATION...")
            use_ori = True
            goal_ori = p.getEulerFromQuaternion(obs['object_orientation'])
            target_pose = np.concatenate([goal_pos, goal_ori])
        from code.utils import keep_state
        while cube_path is None and counter < retry_grasp:
            retry_count = max(0, counter)
            goal_threshold = ((retry_count / retry_grasp)
                              * (max_goal_threshold - min_goal_threshold)
                              + min_goal_threshold)
            print(counter, goal_threshold)
            for cube_tip_positions, current_tip_positions, joint_conf in grasps:
                with keep_state(self.env):
                    self.env.platform.simfinger.reset_finger_positions_and_velocities(joint_conf)
                    cube_path, joint_conf_path = plan_wholebody_motion(
                        self.env.platform.cube.block,
                        dummy_links,
                        self.env.platform.simfinger.finger_id,
                        self.env.platform.simfinger.pybullet_link_indices,
                        target_pose,
                        current_tip_positions,
                        cube_tip_positions,
                        init_joint_conf=joint_conf,
                        ik=self.env.pinocchio_utils.inverse_kinematics,
                        obstacles=[self.env.platform.simfinger.finger_id],
                        disabled_collisions=self._disabled_collisions,
                        custom_limits=custom_limits,
                        resolutions=resolutions,
                        diagnosis=False,
                        max_distance=-COLLISION_TOLERANCE,
                        # vis_fn=vis_cubeori.set_state,
                        iterations=20,
                        use_rrt=use_rrt,
                        use_incremental_rrt=use_incremental_rrt,
                        use_ori=use_ori,
                        goal_threshold=goal_threshold,
                        restarts=0 if counter < 0 else 1  # only check for direct path the first time
                    )
                if cube_path is not None:
                    break
            counter += 1

        self.env.platform.simfinger.reset_finger_positions_and_velocities(org_joint_conf, org_joint_vel)

        if cube_path is None:
            raise RuntimeError('wholebody planning failed')

        one_pose = (len(np.shape(cube_path)) == 1)
        if one_pose:
            cube_path = [cube_path]
        tip_path = self._get_tip_path(cube_tip_positions, cube_path)
        return Path(cube_path, joint_conf_path, tip_path, cube_tip_positions)


if __name__ == '__main__':
    from trifinger_simulation.tasks import move_cube
    from code.make_env import make_training_env

    reward_fn = 'competition_reward'
    termination_fn = 'position_close_to_goal'
    initializer = 'small_rot_init'
    env = make_training_env(move_cube.sample_goal(-1).to_dict(), 4,
                            reward_fn=reward_fn,
                            termination_fn=termination_fn,
                            initializer=initializer,
                            action_space='position',
                            sim=True, visualization=True)

    env = env.env  # HACK to remove FlatObservationWrapper
    for i in range(1):
        obs = env.reset()

        goal_pos = obs["goal_object_position"]
        goal_quat = obs["goal_object_orientation"]
        planner = WholeBodyPlanner(env)
        path = planner.plan(obs, goal_pos, goal_quat, use_rrt=True,
                            use_ori=True, min_goal_threshold=0.01,
                            max_goal_threshold=0.3)

        if path.cube is None:
            print('PATH is NOT found...')
            quit()

        vis_cubeori = VisualCubeOrientation(obs['object_position'], obs['object_orientation'])

        # clear some windows in GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # change camera parameters # You can also rotate the camera by CTRL + drag
        p.resetDebugVisualizerCamera( cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])
        # p.startStateLogging( p.STATE_LOGGING_VIDEO_MP4, f'wholebody_planning_{args.seed}.mp4')
        for cube_pose, joint_conf in zip(path.cube, path.joint_conf):
            point, ori = cube_pose[:3], cube_pose[3:]
            quat = p.getQuaternionFromEuler(ori)
            for i in range(3):
                p.resetBasePositionAndOrientation(env.platform.cube.block, point, quat)
                env.platform.simfinger.reset_finger_positions_and_velocities(joint_conf)
                vis_cubeori.set_state(point, quat)
                time.sleep(0.01)
