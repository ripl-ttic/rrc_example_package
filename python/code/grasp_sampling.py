#!/usr/bin/env python3
from code.grasping import Transform, CoulombFriction, Cube
import itertools
import numpy as np
from code.const import CUBOID_HALF_SIZE


def sample(ax, sign, half_size=CUBOID_HALF_SIZE, shrink_region=[0.0, 0.6, 0.0]):
    point = np.empty(3)
    for i in range(3):
        if i == ax:
            point[ax] = sign * half_size[ax]
        else:
            point[i] = np.random.uniform(-half_size[i] * shrink_region[i],
                                         half_size[i] * shrink_region[i])
    return point


def sample_side_face(n, half_size, object_ori, shrink_region=[0.0, 0.6, 0.0]):
    R_base_to_cube = Transform(np.zeros(3), object_ori).inverse()
    z_cube = R_base_to_cube(np.array([0, 0, 1]))
    axis = np.argmax(np.abs(z_cube))
    sample_ax = np.array([i for i in range(3) if i != axis])
    points = np.stack([
        sample(np.random.choice(sample_ax), np.random.choice([-1, 1]),
               half_size, shrink_region)
        for _ in range(n)
    ])
    return points


def get_side_face_centers(half_size, object_ori):
    R_base_to_cube = Transform(np.zeros(3), object_ori).inverse()
    z_cube = R_base_to_cube(np.array([0, 0, 1]))
    axis = np.argmax(np.abs(z_cube))
    points = []
    for ax in range(3):
        if ax != axis:
            points.append(sample(ax, 1, half_size, np.zeros(3)))
            points.append(sample(ax, -1, half_size, np.zeros(3)))
    return np.array(points)


def get_three_sided_heuristic_grasps(half_size, object_ori):
    points = get_side_face_centers(half_size, object_ori)
    grasps = []
    for ind in range(4):
        grasps.append(points[np.array([x for x in range(4) if x != ind])])
    return grasps


def get_two_sided_heurictic_grasps(half_size, object_ori):
    side_centers = get_side_face_centers(half_size, object_ori)
    ax1 = side_centers[1] - side_centers[0]
    ax2 = side_centers[3] - side_centers[2]
    g1 = np.array([
        side_centers[0],
        side_centers[1] + 0.15 * ax2,
        side_centers[1] - 0.15 * ax2,
    ])
    g2 = np.array([
        side_centers[1],
        side_centers[0] + 0.15 * ax2,
        side_centers[0] - 0.15 * ax2,
    ])
    g3 = np.array([
        side_centers[2],
        side_centers[3] + 0.15 * ax1,
        side_centers[3] - 0.15 * ax1,
    ])
    g4 = np.array([
        side_centers[3],
        side_centers[2] + 0.15 * ax1,
        side_centers[2] - 0.15 * ax1,
    ])
    return [g1, g2, g3, g4]


def get_all_heurisic_grasps(half_size, object_ori):
    return (
        get_three_sided_heuristic_grasps(half_size, object_ori)
        + get_two_sided_heurictic_grasps(half_size, object_ori)
    )


class GraspSampler(object):
    def __init__(self, env, obs, mu=1.0, slacky_collision=False):
        from code.const import INIT_JOINT_CONF
        from code.utils import IKUtils
        self.cube = Cube(CoulombFriction(mu=mu))
        self.object_ori = obs['object_orientation']
        self.ik = env.pinocchio_utils.inverse_kinematics
        self.id = env.platform.simfinger.finger_id
        self.tip_ids = env.platform.simfinger.pybullet_tip_link_indices
        self.link_ids = env.platform.simfinger.pybullet_link_indices
        self.T_cube_to_base = Transform(obs['object_position'],
                                        obs['object_orientation'])
        self.q_init = obs['robot_position']
        self.v_init = obs['robot_velocity']
        self.tips_init = obs['robot_tip_positions']
        self.env = env
        self.ik_utils = IKUtils(env)
        self.slacky_collision = slacky_collision
        self._org_tips_init = np.array(self.env.platform.forward_kinematics(INIT_JOINT_CONF))

    def _reject(self, points):
        contacts = [self.cube.contact_from_tip_position(point)
                    for point in points]
        if not self.cube.force_closure_test(contacts):
            return True, None
        points_base = self.T_cube_to_base(points)
        qs = self.ik_utils.sample_no_collision_ik(points_base, sort_tips=False,
                                                  slacky_collision=self.slacky_collision)
        if len(qs) == 0:
            return True, None
        return False, qs[0]

    def _assign_positions_to_fingers(self, tips):

        min_cost = 1000000
        opt_tips = []
        opt_inds = [0, 1, 2]
        for v in itertools.permutations([0, 1, 2]):
            sorted_tips = tips[v, :]
            cost = np.linalg.norm(sorted_tips - self._org_tips_init) #  self.tips_init)
            if min_cost > cost:
                min_cost = cost
                opt_tips = sorted_tips
                opt_inds = v

        return opt_tips, opt_inds

    def __call__(self, size, shrink_region=[0.0, 0.6, 0.0], max_retries=300):
        retry = 0
        while retry < max_retries:
            points = sample_side_face(3, size, self.object_ori,
                                      shrink_region=shrink_region)
            tips = self.T_cube_to_base(points)
            tips, inds = self._assign_positions_to_fingers(tips)
            points = points[inds, :]
            should_reject, q = self._reject(points)
            if not should_reject:
                self.env.platform.simfinger.reset_finger_positions_and_velocities(self.q_init, self.v_init)  # TEMP: this line lacks somewhere in this class..
                return points, tips, q
            retry += 1
        raise RuntimeError('No feasible grasp is found.')

    def get_heurisic_grasps(self, halfsize):
        grasps = get_all_heurisic_grasps(halfsize, self.object_ori)
        valid_grasps = []
        for points in grasps:
            tips = self.T_cube_to_base(points)
            tips, inds = self._assign_positions_to_fingers(tips)
            points = points[inds, :]
            should_reject, q = self._reject(points)
            if not should_reject:
                self.env.platform.simfinger.reset_finger_positions_and_velocities(self.q_init, self.v_init)  # TEMP: this line lacks somewhere in this class..
                valid_grasps.append([points, tips, q])
        return valid_grasps


if __name__ == '__main__':
    import pybullet as p
    from code.make_env import make_training_env
    from trifinger_simulation.tasks import move_cube
    from code.const import VIRTUAL_CUBOID_HALF_SIZE
    reward_fn = 'competition_reward'
    termination_fn = 'position_close_to_goal'
    initializer = 'training_init'

    env = make_training_env(move_cube.sample_goal(-1).to_dict(), 3,
                            reward_fn=reward_fn,
                            termination_fn=termination_fn,
                            initializer=initializer,
                            action_space='torque',
                            sim=True,
                            visualization=True)
    env = env.env

    obs = env.reset()
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                 cameraPitch=-40,
                                 cameraTargetPosition=[0, 0, 0])

    sampler = GraspSampler(env, obs, slacky_collision=True)
    _, tips, q = sampler(size=VIRTUAL_CUBOID_HALF_SIZE,
                         shrink_region=[0.0, 0.6, 0.0])

    while (p.isConnected()):
        env.platform.simfinger.reset_finger_positions_and_velocities(q)
