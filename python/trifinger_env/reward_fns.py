"""Place reward functions here.

These will be passed as an arguement to the training env, allowing us to
easily try out new reward functions.
"""


import numpy as np
from trifinger_simulation.tasks import move_cube
from scipy.spatial.transform import Rotation


###############################
# Competition Reward Functions
###############################

def competition_reward(previous_observation, observation, info):
    return -move_cube.evaluate_state(
        move_cube.Pose.from_dict(observation['desired_goal']),
        move_cube.Pose.from_dict(observation['achieved_goal']),
        info["difficulty"],
    )


##############################
# Training Reward functions
##############################


def _tip_distance_to_cube(observation):
    # calculate first reward term
    pose = observation['achieved_goal']
    return np.linalg.norm(
        observation["robot_tip_positions"] - pose['position']
    )


def _action_reg(observation, action):
    v = observation['robot_velocity']
    t = np.array(action.torque)
    velocity_reg = v.dot(v)
    torque_reg = t.dot(t)
    return 0.1 * velocity_reg + torque_reg


def _tip_slippage(previous_observation, observation, action):
    pose = observation['achieved_goal']
    prev_pose = previous_observation['achieved_goal']
    obj_rot = Rotation.from_quat(pose['orientation'])
    prev_obj_rot = Rotation.from_quat(prev_pose['orientation'])
    relative_tip_pos = obj_rot.apply(observation["robot_tip_positions"] - observation["object_position"])
    prev_relative_tip_pos = prev_obj_rot.apply(previous_observation["robot_tip_positions"] - previous_observation["object_position"])
    return - np.linalg.norm(relative_tip_pos - prev_relative_tip_pos)


def training_reward(previous_observation, observation, info):
    shaping = (_tip_distance_to_cube(previous_observation)
               - _tip_distance_to_cube(observation))
    r = competition_reward(previous_observation, observation, info)
    reg = _action_reg(observation, observation['action'])
    slippage = _tip_slippage(previous_observation, observation,
                             observation['action'])
    return r - 0.1 * reg + 500 * shaping + 300 * slippage


def _orientation_error(observation):
    goal_rot = Rotation.from_quat(observation['goal_object_orientation'])
    actual_rot = Rotation.from_quat(observation['object_orientation'])
    error_rot = goal_rot.inv() * actual_rot
    return error_rot.magnitude() / np.pi


def match_orientation_reward(previous_observation, observation, info):
    shaping = (_tip_distance_to_cube(previous_observation)
               - _tip_distance_to_cube(observation))
    return -_orientation_error(observation) + 500 * shaping


def match_orientation_reward_shaped(previous_observation, observation, info):
    shaping = (_tip_distance_to_cube(previous_observation)
               - _tip_distance_to_cube(observation))
    ori_shaping = (_orientation_error(previous_observation)
                   - _orientation_error(observation))
    return 500 * shaping + 100 * ori_shaping
