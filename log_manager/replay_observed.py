#!/usr/bin/env python3

import os
import shelve
import time
import argparse
import robot_interfaces
import robot_fingers
import trifinger_simulation
import pybullet as p
import numpy as np
from trifinger_simulation.tasks import move_cube
from trifinger_simulation import camera
import trifinger_object_tracking.py_tricamera_types as tricamera
from trifinger_cameras.utils import convert_image
from code.utils import VisualMarkers
from code.const import TRANSLU_BLUE
import cv2

def load_data(path):
    data = {}
    with shelve.open(path) as f:
        for key, val in f.items():
            data[key] = val
    print(data)
    return data

class VideoRecorder:
    def __init__(self, image_size=(360, 270)):
        self.image_size = image_size
        self.frame_size = None
        self.cameras = camera.TriFingerCameras(image_size=image_size)
        self.frames = []

    def get_views(self):
        images = [self.cameras.cameras[i].get_image() for i in range(3)]
        three_views = np.concatenate((*images,), axis=1)
        return three_views

    def capture_frame(self):
        three_views = self.get_views()
        self.add_frame(three_views)
        return three_views

    def add_frame(self, frame):
        if self.frame_size is None:
            self.frame_size = frame.shape[:2]
        assert frame.shape[:2] == self.frame_size
        self.frames.append(frame)

    def save_video(self, filepath):
        out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'DIVX'),
                              15, (self.frame_size[1], self.frame_size[0]))
        for frame in self.frames:
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        out.release()

sleep_duration = 0.1
def main(logdir, video_path):
    robot_logfile = os.path.join(logdir, 'robot_data.dat')
    cam_logfile = os.path.join(logdir, 'camera_data.dat')
    custom_log = load_data(os.path.join(logdir, 'user', 'custom_data'))
    video_recorder = VideoRecorder()

    initial_object_pose = move_cube.Pose(custom_log['init_cube_pos'], custom_log['init_cube_ori'])
    platform = trifinger_simulation.TriFingerPlatform(
        visualization=True,
        initial_object_pose=initial_object_pose,
    )

    init_cube_marker = trifinger_simulation.visual_objects.CubeMarker(
        width=0.065,
        position=custom_log["init_cube_pos"],
        orientation=custom_log["init_cube_ori"],
        color=(1, 0, 0, 0.4),
        physicsClientId=platform.simfinger._pybullet_client_id,
    )

    goal_marker = trifinger_simulation.visual_objects.CubeMarker(
        width=0.065,
        position=custom_log["goal_pos"],
        orientation=custom_log["goal_ori"],
        physicsClientId=platform.simfinger._pybullet_client_id,
    )

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])

    log = robot_fingers.TriFingerPlatformLog(robot_logfile, cam_logfile)
    print('first index', log.get_first_timeindex())
    print('last index', log.get_last_timeindex())
    first_time_index = log.get_first_timeindex()
    # last_time_index = log.get_last_timeindex()
    last_time_index = first_time_index + 30000  # TEMP
    for t in range(first_time_index, last_time_index):
        # reduce frames per second
        if t % 10 != 0:
            continue
        # TriFingerPlatformLog provides the same getters as
        # TriFingerPlatformFrontend:
        print('frame: {} / {}'.format(t, last_time_index - first_time_index))
        robot_observation = log.get_robot_observation(t)
        desired_action = log.get_desired_action(t)
        camera_observation = log.get_camera_observation(t)

        platform.simfinger.reset_finger_positions_and_velocities(desired_action.position)
        platform.cube.set_state(camera_observation.object_pose.position, camera_observation.object_pose.orientation)
        frame_desired = video_recorder.get_views()
        platform.simfinger.reset_finger_positions_and_velocities(robot_observation.position)
        frame_observed = video_recorder.get_views()
        frame_real = np.concatenate([convert_image(camera_observation.cameras[i].image) for i in range(3)], axis=1)
        frame_real = cv2.resize(frame_real, (frame_observed.shape[1], frame_observed.shape[0]))

        frame = np.concatenate((frame_desired, frame_observed, frame_real), axis=0)
        video_recorder.add_frame(frame)
    video_recorder.save_video(video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="path to the log directory")
    parser.add_argument("video_path", help="video file to save (.avi file)")
    args = parser.parse_args()
    main(args.logdir, args.video_path)
