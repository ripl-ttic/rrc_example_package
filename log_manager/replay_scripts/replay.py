#!/usr/bin/env python3

import os
import shelve
import argparse
import robot_fingers
import trifinger_simulation
import pybullet as p
import numpy as np
from trifinger_simulation.tasks import move_cube
from trifinger_simulation import camera, visual_objects
import trifinger_object_tracking.py_tricamera_types as tricamera
import trifinger_cameras
from trifinger_cameras.utils import convert_image
import cv2


def load_data(path):
    data = {}
    with shelve.open(path) as f:
        for key, val in f.items():
            data[key] = val
    return data


class CubeDrawer:
    def __init__(self, logdir):
        calib_files = []
        for name in ("camera60", "camera180", "camera300"):
            calib_files.append(os.path.join(logdir, name + ".yml"))
        self.cube_visualizer = tricamera.CubeVisualizer(calib_files)

    def add_cube(self, images, object_pose):
        cvmats = [trifinger_cameras.camera.cvMat(img) for img in images]
        images = self.cube_visualizer.draw_cube(cvmats, object_pose, False)
        images = [np.array(img) for img in images]

        images = [cv2.putText(
            image,
            "confidence: %.2f" % object_pose.confidence,
            (0, image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0)
        ) for image in images]
        return images


class VideoRecorder:
    def __init__(self, fps, image_size=(270, 270)):
        self.fps = fps
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
        out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'XVID'),
                              self.fps, (self.frame_size[1], self.frame_size[0]))
        for frame in self.frames:
            out.write(frame)
        out.release()


def get_synced_log_data(logdir):
    log = robot_fingers.TriFingerPlatformLog(os.path.join(logdir, "robot_data.dat"),
                                             os.path.join(logdir, "camera_data.dat"))
    log_camera = tricamera.LogReader(os.path.join(logdir, "camera_data.dat"))
    stamps = log_camera.timestamps

    obs = {'robot': [], 'cube': [], 'images': [], 't': [], 'desired_action': [],
           'stamp': []}
    ind = 0
    for t in range(log.get_first_timeindex(), log.get_last_timeindex() + 1):
        if 1000 * log.get_timestamp_ms(t) >= stamps[ind]:
            robot_observation = log.get_robot_observation(t)
            camera_observation = log.get_camera_observation(t)
            obs['robot'].append(robot_observation)
            obs['cube'].append(camera_observation.filtered_object_pose)
            obs['images'].append([convert_image(camera.image)
                                  for camera in camera_observation.cameras])
            obs['desired_action'].append(log.get_desired_action(t))
            obs['t'].append(t)
            obs['stamp'].append(log.get_timestamp_ms(t))
            ind += 1
    return obs


def main(logdir, video_path):
    custom_log = load_data(os.path.join(logdir, 'user/custom_data'))
    data = get_synced_log_data(logdir)
    fps = len(data['t']) / (data['stamp'][-1] - data['stamp'][0])
    video_recorder = VideoRecorder(fps)
    cube_drawer = CubeDrawer(logdir)

    initial_object_pose = move_cube.Pose(custom_log['init_cube_pos'],
                                         custom_log['init_cube_ori'])
    platform = trifinger_simulation.TriFingerPlatform(
        visualization=True,
        initial_object_pose=initial_object_pose,
    )

    visual_objects.CubeMarker(
        width=0.065,
        position=custom_log["init_cube_pos"],
        orientation=custom_log["init_cube_ori"],
        color=(1, 0, 0, 0.4),
        physicsClientId=platform.simfinger._pybullet_client_id,
    )

    visual_objects.CubeMarker(
        width=0.065,
        position=custom_log["goal_pos"],
        orientation=custom_log["goal_ori"],
        physicsClientId=platform.simfinger._pybullet_client_id,
    )

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])

    for i, t in enumerate(data['t']):
        platform.simfinger.reset_finger_positions_and_velocities(data['desired_action'][i].position)
        platform.cube.set_state(data['cube'][i].position, data['cube'][i].orientation)
        frame_desired = video_recorder.get_views()
        frame_desired = cv2.cvtColor(frame_desired, cv2.COLOR_RGB2BGR)
        platform.simfinger.reset_finger_positions_and_velocities(data['robot'][i].position)
        frame_observed = video_recorder.get_views()
        frame_observed = cv2.cvtColor(frame_observed, cv2.COLOR_RGB2BGR)
        frame_real = np.concatenate(data['images'][i], axis=1)
        frame_real_cube = np.concatenate(cube_drawer.add_cube(data['images'][i],
                                                              data['cube'][i]),
                                         axis=1)

        frame = np.concatenate((frame_desired, frame_observed,
                                frame_real_cube, frame_real), axis=0)
        video_recorder.add_frame(frame)
    video_recorder.save_video(video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="path to the log directory")
    parser.add_argument("video_path", help="video file to save (.avi file)")
    args = parser.parse_args()
    main(args.logdir, args.video_path)
