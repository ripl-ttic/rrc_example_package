from leap.leap_camera import LeapCamera
import pybullet as p
from code.grasping import Transform
import numpy as np


class TransformAndScale(object):
    def __init__(self, pos, ori, x_scale, y_scale, z_scale):
        self.T = Transform(pos=pos, ori=ori)
        self.scale = np.array([[x_scale, y_scale, z_scale]])

    def __call__(self, points):
        points = self.T(np.asarray(points))
        points = points * self.scale
        return points


class SphereMarker:
    def __init__(self, radius, position, color=(0, 1, 0, 0.5)):
        """
        Create a sphere marker for visualization

        Args:
            width (float): Length of one side of the cube.
            position: Position (x, y, z)
            orientation: Orientation as quaternion (x, y, z, w)
            color: Color of the cube as a tuple (r, b, g, q)
            """
        self.shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
        )
        self.body_id = p.createMultiBody(
            baseVisualShapeIndex=self.shape_id,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1],
        )

    def set_state(self, position):
        """Set pose of the marker.

        Args:
            position: Position (x, y, z)
        """
        orientation = [0, 0, 0, 1]
        p.resetBasePositionAndOrientation(
            self.body_id, position, orientation
        )

    def __del__(self):
        """
        Removes the visual object from the environment
        """
        # At this point it may be that pybullet was already shut down. To avoid
        # an error, only remove the object if the simulation is still running.
        if p.isConnected():
            p.removeBody(self.body_id)


class LeapPyBullet(object):
    def __init__(self, camera_transform, viz):
        self.camera = LeapCamera()
        self.T = camera_transform
        self.should_viz = viz
        self.viz_left = None
        self.viz_right = None
        self.colors = [
            [0, 1, 0, 0.5],
            [0, 1, 0.33, 0.5],
            [0, 1, 0.66, 0.5],
            [0, 1, 1, 0.5],
        ]

    def draw_hands(self, hands):
        for hand in hands:
            if hand['is_left']:
                self.viz_left = self._draw_hand(hand, self.viz_left)
            else:
                self.viz_right = self._draw_hand(hand, self.viz_right)

    def _draw_hand(self, hand, viz):
        if viz is None:
            viz = {
                'palm': SphereMarker(0.03, hand['palm'], self.colors[0]),
                'fingers': [
                    [
                        SphereMarker(0.01, point, self.colors[i])
                        for i, point in enumerate(finger)
                    ]
                    for finger in hand['fingers']
                ]
            }

        else:
            viz['palm'].set_state(hand['palm'])
            for markers, finger in zip(viz['fingers'], hand['fingers']):
                for i in range(4):
                    markers[i].set_state(finger[i])
        return viz

    def reset(self):
        del self.viz_left
        del self.viz_right
        self.viz_left = None
        self.viz_right = None

    def detect(self):
        hands = self.camera.detect()
        hands = [self.transform_points(hand) for hand in hands]
        if self.should_viz:
            self.draw_hands(hands)
        return hands

    def transform_points(self, hand):
        points = np.concatenate([[hand['palm']]] + [
            hand['fingers'][k] for k in ['0', '1', '2', '3', '4']
        ])
        points = self.T(points)
        return {
            'palm': points[0],
            'fingers': points[1:].reshape(5, 4, 3),
            'is_left': hand['is_left']
        }


if __name__ == '__main__':
    from trifinger_simulation.tasks import move_cube
    from code.make_env import make_training_env
    import time
    env = make_training_env(move_cube.sample_goal(3).to_dict(), 3,
                            action_space='position',
                            frameskip=3,
                            sim=True,
                            visualization=True,
                            reward_fn='competition_reward',
                            termination_fn='position_close_to_goal',
                            initializer='training_init',
                            episode_length=3750,
                            residual=False,
                            randomize=False,
                            skip_motions=True)
    obs = env.reset()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=0,
                                 cameraPitch=-89.99,
                                 cameraTargetPosition=[0, 0, 0])

    camera_transform = TransformAndScale(pos=np.array([0, 0, 0]),
                                         ori=np.array([1, 0, 0, 1]) * np.sqrt(2) / 2,
                                         x_scale=0.001,
                                         y_scale=0.001,
                                         z_scale=0.001)
    leap = LeapPyBullet(camera_transform, viz=True)

    while True:
        hands = leap.detect()
        time.sleep(0.01)
