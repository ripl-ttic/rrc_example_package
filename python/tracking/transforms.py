import numpy as np
from scipy.spatial.transform import Rotation


def get_rotation_between_vecs(v1, v2):
    """Rotation from v1 to v2."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    if np.linalg.norm(axis) == 0:
        if np.dot(v1, v2) > 0:
            # zero rotation
            return np.array([0, 0, 0, 1])
        else:
            # 180 degree rotation

            # get perp vec
            axis = np.random.rand(3)
            axis /= np.linalg.norm(axis)
            axis -= np.dot(axis, v1)
            axis /= np.linalg.norm(axis)
            assert np.dot(axis, v1) == 0
            assert np.dot(axis, v2) == 0
            return np.array([axis[0], axis[1], axis[2], 0])
    axis /= np.linalg.norm(axis)
    angle = np.arccos(v1.dot(v2))
    quat = np.zeros(4)
    quat[:3] = axis * np.sin(angle / 2)
    quat[-1] = np.cos(angle / 2)
    return quat


class Transform(object):
    def __init__(self, pos=None, ori=None, T=None):
        if pos is not None and ori is not None:
            self.rot = Rotation.from_quat(ori)
            self.R = self.rot.as_matrix()
            print(self.R, type(self.R))
            self.pos = pos
            self.T = np.eye(4)
            self.T[:3, :3] = self.R
            self.T[:3, -1] = self.pos
        elif T is not None:
            self.T = T
            self.R = T[:3, :3]
            self.pos = T[:3, -1]
        else:
            raise ValueError("You must specify T or both pos and ori.")

    def adjoint(self):
        def _skew(p):
            return np.array([
                [0, -p[2], p[1]],
                [p[2], 0, -p[0]],
                [-p[1], p[0], 0],
            ])

        adj = np.zeros((6, 6))
        adj[:3, :3] = self.R
        adj[3:, 3:] = self.R
        adj[3:, :3] = _skew(self.pos).dot(self.R)
        return adj

    def inverse(self):
        T = np.eye(4)
        T[:3, :3] = self.R.T
        T[:3, -1] = -self.R.T.dot(self.pos)
        return Transform(T=T)

    def __call__(self, x):
        if isinstance(x, Transform):
            return Transform(T=self.T.dot(x.T))
        else:
            # check for different input forms
            one_dim = len(x.shape) == 1
            homogeneous = x.shape[-1] == 4
            if one_dim:
                x = x[None]
            if not homogeneous:
                x_homo = np.ones((x.shape[0], 4))
                x_homo[:, :3] = x
                x = x_homo

            # transform points
            x = self.T.dot(x.T).T

            # create output to match input form
            if not homogeneous:
                x = x[:, :3]
            if one_dim:
                x = x[0]
            return x
