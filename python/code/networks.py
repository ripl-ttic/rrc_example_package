"""Define networks and ResidualPPO2."""
from dl.rl import PolicyBase, ValueFunctionBase, Policy, ValueFunction
from dl.modules import DiagGaussian, ProductDistribution, Normal
from dl import nest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import numpy as np
import gym
import gin


class ScaledNormal(Normal):
    def __init__(self, loc, scale, fac):
        super().__init__(loc, scale)
        self.fac = fac

    def mode(self):
        return self.mean * self.fac

    def sample(self):
        return super().sample() * self.fac

    def rsample(self):
        return super().rsample() * self.fac

    def log_prob(self, ac):
        return super().log_prob(ac / self.fac)

    def to_tensors(self):
        return {'loc': self.mean, 'scale': self.stddev}

    def from_tensors(self, tensors):
        return ScaledNormal(tensors['loc'], tensors['scale'], fac=self.fac)


class ObservationFilter(object):
    """Filters information to policy and value networks."""

    def get_policy_ob_shape(self, ob_space):
        if isinstance(ob_space, gym.spaces.Box):
            return np.prod(ob_space.shape)
        else:
            shapes = [
                ob_space[k].shape for k in ob_space.spaces
                if k not in ['clean', 'params']
            ]
            return np.sum([np.prod(s) for s in shapes])

    def get_value_fn_ob_shape(self, ob_space):
        if isinstance(ob_space, gym.spaces.Box):
            return np.prod(ob_space.shape)
        else:
            shapes = [ob_space[k].shape for k in ob_space['clean'].spaces]
            n = np.sum([np.prod(s) for s in shapes])
            return n + np.prod(ob_space['params'].shape)

    def get_policy_observation(self, ob):
        if isinstance(ob, torch.Tensor):
            return ob
        else:
            obs = {k: v for k, v in ob.items() if k not in ['clean', 'params']}
            obs = nest.flatten(obs)
            return torch.cat([torch.flatten(ob, 1) for ob in obs], dim=1)

    def get_value_fn_observation(self, ob):
        if isinstance(ob, torch.Tensor):
            return ob
        else:
            obs = nest.flatten(ob['clean']) + nest.flatten(ob['params'])
            return torch.cat([torch.flatten(ob, 1) for ob in obs], dim=1)


class PolicyNet(PolicyBase):
    """Policy network."""

    def __init__(self, observation_space, action_space, torque_std=0.05):
        self.torque_std = torque_std
        self.obs_filter = ObservationFilter()
        super().__init__(observation_space, action_space)

    def build(self):
        """Build."""
        n_in = self.obs_filter.get_policy_ob_shape(self.observation_space)
        self.fc1 = nn.Linear(n_in, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dist = DiagGaussian(256, self.action_space.shape[0],
                                 constant_log_std=False)
        for p in self.dist.fc_mean.parameters():
            nn.init.constant_(p, 0.)

    def forward(self, x):
        """Forward."""
        x = self.obs_filter.get_policy_observation(x)
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        dist = self.dist(x)
        return ScaledNormal(dist.mean, dist.stddev, fac=self.torque_std)


class RecurrentPolicyNet(PolicyBase):
    """Policy network."""

    def __init__(self, observation_space, action_space, torque_std=0.05):
        self.torque_std = torque_std
        self.obs_filter = ObservationFilter()
        super().__init__(observation_space, action_space)

    def build(self):
        """Build."""
        n_in = self.obs_filter.get_policy_ob_shape(self.observation_space)
        self.fc1 = nn.Linear(n_in, 256)
        self.fc2 = nn.Linear(256, 256)
        self.gru = nn.GRU(256, 256, 1)
        self.dist = DiagGaussian(256, self.action_space.shape[0],
                                 constant_log_std=False)
        for p in self.dist.fc_mean.parameters():
            nn.init.constant_(p, 0.)

    def forward(self, ob, state_in=None):
        """Forward."""
        if isinstance(ob, PackedSequence):
            x = ob.data
        else:
            x = ob
        x = self.obs_filter.get_policy_observation(x)
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if isinstance(ob, PackedSequence):
            x = PackedSequence(x, batch_sizes=ob.batch_sizes,
                               sorted_indices=ob.sorted_indices,
                               unsorted_indices=ob.unsorted_indices)
        else:
            x = x.unsqueeze(0)
        if state_in is None:
            x, state_out = self.gru(x)
        else:
            x, state_out = self.gru(x, state_in)
        if isinstance(x, PackedSequence):
            x = x.data
        else:
            x = x.squeeze(0)
        dist = self.dist(x)
        return ScaledNormal(dist.mean, dist.stddev, fac=self.torque_std), state_out


class TorqueAndPositionPolicyNet(PolicyBase):
    """Policy network."""

    def __init__(self, observation_space, action_space, torque_std=0.05,
                 position_std=0.001):
        self.obs_filter = ObservationFilter()
        self.torque_std = torque_std
        self.position_std = position_std
        super().__init__(observation_space, action_space)

    def build(self):
        """Build."""
        n_in = self.obs_filter.get_policy_ob_shape(self.observation_space)
        self.fc1 = nn.Linear(n_in, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dist_torque = DiagGaussian(256,
                                        self.action_space['torque'].shape[0],
                                        constant_log_std=False)
        self.dist_position = DiagGaussian(256,
                                          self.action_space['position'].shape[0],
                                          constant_log_std=False)
        for p in self.dist_torque.fc_mean.parameters():
            nn.init.constant_(p, 0.)
        for p in self.dist_position.fc_mean.parameters():
            nn.init.constant_(p, 0.)

    def forward(self, x):
        """Forward."""
        x = self.obs_filter.get_policy_observation(x)
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        d_torque = self.dist_torque(x)
        d_torque = ScaledNormal(d_torque.mean, d_torque.stddev,
                                fac=self.torque_std)
        d_position = self.dist_position(x)
        d_position = ScaledNormal(d_position.mean, d_position.stddev,
                                  fac=self.position_std)
        return ProductDistribution({'torque': d_torque,
                                    'position': d_position})


class RecurrentTorqueAndPositionPolicyNet(PolicyBase):
    """Policy network."""

    def __init__(self, observation_space, action_space, torque_std=0.05,
                 position_std=0.001):
        self.obs_filter = ObservationFilter()
        self.torque_std = torque_std
        self.position_std = position_std
        super().__init__(observation_space, action_space)

    def build(self):
        """Build."""
        n_in = self.obs_filter.get_policy_ob_shape(self.observation_space)
        self.fc1 = nn.Linear(n_in, 256)
        self.fc2 = nn.Linear(256, 256)
        self.gru = nn.GRU(256, 256, 1)
        self.dist_torque = DiagGaussian(256,
                                        self.action_space['torque'].shape[0],
                                        constant_log_std=False)
        self.dist_position = DiagGaussian(256,
                                          self.action_space['position'].shape[0],
                                          constant_log_std=False)
        for p in self.dist_torque.fc_mean.parameters():
            nn.init.constant_(p, 0.)
        for p in self.dist_position.fc_mean.parameters():
            nn.init.constant_(p, 0.)

    def forward(self, ob, state_in=None):
        """Forward."""
        if isinstance(ob, PackedSequence):
            x = ob.data
        else:
            x = ob
        x = self.obs_filter.get_policy_observation(x)
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if isinstance(ob, PackedSequence):
            x = PackedSequence(x, batch_sizes=ob.batch_sizes,
                               sorted_indices=ob.sorted_indices,
                               unsorted_indices=ob.unsorted_indices)
        else:
            x = x.unsqueeze(0)
        if state_in is None:
            x, state_out = self.gru(x)
        else:
            x, state_out = self.gru(x, state_in)
        if isinstance(x, PackedSequence):
            x = x.data
        else:
            x = x.squeeze(0)
        d_torque = self.dist_torque(x)
        d_torque = ScaledNormal(d_torque.mean, d_torque.stddev,
                                fac=self.torque_std)
        d_position = self.dist_position(x)
        d_position = ScaledNormal(d_position.mean, d_position.stddev,
                                  fac=self.position_std)
        dist = ProductDistribution({
            'torque': d_torque, 'position': d_position
        })
        return dist, state_out


class VFNet(ValueFunctionBase):
    """Value Function."""

    def build(self):
        """Build."""
        self.obs_filter = ObservationFilter()
        n_in = self.obs_filter.get_value_fn_ob_shape(self.observation_space)
        self.fc1 = nn.Linear(n_in, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.vf = nn.Linear(256, 1)

    def forward(self, x):
        """Forward."""
        if isinstance(x, PackedSequence):
            x = x.data
        x = self.obs_filter.get_value_fn_observation(x)
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.vf(x)


@gin.configurable
def policy_fn(env, torque_std=0.01):
    """Create policy."""
    return Policy(PolicyNet(env.observation_space, env.action_space,
                            torque_std=torque_std))


@gin.configurable
def torque_and_position_policy_fn(env, torque_std=0.01, position_std=0.001):
    """Create policy."""
    return Policy(TorqueAndPositionPolicyNet(env.observation_space,
                                             env.action_space,
                                             torque_std=torque_std,
                                             position_std=position_std))


@gin.configurable
def recurrent_policy_fn(env, torque_std=0.01):
    """Create policy."""
    return Policy(RecurrentPolicyNet(env.observation_space, env.action_space,
                                     torque_std=torque_std))


@gin.configurable
def recurrent_torque_and_position_policy_fn(env, torque_std=0.01,
                                            position_std=0.001):
    """Create policy."""
    return Policy(RecurrentTorqueAndPositionPolicyNet(env.observation_space,
                                                      env.action_space,
                                                      torque_std=torque_std,
                                                      position_std=position_std))


@gin.configurable
def value_fn(env):
    """Create value function network."""
    return ValueFunction(VFNet(env.observation_space, env.action_space))
