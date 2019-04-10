from importlib import import_module

import gym
import json
import numpy as np

import gym.spaces as spaces
import os.path as osp

from enum import Enum

from lxml import etree
import numpy as np

from common.envs.assets import MODEL_PATH
from common.envs.dimension import Dimension


class RandomizedEnvWrapper(gym.Wrapper):
    """Creates a randomization-enabled enviornment, which can change
    physics / simulation parameters without relaunching everything
    """

    def __init__(self, env, seed):
        super(RandomizedEnvWrapper, self).__init__(env)
        self.config_file = self.unwrapped.config_file

        self._load_randomization_dimensions(seed)
        self.unwrapped._update_randomized_params()
        self.randomized_default = ['random'] * len(self.unwrapped.dimensions)

    def _load_randomization_dimensions(self, seed):
        """ Helper function to load environment defaults ranges
        """
        self.unwrapped.dimensions = []

        with open(self.config_file, mode='r') as f:
            config = json.load(f)

        for dimension in config['dimensions']:
            self.unwrapped.dimensions.append(
                Dimension(
                    default_value=dimension['default'],
                    seed=seed,
                    multiplier_min=dimension['multiplier_min'],
                    multiplier_max=dimension['multiplier_max'],
                    name=dimension['name']
                )
            )

        nrand = len(self.unwrapped.dimensions)
        self.unwrapped.randomization_space = spaces.Box(0, 1, shape=(nrand,), dtype=np.float32)

    # TODO: The default is not informative of the type of randomize_values
    # TODO: The .randomize API is counter intuitive...
    def randomize(self, randomized_values=-1):
        """Creates a randomized environment, using the dimension and value specified 
        to randomize over
        """
        for dimension, randomized_value in enumerate(randomized_values):
            if randomized_value == 'default':
                self.unwrapped.dimensions[dimension].current_value = \
                    self.unwrapped.dimensions[dimension].default_value
            elif randomized_value != 'random' and randomized_value != -1:
                assert 0.0 <= randomized_value <= 1.0, "using incorrect: {}".format(randomized_value)
                self.unwrapped.dimensions[dimension].current_value = \
                    self.unwrapped.dimensions[dimension]._rescale(randomized_value)
            else:  # random
                self.unwrapped.dimensions[dimension].randomize()

        self.unwrapped._update_randomized_params()

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)