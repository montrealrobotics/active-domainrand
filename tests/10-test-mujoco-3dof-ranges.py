#!/usr/bin/env python
import time

import numpy as np
import gym
from tqdm import tqdm

import common.envs
from common.envs.wrappers import RandomizedEnvWrapper

env = gym.make('Pusher3DOFRandomized-v0')
env = RandomizedEnvWrapper(env=env, seed=0)# env.randomize()

actions = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0])
]
actions.reverse()

discretization = 50

randomized_values = ["default", "default"]
values = np.linspace(0, 1, discretization)

for dim in range(2):
    for i in tqdm(range(discretization)):
        rands = randomized_values
        rands[dim] = values[i]
        env.randomize(rands)
        env.reset()

        for _ in range(50):
            env.step(env.action_space.sample())
            env.render()
            time.sleep(0.01)

# print (np.min(env.unwrapped.qposes, axis=0),np.max(env.unwrapped.qposes, axis=0))
