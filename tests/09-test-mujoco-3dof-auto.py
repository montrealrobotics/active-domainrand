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

action_change_freq = 50

for env_idx in tqdm(range(10)):
    env.reset()
    env.render()

    for action in actions:

        for _ in range(action_change_freq):
            _, _, _, _ = env.step(action)
            env.render()
            time.sleep(0.01)

# print (np.min(env.unwrapped.qposes, axis=0),np.max(env.unwrapped.qposes, axis=0))
