# /usr/bin/env python

import time
import gym
import common.envs
from common.envs.wrappers import RandomizedEnvWrapper

env = gym.make('HumanoidRandomizedEnv-v0')
env = RandomizedEnvWrapper(env=env, seed=0)

reward = 0.

env.randomize(randomized_values=["random", "random", "random", "random", "random", "random"])
env.reset()
env.render()

d = False
while True:
    s_, r, d, info = env.step(env.action_space.sample())
    time.sleep(0.1)

    if d:
        env.randomize(
            randomized_values=["random", "random", "random", "random", "random", "random"])
        env.reset()
    env.render()
