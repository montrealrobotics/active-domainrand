import time
from timeit import default_timer as timer
import numpy as np
import tqdm
import gym
import common.envs
from common.envs.wrappers import RandomizedEnvWrapper

np.random.seed(1234)


env = gym.make('PusherRandomized-v0')
env = RandomizedEnvWrapper(env=env, seed=0)

obs = env.reset()

start = timer()
for i in tqdm.tqdm(range(int(1e6))):
    env.randomize(randomized_values=["random", "random", "random"])
print(timer() - start)
