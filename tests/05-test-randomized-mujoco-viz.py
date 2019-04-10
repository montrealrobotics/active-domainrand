import time
from timeit import default_timer as timer
import numpy as np
import tqdm
import gym
import common.envs
from common.envs.wrappers import RandomizedEnvWrapper

np.random.seed(1234)


env = gym.make('Pusher3DOFRandomized-v0')
env = RandomizedEnvWrapper(env=env, seed=0)

# obs = env.reset()

start = timer()
for i in tqdm.tqdm(range(100)):
    env.randomize(randomized_values=["random", "random", "random"])
    env.reset()
    for _ in range(200):
        obs, reward, done, _ = env.step(env.action_space.sample())
        env.render()
        print(obs)

env.close()
print(timer() - start)
