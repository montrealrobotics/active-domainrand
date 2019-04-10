import gym
import gym_ergojr
import time
from tqdm import tqdm
from common.envs.wrappers import RandomizedEnvWrapper

# MODE = "MANUAL" # slow but let's you see what's happening
MODE = "SPEED"  # as fast as possible

def no_op(x):
    pass


if MODE == "MANUAL":
    env = gym.make("ErgoReacher-Halfdisk-Randomized-Graphical-v0")  # looks nice
    timer = time.sleep
else:
    env = gym.make("ErgoReacher-Halfdisk-Randomized-Headless-v0")  # runs fast
    timer = no_op

env = RandomizedEnvWrapper(env=env, seed=0)

for _ in tqdm(range(100)):
    env.reset()
    env.randomize(randomized_values=["random"] * 8)  # 8 values to randomize over

    while True:
        action = env.action_space.sample()
        obs, rew, done, misc = env.step(action)
        timer(0.05)

        if done:
            break