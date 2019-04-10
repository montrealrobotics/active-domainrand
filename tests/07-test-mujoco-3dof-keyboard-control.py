#/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
from pynput import keyboard
from pynput.keyboard import Key
import numpy as np
import gym
import common.envs
from common.envs.wrappers import RandomizedEnvWrapper

env = gym.make('Pusher3DOFUberHard-v0')
env = RandomizedEnvWrapper(env=env, seed=0)

reward = 0.

print('hi')
env.randomize(randomized_values=["random", "random"])
env.reset()
env.render()

ACTIONS = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0])
]

ACTION_KEYS = [Key.up, Key.down, Key.page_up , Key.page_down, Key.right, Key.left]



def on_press(key):
    global reward
    if key in ACTION_KEYS:
        s_, r, d, info = env.step(ACTIONS[ACTION_KEYS.index(key)])
        env.render()
        reward += r

        if d: 
            print(info['goal_dist'], reward)
            env.randomize(randomized_values=["random", "random"])
            env.reset()
            reward = 0

with keyboard.Listener(on_press=on_press) as listener:
    listener.join() 

env.close()
