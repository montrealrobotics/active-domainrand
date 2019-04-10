#!/usr/bin/env python
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

env = gym.make("ErgoReacher4DOFRandomizedHardVisual-v0")
env = RandomizedEnvWrapper(env=env, seed=0)

env.reset()
env.render()

ACTION_KEYS = [Key.up, Key.down, Key.page_up , Key.page_down, Key.right, Key.left, Key.home, Key.end, Key.alt]

def on_press(key):
    if key == Key.tab: env.reset()
    if key in ACTION_KEYS:
        action = np.zeros(4)
        if key != Key.alt:
            index = ACTION_KEYS.index(key)
            multiplier = 1 if index % 2 == 0 else -1

            act_idx = index // 2
            action[act_idx] = multiplier

        s_, r, d, info = env.step(action)
        print(info)
        env.render()
        if d: 
            env.randomize(randomized_values=["random", "random", "random"])
            env.reset()

with keyboard.Listener(on_press=on_press) as listener:
    listener.join() 

env.close()