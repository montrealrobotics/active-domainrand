import common.envs
import gym
import time

env = gym.make("FetchReachDenseDS-v1")

print("action dim: {}, obs dim: {}".format(env.action_space, env.observation_space))

# exploration
exploration_actions = [ # the actions are for the end effector, thus implying IK
    [1, 0, 0, 0], # forward
    [-1, 0, 0, 0], # backward
    [0, 1, 0, 0], # left (from robot's perspective
    [0, -1, 0, 0], # right
    [0, 0, 1, 0], # up
    [0, 0, -1, 0] # down
#     [0, 0, 0, 1], # gripper open/close, unused in fetch
#     [0, 0, 0, -1] # gripper open/close, unused in fetch
]
exploration_length = 50
env.reset()
done = False
i = 0
exploration_action_idx = 0
while True:
    action = exploration_actions[exploration_action_idx]
    obs, rew, done, misc = env.step(action)
    env.render()
    i += 1
    if i % exploration_length == 0:
        exploration_action_idx += 1
        if exploration_action_idx == len(exploration_actions):
            break
    time.sleep(0.02)

# # randome movement
# for i in range(5):
#     env.reset()
#     done = False
#
#     while not done:
#         action = env.action_space.sample()
#         obs, rew, done, misc = env.step(action)
#         print (obs, rew, misc)
#         env.render()
