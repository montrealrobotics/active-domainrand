import h5py
import matplotlib

matplotlib.use('Agg')

import random
import logging

import time
import numpy as np
import torch
import gym
import argparse
import os
import os.path as osp

from tqdm import tqdm, trange

from common.agents.ddpg_actor import DDPGActor
import poppy_helpers
import gym_ergojr
import cv2

parser = argparse.ArgumentParser(description='Real Robot Experiment Driver')

parser.add_argument('--nepisodes', type=int, default=25, help='Number of trials per *seed*')
parser.add_argument('--experiment-prefix', type=str, default='real', help='Prefix to append to logs')
parser.add_argument('--log-dir', type=str, default='results/real-robot', help='Log Directory Prefix')
parser.add_argument('--model-dir', type=str, default='saved-models/real-robot', help='Model Directory Prefix')

args = parser.parse_args()

TIMESTAMP = time.strftime("%y%m%d-%H%M%S")
MAX_EPISODE_STEPS = 100
EPISODES = args.nepisodes

# Policies to look for
policies = ['baseline', 'usdr', 'adr']

env = gym.make('ErgoReacher-Live-v1')
# env = gym.make('ErgoReacher-Graphical-Simple-Halfdisk-v1')

npa = np.array

img_buffer = []

if not osp.exists(args.log_dir):
    os.makedirs(args.log_dir)

with h5py.File("{}/{}-{}.hdf5".format(args.log_dir, args.experiment_prefix, TIMESTAMP), "w") as f:
    for policy_type in tqdm(policies):
        log_group = f.create_group(policy_type)
        model_path = osp.join(args.model_dir, policy_type)

        no_models = len(os.listdir(model_path))

        rewards = log_group.create_dataset("rewards", (no_models, EPISODES, MAX_EPISODE_STEPS), dtype=np.float32)
        distances = log_group.create_dataset("distances", (no_models, EPISODES, MAX_EPISODE_STEPS), dtype=np.float32)
        trajectories = log_group.create_dataset("trajectories", (no_models, EPISODES, MAX_EPISODE_STEPS, 24),
                                                dtype=np.float32)
        imgs = log_group.create_dataset("images", (no_models, EPISODES, MAX_EPISODE_STEPS, 480, 640, 3),
                                        dtype=np.uint8)

        tqdm.write('Starting analysis of {}'.format(policy_type))

        for model_idx, actorpth in enumerate(tqdm(os.listdir(model_path))):
            agent_policy = DDPGActor(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                agent_name='real-{}'.format(policy_type),
                load_agent=True,
                model_path=osp.join(model_path, actorpth)
            )

            for ep_num in trange(EPISODES):
                obs = env.reset()
                done = False
                cumulative = 0
                counter = 0
                while not done and counter < MAX_EPISODE_STEPS:
                    action = agent_policy.select_action(obs)
                    nobs, reward, done, misc = env.step(action)
                    # tqdm.write("obs: {} {} ".format(np.around(obs, 2), np.around(action, 2)))
                    cumulative += reward
                    trajectories[model_idx, ep_num, counter, :] = np.concatenate([obs, action, nobs])
                    rewards[model_idx, ep_num, counter] = reward
                    distances[model_idx, ep_num, counter] = misc["distance"]
                    imgs[model_idx, ep_num, counter, :, :, :] = np.copy(misc["img"])
                    # print(
                    #     np.around(trajectories[model_idx, ep_num, counter, :], 1),
                    #     np.around(rewards[model_idx, ep_num, counter], 4),
                    #     np.around(distances[model_idx, ep_num, counter], 4)
                    # )

                    obs = np.copy(nobs)
                    counter += 1

                tqdm.write('Episode: {}, Reward: {}'.format(ep_num, cumulative))

            # write to disk after every model run
            f.flush()
            env.reset()
