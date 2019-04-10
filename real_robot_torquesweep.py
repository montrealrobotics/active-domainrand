import h5py
import matplotlib

matplotlib.use('Agg')
import time
import numpy as np
import gym
import argparse
import os
import os.path as osp

from tqdm import tqdm, trange

from common.agents.ddpg_actor import DDPGActor
import poppy_helpers
import gym_ergojr

parser = argparse.ArgumentParser(description='Real Robot Experiment Driver')

parser.add_argument('--nepisodes', type=int, default=25, help='Number of trials per *seed*')
parser.add_argument('--torques', type=list, nargs='+', default=[25, 50, 100, 200, 400],
                    help='torque settings to iterate')
parser.add_argument('--experiment-prefix', type=str, default='real', help='Prefix to append to logs')
parser.add_argument('--log-dir', type=str, default='results/real-robot', help='Log Directory Prefix')
parser.add_argument('--model-dir', type=str, default='saved-models/real-robot', help='Model Directory Prefix')
parser.add_argument('--cont', type=str, default='190329-180631', help='To continue existing file, enter timestamp here')

args = parser.parse_args()

if len(args.cont) == 0:
    TIMESTAMP = time.strftime("%y%m%d-%H%M%S")
    file_flag = "w"

else:
    TIMESTAMP = args.cont
    file_flag = "r+"

file_path = "{}/{}-{}.hdf5".format(args.log_dir, args.experiment_prefix, TIMESTAMP)

MAX_EPISODE_STEPS = 100
EPISODES = args.nepisodes
TORQUES = args.torques

# Policies to look for
policies = ['baseline', 'usdr', 'adr']

env = gym.make('ErgoReacher-Live-v1')
# env = gym.make('ErgoReacher-Graphical-Simple-Halfdisk-v1')

npa = np.array

img_buffer = []

if not osp.exists(args.log_dir):
    os.makedirs(args.log_dir)

with h5py.File(file_path, file_flag) as f:
    for policy_type in tqdm(policies, desc="approaches"):
        if policy_type not in f:  # if dataset doesn't have these tables
            log_group = f.create_group(policy_type)
            rewards = log_group.create_dataset("rewards", (no_models, len(TORQUES), EPISODES, MAX_EPISODE_STEPS),
                                               dtype=np.float32)
            distances = log_group.create_dataset("distances", (no_models, len(TORQUES), EPISODES, MAX_EPISODE_STEPS),
                                                 dtype=np.float32)
            trajectories = log_group.create_dataset("trajectories",
                                                    (no_models, len(TORQUES), EPISODES, MAX_EPISODE_STEPS, 24),
                                                    dtype=np.float32)
            imgs = log_group.create_dataset("images",
                                            (no_models, len(TORQUES), EPISODES, MAX_EPISODE_STEPS, 480, 640, 3),
                                            dtype=np.uint8, compression="lzf")
        else:  # if tables are in dataset, grab their pointers
            rewards = f.get("/{}/{}".format(policy_type, "rewards"))
            distances = f.get("/{}/{}".format(policy_type, "distances"))
            trajectories = f.get("/{}/{}".format(policy_type, "trajectories"))
            imgs = f.get("/{}/{}".format(policy_type, "images"))

        model_path = osp.join(args.model_dir, policy_type)

        no_models = len(os.listdir(model_path))

        tqdm.write('Starting analysis of {}'.format(policy_type))

        for model_idx, actorpth in enumerate(tqdm(os.listdir(model_path), desc="models....")):
            agent_policy = DDPGActor(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                agent_name='real-{}'.format(policy_type),
                load_agent=True,
                model_path=osp.join(model_path, actorpth)
            )

            for torque_idx, torque in enumerate(tqdm(TORQUES, desc="torques...")):

                for ep_num in trange(EPISODES, desc="episodes.."):
                    non_zero_steps = np.count_nonzero(trajectories[model_idx, torque_idx, ep_num], axis=1)

                    if np.count_nonzero(non_zero_steps) == 0:
                        obs = env.reset()
                        env.unwrapped.setSpeed(torque)
                        done = False
                        cumulative = 0
                        counter = 0
                        img_buffer = []
                        while counter < MAX_EPISODE_STEPS:
                            action = agent_policy.select_action(obs)
                            nobs, reward, _, misc = env.step(action)
                            cumulative += reward
                            trajectories[model_idx, torque_idx, ep_num, counter, :] = np.concatenate(
                                [obs, action, nobs])
                            rewards[model_idx, torque_idx, ep_num, counter] = reward
                            distances[model_idx, torque_idx, ep_num, counter] = misc["distance"]
                            img_buffer.append(np.copy(misc["img"]))

                            obs = np.copy(nobs)
                            counter += 1

                        imgs[model_idx, torque_idx, ep_num, :counter, :, :, :] = img_buffer

                    # tqdm.write('Episode: {}, Reward: {}'.format(ep_num, cumulative))

                    # write to disk after every model run
                    f.flush()

                env.reset()
