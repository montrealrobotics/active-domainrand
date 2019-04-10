import matplotlib
matplotlib.use('Agg')

import random
import logging

import numpy as np
import torch
import gym
import argparse
import os
import glob
import json

from common.agents.ddpg.ddpg import DDPG
from common.agents.ddpg_actor import DDPGActor
from common.agents.svpg_simulator_agent import SVPGSimulatorAgent
from common.envs import *
from common.utils.visualization import Visualizer
from common.utils.logging import setup_experiment_logs, reshow_hyperparameters

from experiments.domainrand.args import get_args, check_args

from common.utils.rollout_evaluation import evaluate_policy
from common.envs.randomized_vecenv import make_vec_envs

NEVAL_EPISODES = 10
N_PROCESSES = 5
N_SEEDS = 5

if __name__ == '__main__':
    args = get_args()
    paths = setup_experiment_logs(experiment_name='unfreeze-policy', args=args)
    check_args(args, experiment_name='unfreeze-policy')
    reference_env = gym.make(args.reference_env_id)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    environment_prototype = 'Pusher3DOFGeneralization{}{}-v0'

    rewards_grid = np.zeros((3, 3, 5, NEVAL_EPISODES))
    finaldists_grid = np.zeros((3, 3, 5, NEVAL_EPISODES))

    for i in range(3):
        for j in range(3):
            randomized_env = make_vec_envs(environment_prototype.format(i, j), args.seed + i + j, N_PROCESSES)            
            actor_paths = glob.glob(os.path.join(os.getcwd(), paths['paper'], 'best-seed*_actor.pth'))
            print(actor_paths)
            for actor_idx, actor_path in enumerate(actor_paths):
                agent_policy = DDPGActor(
                    state_dim=reference_env.observation_space.shape[0], 
                    action_dim=reference_env.action_space.shape[0], 
                    agent_name=args.agent_name,
                    load_agent=True,
                    model_path=actor_path
                )
    
                rewards_rand, dist_rand = evaluate_policy(nagents=N_PROCESSES,
                                                      env=randomized_env,
                                                      agent_policy=agent_policy,
                                                      replay_buffer=None,
                                                      eval_episodes=NEVAL_EPISODES // N_PROCESSES,
                                                      max_steps=args.max_env_timesteps,
                                                      return_rewards=True,
                                                      add_noise=False,
                                                      log_distances=True)

                rewards_grid[i, j, actor_idx, :] = rewards_rand
                finaldists_grid[i, j, actor_idx, :] = dist_rand 
   
    reshow_hyperparameters(args, paths)
    print(finaldists_grid)

    np.savez(os.path.join(paths['paper'], 'grid_generalization.npz'), 
        rewards_grid=rewards_grid,
        finaldists_grid=finaldists_grid
    )
