import matplotlib
matplotlib.use('Agg')

import random
import logging

import numpy as np
import torch
import gym
import argparse
import os

from common.agents.ddpg.ddpg import DDPG
from common.agents.ddpg_actor import DDPGActor
from common.utils.visualization import Visualizer
from common.utils.sim_agent_helper import generate_simulator_agent
from common.utils.logging import setup_experiment_logs, reshow_hyperparameters, StatsLogger

from experiments.domainrand.args import get_args, check_args


if __name__ == '__main__':
    args = get_args()
    paths = setup_experiment_logs(args)
    check_args(args)
    
    stats_logger = StatsLogger(args)
    visualizer = Visualizer(randomized_env_id=args.randomized_eval_env_id, seed=args.seed)

    reference_env = gym.make(args.reference_env_id)    

    if args.freeze_agent:
        # only need the actor
        agent_policy = DDPGActor(
            state_dim=reference_env.observation_space.shape[0], 
            action_dim=reference_env.action_space.shape[0], 
            agent_name=args.agent_name,
            load_agent=args.load_agent
        )
    else:
        agent_policy = DDPG(
            state_dim=reference_env.observation_space.shape[0], 
            action_dim=reference_env.action_space.shape[0], 
            agent_name=args.agent_name,
        )

        if args.load_agent:
            agent_policy.load_model()

    
    simulator_agent = generate_simulator_agent(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    svpg_timesteps = 0

    while simulator_agent.agent_timesteps < args.max_agent_timesteps:
        if svpg_timesteps % args.plot_frequency == 0:
            generalization_metric = visualizer.generate_ground_truth(simulator_agent, agent_policy, svpg_timesteps, 
                log_path=paths['groundtruth_logs'])

            np.savez('{}/generalization-seed{}.npz'.format(paths['paper'], args.seed),
                generalization_metric=generalization_metric,
                svpg_timesteps=svpg_timesteps,
                learning_curve_timesteps=simulator_agent.agent_timesteps
            )

            visualizer.plot_reward(simulator_agent, agent_policy, 
                svpg_timesteps, log_path=paths['policy_logs'], plot_path=paths['policy_plots'])
            visualizer.plot_value(simulator_agent, agent_policy, 
                svpg_timesteps, log_path=paths['policy_logs'], plot_path=paths['policy_plots'])
            visualizer.plot_discriminator_reward(simulator_agent, agent_policy, 
                svpg_timesteps, log_path=paths['policy_logs'], plot_path=paths['policy_plots'])

            if not args.freeze_svpg:
                visualizer.plot_sampling_frequency(simulator_agent, agent_policy, 
                    svpg_timesteps, log_path=paths['sampling_logs'], plot_path=paths['sampling_plots'])
                
        logging.info("SVPG TS: {}, Agent TS: {}".format(svpg_timesteps, simulator_agent.agent_timesteps))
        
        solved, info = simulator_agent.select_action(agent_policy)
        svpg_timesteps += 1
        
        if info is not None:
            new_best = stats_logger.update(args, paths, info)

            if new_best:
                agent_policy.save(filename='best-seed{}'.format(args.seed), directory=paths['paper'])
                if args.save_particles:
                    simulator_agent.svpg.save(directory=paths['particles'])

                generalization_metric = visualizer.generate_ground_truth(simulator_agent, agent_policy, svpg_timesteps, 
                log_path=paths['groundtruth_logs'])

                np.savez('{}/best-generalization-seed{}.npz'.format(paths['paper'], args.seed),
                    generalization_metric=generalization_metric,
                    svpg_timesteps=svpg_timesteps,
                    learning_curve_timesteps=simulator_agent.agent_timesteps
                )
            
            if solved:
                logging.info("[SOLVED]")

    agent_policy.save(filename='final-seed{}'.format(args.seed), directory=paths['paper'])
    visualizer.plot_reward(simulator_agent, agent_policy, 
            svpg_timesteps, log_path=paths['policy_logs'], plot_path=paths['policy_plots'])
    visualizer.plot_sampling_frequency(simulator_agent, agent_policy, 
        svpg_timesteps, log_path=paths['sampling_logs'], plot_path=paths['sampling_plots'])
    reshow_hyperparameters(args, paths)
