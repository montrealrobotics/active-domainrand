import matplotlib
matplotlib.use('Agg')

import re
import os
import glob
import numpy as np
import torch
import gym
import argparse
import json
import logging

from itertools import combinations

from common.utils.logging import setup_experiment_logs

from experiments.domainrand.args import get_args, check_args


def get_converged_modelpaths(paths):
    """
    Function to find the learning curves and best generalization curves for each seed
    """

    paper_path = paths['paper']
    agent_paths = os.listdir(paper_path)

    learning_curves_files = glob.glob(os.path.join(os.getcwd(), paper_path, 'learning-curves*.npz'))
    generalization_files = glob.glob(os.path.join(os.getcwd(), paper_path, 'best-generalization*.npz'))

    print(learning_curves_files)

    learning_curves_combinations = combinations(learning_curves_files, 5)
    generalization_combinations = combinations(generalization_files, 5)

    agent_name_start = paper_path.find('v0') + 3
    agent_name_end = paper_path.find('-exp')

    agent_name = paper_path[agent_name_start:agent_name_end]

    return agent_name, list(learning_curves_files), generalization_files


if __name__ == '__main__':
    args = get_args()
    experiment_name = 'unfreeze-policy' if not args.use_bootstrapping_results else 'bootstrapping'
    paths = setup_experiment_logs(experiment_name=experiment_name, args=args)
    check_args(args, experiment_name=experiment_name)

    agent_name, learning_curves_files, generalization_files = get_converged_modelpaths(paths)
    nseeds = len(learning_curves_files)

    nmetrics = len(np.load(learning_curves_files[0]).files)

    # Learning curves 
    # Find Max Length and resize each array to that length

    # for combination in combinations: for lc in combination

    # for i, learning_curves_files in enumerate(learning_curves_combinations):
    #     print(i, learning_curves_files, '\n\n')
    max_length = 0
    for lc in learning_curves_files:
        loaded_curve = np.load(lc)['ref_learning_curve_mean']
        if loaded_curve.shape[0] > max_length:
            max_length = loaded_curve.shape[0]
    
    all_curves = np.zeros((nseeds, max_length))
    all_metrics = {}

    for metric in np.load(learning_curves_files[0]).files:
        all_metrics[metric] = np.copy(all_curves)

    # Load each seed's metric (5 - 9 per file)
    for seed, lc in enumerate(learning_curves_files):
        loaded_curve = np.load(lc)
        for metric in loaded_curve.files:
            # hacky "Broadcast" of array
            length = len(loaded_curve[metric])
            all_metrics[metric][seed][:length] = loaded_curve[metric]
            # If not same size, some will be 0s, do so we can use np.nanmean
            try:
                all_metrics[metric][seed][all_metrics[metric][seed] == 0] = np.nan
            except:
                pass
            
    all_metrics['label'] = np.array([agent_name])

    np.savez(os.path.join(paths['paper'],'{}-{}-batched-learning-curves.npz'.format(0, agent_name)), **all_metrics)

    # Generalization Curves
    loaded_curve = np.load(generalization_files[0])['generalization_metric']
    generalization_shape = loaded_curve.shape

    all_seeds_generalization = np.zeros((nseeds,) + generalization_shape)

    for seed, lc in enumerate(generalization_files):
        loaded_curve = np.load(lc)
        all_seeds_generalization[seed] = loaded_curve['generalization_metric']

    np.savez(os.path.join(paths['paper'],'{}-batched-generalizations.npz'.format(agent_name)), 
        all_seeds_generalization=all_seeds_generalization)
