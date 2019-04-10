import os
import time
import logging
import numpy as np

from common.utils.rollout_evaluation import check_new_best


def reshow_hyperparameters(args, paths):
    logging.info("---------------------------------------")
    logging.info("Arguments for Finished Experiment:")
    for arg in vars(args):
        logging.info("{}: {}".format(arg, getattr(args, arg)))
    logging.info("Relevant Paths for Finished Experiment:")
    for key, value in paths.items():
        logging.info("{}: {}".format(key, value))
    logging.info("---------------------------------------\n")


def log_hyperparameters(experiment_name, path, args, paths):
    # Write Hyperparameters to file

    # Set up logging
    filename = os.path.join(path, 'training.log')
    log_format = "{} S{}: %(asctime)s %(filename)s:%(lineno)d %(message)s".format(
        args.experiment_prefix, args.seed)

    log_level = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(
                format=log_format,
                level=log_level,
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=[
                    logging.FileHandler(filename),
                    logging.StreamHandler()
                ])

    logging.info("---------------------------------------")
    logging.info("Current Arguments for Experiment {}:".format(experiment_name))
    with open(os.path.join(path, "hps.txt"), 'w') as f:
        for arg in vars(args):
            logging.info("{}: {}".format(arg, getattr(args, arg)))
            f.write("{}: {}\n".format(arg, getattr(args, arg)))
        logging.info("Relevant Paths for Experiment {}:".format(experiment_name))
        for key, value in paths.items():
            logging.info("{}: {}".format(key, value))
            f.write("{}: {}\n".format(key, value))
    logging.info("---------------------------------------\n")


def create_directories(args, base_path, experiment_path, no_seed_experiment_path, 
    outer_directories, inner_directories):
    final_paths = dict()

    for outer_dir in outer_directories:
        final_paths[outer_dir] = dict()

        if outer_dir == 'logs':
            experiment_directory = os.path.join(base_path.format(outer_dir), experiment_path)
            model_path = os.path.join(experiment_directory, 'models')
            particle_path = os.path.join(experiment_directory, 'models', 'particles')
            paper_ready_path = os.path.join(base_path.format(outer_dir), no_seed_experiment_path, 'paper')

            final_paths['experiment'] = experiment_directory
            final_paths['models'] = model_path
            final_paths['particles'] = particle_path
            final_paths['paper'] = paper_ready_path
            
            if not os.path.exists(experiment_directory):
                os.makedirs(experiment_directory)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if not os.path.exists(particle_path):
                os.makedirs(particle_path)
            if not os.path.exists(paper_ready_path):
                os.makedirs(paper_ready_path)
            
        for inner_dir in inner_directories:
            path = os.path.join(base_path.format(outer_dir), experiment_path, inner_dir)
            if not os.path.exists(path):
                os.makedirs(path)

            final_paths[outer_dir][inner_dir] = path

    return final_paths


def setup_experiment_logs(args):
    experiment_name = args.experiment_name
    base_path = 'results/{}/'.format(args.subparser_name)
    base_path += '{}'
    experiment_path = '{}/{}/{}-exp{}-{}/{}'.format(experiment_name, args.randomized_eval_env_id, args.agent_name, 
        args.experiment_prefix, args.nagents, args.seed)
    no_seed_experiment_path = '{}/{}/{}-exp{}-{}/'.format(experiment_name, args.randomized_eval_env_id, 
        args.agent_name, args.experiment_prefix, args.nagents)
    final_paths = paths = None

    if experiment_name == 'unfreeze-policy' or experiment_name == 'gail-baseline':
        outer_directories = ['plots', 'logs']
        inner_directories = ['policy', 'sampling', 'groundtruth']

        final_paths = create_directories(args, base_path, experiment_path, no_seed_experiment_path,
            outer_directories, inner_directories)
        paths = {
            'path': final_paths['experiment'],
            'paper': final_paths['paper'],
            'models': final_paths['models'],
            'particles': final_paths['particles'],
            'policy_plots': final_paths['plots']['policy'],
            'sampling_plots': final_paths['plots']['sampling'],
            'policy_logs': final_paths['logs']['policy'],
            'sampling_logs': final_paths['logs']['sampling'],
            'groundtruth_logs': final_paths['logs']['groundtruth'],
        }

    elif experiment_name == 'adaptive-randomization':
        outer_directories = ['plots', 'logs']
        inner_directories = ['policy', 'sampling', 'groundtruth']

        final_paths = create_directories(args, base_path, experiment_path, no_seed_experiment_path,
            outer_directories, inner_directories)
        paths = {
            'path': final_paths['experiment'],
            'paper': final_paths['paper'],
            'models': final_paths['models'],
            'particles': final_paths['particles'],
            'policy_plots': final_paths['plots']['policy'],
            'sampling_plots': final_paths['plots']['sampling'],
            'policy_logs': final_paths['logs']['policy'],
            'sampling_logs': final_paths['logs']['sampling'],
            'groundtruth_logs': final_paths['logs']['groundtruth'],
        }

    elif experiment_name == 'batch-reward-analysis':
        outer_directories = ['plots', 'logs']
        inner_directories = ['groundtruth']

        final_paths = create_directories(args, base_path, experiment_path, no_seed_experiment_path,
            outer_directories, inner_directories)
        paths = {
            'path': final_paths['experiment'],
            'paper': final_paths['paper'],
            'models': final_paths['models'],
            'particles': final_paths['particles'],
            'groundtruth_plots': final_paths['plots']['groundtruth'],
            'groundtruth_logs': final_paths['logs']['groundtruth'],
        }

    elif experiment_name == 'bootstrapping':
        outer_directories = ['plots', 'logs']
        inner_directories = ['trajectories', 'sampling', 'groundtruth', 'policy']

        final_paths = create_directories(args, base_path, experiment_path, no_seed_experiment_path,
            outer_directories, inner_directories)
        paths = {
            'path': final_paths['experiment'],
            'paper': final_paths['paper'],
            'models': final_paths['models'],
            'particles': final_paths['particles'],
            'policy_plots': final_paths['plots']['policy'],
            'sampling_plots': final_paths['plots']['sampling'],
            'policy_logs': final_paths['logs']['policy'],
            'sampling_logs': final_paths['logs']['sampling'],
            'groundtruth_logs': final_paths['logs']['groundtruth'],
        }

    log_hyperparameters(experiment_name, final_paths['experiment'], args, paths)
    return paths

class StatsLogger:
    def __init__(self, args):
        self.learning_curve_timesteps = []
        self.ref_learning_curve_mean = []
        self.ref_learning_curve_median = []

        self.rand_learning_curve_mean = []
        self.rand_learning_curve_median = []

        self.hard_learning_curve_mean = []
        self.hard_learning_curve_median = []
        
        self.ref_final_dists_mean = []
        self.rand_final_dists_mean = []
        self.hard_final_dists_mean = []

        self.ref_final_dists_median = []
        self.rand_final_dists_median = []
        self.hard_final_dists_median = []

        self.randomized_discrim_scores_mean = []
        self.reference_discrim_scores_mean = []
        self.randomized_discrim_scores_median = []
        self.reference_discrim_scores_median = []

        self.current_best = -np.inf if args.randomized_env_id.find('Lunar') != -1 else np.inf

    def update(self, args, paths, info):
        self.randomized_discrim_scores_mean.append(info['randomized_discrim_score_mean'])
        self.reference_discrim_scores_mean.append(info['reference_discrim_score_mean'])
        self.randomized_discrim_scores_median.append(info['randomized_discrim_score_median'])
        self.reference_discrim_scores_median.append(info['reference_discrim_score_median'])

        np.savez('{}/discriminator-scores-seed{}.npz'.format(paths['paper'], args.seed),
            randomized_discrim_scores_mean=self.randomized_discrim_scores_mean,
            reference_discrim_scores_mean=self.reference_discrim_scores_mean,
            randomized_discrim_scores_median=self.randomized_discrim_scores_median,
            reference_discrim_scores_median=self.reference_discrim_scores_median
        )

        new_best = False

        if not args.freeze_agent:
            self.learning_curve_timesteps.append(info['agent_timesteps'])
            self.ref_learning_curve_mean.append(info['agent_reference_eval_rewards_mean'])
            self.ref_learning_curve_median.append(info['agent_reference_eval_rewards_median'])
            self.rand_learning_curve_mean.append(info['agent_randomized_eval_rewards_mean'])
            self.rand_learning_curve_median.append(info['agent_randomized_eval_rewards_median'])

            evaluation_key = 'agent_randomized_eval_rewards_median'
            if args.randomized_env_id.find('Lunar') == -1:
                self.ref_final_dists_mean.append(info['final_dist_ref_mean'])
                self.rand_final_dists_mean.append(info['final_dist_rand_mean'])
                self.ref_final_dists_median.append(info['final_dist_ref_median'])
                self.self.rand_final_dists_median.append(info['final_dist_rand_median'])

                evaluation_key = 'final_dist_rand_median'

            self.hard_learning_curve_mean.append(info['agent_hard_eval_rewards_mean'])
            self.hard_learning_curve_median.append(info['agent_hard_eval_rewards_median'])
            self.hard_final_dists_mean.append(info['final_dist_hard_mean'])
            self.hard_final_dists_median.append(info['final_dist_hard_median'])
            
            np.savez('{}/learning-curves-seed{}.npz'.format(paths['paper'], args.seed),
                ref_learning_curve_mean=self.ref_learning_curve_mean,
                ref_learning_curve_median=self.ref_learning_curve_mean,
                rand_learning_curve_mean=self.rand_learning_curve_mean,
                rand_learning_curve_median=self.rand_learning_curve_mean,
                ref_final_dists_mean=self.ref_final_dists_mean,
                rand_final_dists_mean=self.rand_final_dists_mean,
                ref_final_dists_median=self.ref_final_dists_median,
                rand_final_dists_median=self.rand_final_dists_median,
                hard_learning_curve_mean=self.hard_learning_curve_mean,
                hard_learning_curve_median=self.hard_learning_curve_median,
                hard_final_dists_mean=self.hard_final_dists_mean,
                hard_final_dists_median=self.hard_final_dists_median,
                learning_curve_timesteps=self.learning_curve_timesteps
            )

            if check_new_best(args.randomized_env_id, info[evaluation_key], self.current_best):
                self.current_best = info[evaluation_key]
                new_best = True

        info['current_best'] = self.current_best
        
        for key, value in sorted(info.items()):
            logging.info("{}: {}".format(key, value))

        return new_best

