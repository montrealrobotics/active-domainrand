import os
import numpy as np
import logging
import matplotlib.pyplot as plt

from common.utils.rollout_evaluation import evaluate_policy

from common.envs.randomized_vecenv import make_vec_envs


DISPLAY_FREQUENCY = 5
PLOTCOLORS = ['red', 'orange', 'purple', 'brown', 'darkgreen', 'teal', 'coral', 
        'lightblue', 'lime', 'lavender', 'turquoise',
         'tan', 'salmon', 'gold', 'darkred', 'darkblue']
plt.rcParams.update({'font.size': 22})

logger = logging.getLogger(__name__)

def get_config(env_id):
    if env_id.find('Lunar') != -1:
        return {
            'ylabel': 'Average Reward',
            'ylim_low': -200,
            'ylim_high': 300,
            'hist_ylim_high': 400,
            'hist_xlims': [8, 20],
            'solved': 200,
            'npoints': 50,
            'max_steps': 1000
        }

    elif env_id.find('Ergo') != -1:
        return {
            'ylabel': 'Average Final End Effector Distance from Goal',
            'ylim_low': 0,
            'ylim_high': 1.0,
            'hist_ylim_high': 1000,
            'solved': 0.025,
            'npoints': 40,
            'max_steps': 100,
        }
    else:
        return {
            'ylabel': 'Average Final Puck Distance from Goal',
            'ylim_low': 0,
            'ylim_high': 1.0,
            'hist_ylim_high': 1000,
            'solved': 0.25,
            'npoints': 50,
            'max_steps': 100
        }


class Visualizer(object):
    def __init__(self, randomized_env_id, seed, neval_eps=5):
        self.evaluation_scores = None
        self.randomized_env_id = randomized_env_id
        self.config = get_config(randomized_env_id)

        self.npoints = self.config['npoints']
        self.max_steps = self.config['max_steps']
        self.ground_truth_x = np.linspace(0, 1, num=self.npoints)
        
        self.neval_eps = neval_eps
        self.seed = seed

        self.log_distances = randomized_env_id.find('Lunar') == -1
        self.randomized_env = make_vec_envs(self.randomized_env_id, self.seed, self.neval_eps)

    def generate_ground_truth(self, simulator_agent, agent_policy, timesteps, log_path):
        logger.debug('Generating ground truth...')

        self.evaluation_scores = [None] * simulator_agent.nparams
        default_values = [['default'] * simulator_agent.nparams] * self.neval_eps

        for randomized_dimension in range(simulator_agent.nparams):
            evaluation_array = []
            for i, x in enumerate(self.ground_truth_x):
                if i % DISPLAY_FREQUENCY == 0:
                    logger.info("Dim: {}, Index: {}/{}".format(randomized_dimension, i, len(self.ground_truth_x)))

                values = default_values
                for index in range(self.neval_eps):
                    values[index][randomized_dimension] = x

                self.randomized_env.randomize(values)
                
                randomized_rewards, final_distances = evaluate_policy(nagents=self.neval_eps, env=self.randomized_env, 
                    agent_policy=agent_policy, replay_buffer=None, eval_episodes=1,
                    max_steps=self.max_steps, return_rewards=True, add_noise=False, log_distances=self.log_distances)

                if self.log_distances:
                    evaluation_array.append(np.array([np.mean(final_distances), np.std(final_distances)]))
                else:
                    evaluation_array.append(np.array([np.mean(randomized_rewards), np.std(randomized_rewards)]))

            self.evaluation_scores[randomized_dimension] = np.array(evaluation_array)

        self.evaluation_scores = np.array(self.evaluation_scores)
        
        for randomized_dimension in range(simulator_agent.nparams):
            name = self.randomized_env.get_dimension_name(randomized_dimension)

        np.savez('{}.npz'.format(os.path.join(log_path, 'raw_rewards-{}'.format(timesteps))), 
            raw_rewards=self.evaluation_scores)

        logger.info('Ground truth generated.')
        return self.evaluation_scores


    def plot_discriminator_reward(self, simulator_agent, agent_policy, timesteps, plot_path, log_path):
        logger.debug('Generating ground truth...')

        default_values = [['default'] * simulator_agent.nparams] * self.neval_eps

        for randomized_dimension in range(simulator_agent.nparams):
            evaluation_array_mean = []
            evaluation_array_median = []
            for i, x in enumerate(self.ground_truth_x):
                if i % DISPLAY_FREQUENCY == 0:
                    logger.info("Dim: {}, Index: {}/{}".format(randomized_dimension, i, len(self.ground_truth_x)))

                values = default_values
                for index in range(self.neval_eps):
                    values[index][randomized_dimension] = x

                self.randomized_env.randomize(values)
                trajectory = evaluate_policy(nagents=self.neval_eps,
                                         env=self.randomized_env,
                                         agent_policy=agent_policy,
                                         replay_buffer=None,
                                         eval_episodes=1,
                                         max_steps=self.max_steps,
                                         freeze_agent=True,
                                         add_noise=False,
                                         log_distances=self.log_distances)

                trajectory = [trajectory[i] for i in range(self.neval_eps)]
                trajectory = np.concatenate(trajectory)

                randomized_discrim_score_mean, randomized_discrim_score_median, _ = \
                    simulator_agent.discriminator_rewarder.get_score(trajectory)

                evaluation_array_mean.append(randomized_discrim_score_mean)
                evaluation_array_median.append(randomized_discrim_score_median)

            ground_truth_scaled = self.randomized_env.rescale(randomized_dimension, self.ground_truth_x)
            name = self.randomized_env.get_dimension_name(randomized_dimension)
            print('MeanDR', evaluation_array_mean[::10])
            print('MedianDR', evaluation_array_median[::10])

            plt.plot(ground_truth_scaled, evaluation_array_mean, c="green")
            plt.savefig('{}.png'.format(os.path.join(plot_path, 'mean-discrimrew-{}-{}'.format(name, timesteps))))
            plt.close()

            plt.plot(ground_truth_scaled, evaluation_array_median, c="green")
            plt.savefig('{}.png'.format(os.path.join(plot_path, 'med-discrimrew-{}-{}'.format(name, timesteps))))
            plt.close()

            np.savez('{}.npz'.format(os.path.join(log_path, 'discriminator_rewards-{}'.format(timesteps))), 
                discriminator_mean=evaluation_array_mean,
                discriminator_median=evaluation_array_median)
        

    def plot_value(self, simulator_agent, agent_policy, timesteps, plot_path, log_path):
        logger.debug('Generating ground truth...')

        default_values = [['default'] * simulator_agent.nparams]

        for randomized_dimension in range(simulator_agent.nparams):
            evaluation_array_mean = []
            evaluation_array_median = []
            for i, x in enumerate(self.ground_truth_x):
                if i % DISPLAY_FREQUENCY == 0:
                    logger.info("Dim: {}, Index: {}/{}".format(randomized_dimension, i, len(self.ground_truth_x)))

                values = default_values
                for index in range(self.neval_eps):
                    values[0][randomized_dimension] = x

                values = np.array(values)
                empirical_values = []
                for policy_idx in range(simulator_agent.nagents):
                    _, value = simulator_agent.svpg.select_action(policy_idx, values)
                    empirical_values.append(value.item())

                print(values, empirical_values)
                evaluation_array_mean.append(np.mean(empirical_values))
                evaluation_array_median.append(np.median(empirical_values))

            ground_truth_scaled = self.randomized_env.rescale(randomized_dimension, self.ground_truth_x)
            name = self.randomized_env.get_dimension_name(randomized_dimension)
            print('MeanVal', evaluation_array_mean[::10])
            print('MedianVal', evaluation_array_median[::10])

            plt.plot(ground_truth_scaled, evaluation_array_mean, c="green")
            plt.savefig('{}.png'.format(os.path.join(plot_path, 'mean-value-{}-{}'.format(name, timesteps))))
            plt.close()

            plt.plot(ground_truth_scaled, evaluation_array_median, c="green")
            plt.savefig('{}.png'.format(os.path.join(plot_path, 'med-value-{}-{}'.format(name, timesteps))))
            plt.close()

        logger.info('Ground truth generated.')
        return


    def plot_sampling_frequency(self, simulator_agent, agent_policy, timesteps, plot_path, log_path):
        for dimension in range(simulator_agent.nparams):
            plt.figure(figsize=(16, 9))
            dimension_name = self.randomized_env.get_dimension_name(dimension)
            sampled_regions = np.array(simulator_agent.sampled_regions[dimension]).flatten()

            np.savez('{}.npz'.format(os.path.join(log_path, 'sampled_regions-{}-{}'.format(dimension, timesteps))), 
                sampled_regions=sampled_regions)

            scaled_data = self.randomized_env.rescale(dimension, sampled_regions)
            plt.hist(scaled_data, bins=self.npoints)

            if self.config.get('hist_xlims') is not None:
                xlims = self.config.get('hist_xlims')
                plt.xlim(xlims[0], xlims[1])

            plt.ylim(0, self.config['hist_ylim_high'])
            plt.ylabel('Number of environment instances seen')
            plt.xlabel('Sampling frequency for {}'.format(dimension_name))
            plt.savefig('{}.png'.format(os.path.join(plot_path, '{}-{}'.format(dimension, timesteps))))
            plt.close()

    def plot_reward(self, simulator_agent, agent_policy, timesteps, plot_path, log_path, means=None, sigmas=None):
        """Plots estimated ground truth reward of baseline policy with stddev estimate
        """
        if means is None and self.evaluation_scores is None:
            self.generate_ground_truth(simulator_agent, agent_policy, timesteps)

        for dimension in range(simulator_agent.nparams):
            dimension_name = self.randomized_env.get_dimension_name(dimension)
            ground_truth_scaled = self.randomized_env.rescale(dimension, self.ground_truth_x)

            plt.figure(figsize=(16, 9))

            if means is None:
                mean = self.evaluation_scores[dimension, :, 0]
                sigma = self.evaluation_scores[dimension, :, 1]
            else:
                mean = means
                sigma = sigmas

            np.savez('{}.npz'.format(os.path.join(log_path, 'rewards-{}-{}'.format(dimension, timesteps))), 
                mean=mean, sigma=sigma)

            plt.title('Avg Agent Reward for {} (N={}) when varying {}'.format(agent_policy.agent_name, self.neval_eps, 
                dimension_name))
            plt.xlabel('Value of {}'.format(dimension_name))
            plt.ylabel(self.config['ylabel'])
            plt.ylim(self.config['ylim_low'], self.config['ylim_high'])
            plt.axhline(self.config['solved'], color='r', linestyle='--')

            plt.plot(ground_truth_scaled, mean, c="green")
            plt.fill_between(ground_truth_scaled, mean + sigma, mean - sigma, facecolor="green", alpha=0.15)
            # TODO: make the figure tight
            plt.savefig('{}.png'.format(os.path.join(plot_path, '{}-{}'.format(dimension, timesteps))))
            plt.close()