import gym
import numpy as np
import logging

import torch
from common.envs.randomized_vecenv import make_vec_envs

from common.discriminator.discriminator_rewarder import DiscriminatorRewarder
from common.svpg.svpg import SVPG

from common.utils.rollout_evaluation import evaluate_policy, check_solved
from common.agents.ddpg.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class SVPGSimulatorAgent(object):
    """Simulation object which creates randomized environments based on specified params, 
    handles SVPG-based policy search to create envs, 
    and evaluates controller policies in those environments
    """

    def __init__(self,
                 reference_env_id,
                 randomized_env_id,
                 randomized_eval_env_id,
                 agent_name,
                 nagents,
                 nparams,
                 temperature,
                 svpg_rollout_length,
                 svpg_horizon,
                 max_step_length,
                 reward_scale,
                 initial_svpg_steps,
                 max_env_timesteps,
                 episodes_per_instance,
                 discrete_svpg,
                 load_discriminator,
                 freeze_discriminator,
                 freeze_agent,
                 seed,
                 train_svpg=True,
                 particle_path="",
                 discriminator_batchsz=320,
                 randomized_eval_episodes=3,
                 ):

        # TODO: Weird bug
        assert nagents > 2

        self.reference_env_id = reference_env_id
        self.randomized_env_id = randomized_env_id
        self.randomized_eval_env_id = randomized_eval_env_id
        self.agent_name = agent_name

        self.log_distances = reference_env_id.find('Lunar') == -1

        self.randomized_eval_episodes = randomized_eval_episodes

        # Vectorized environments - step with nagents in parallel
        self.reference_env = make_vec_envs(reference_env_id, seed, nagents)
        self.randomized_env = make_vec_envs(randomized_env_id, seed, nagents)

        self.state_dim = self.reference_env.observation_space.shape[0]
        self.action_dim = self.reference_env.action_space.shape[0]
    
        if reference_env_id.find('Pusher') != -1:
            self.hard_env = make_vec_envs('Pusher3DOFHard-v0', seed, nagents)
        elif reference_env_id.find('Lunar') != -1:
            self.hard_env = make_vec_envs('LunarLander10-v0', seed, nagents)
        elif reference_env_id.find('Backlash') != -1:
            self.hard_env = make_vec_envs('ErgoReacherRandomizedBacklashHard-v0', seed, nagents)
        else:
            self.hard_env = make_vec_envs('ErgoReacher4DOFRandomizedHard-v0', seed, nagents)

        self.sampled_regions = [[] for _ in range(nparams)]

        self.nagents = nagents
        self.nparams = self.randomized_env.randomization_space.shape[0]
        assert self.nparams == nparams, "Double check number of parameters: Args: {}, Env: {}".format(
            nparams, self.nparams)

        self.svpg_horizon = svpg_horizon
        self.initial_svpg_steps = initial_svpg_steps
        self.max_env_timesteps = max_env_timesteps
        self.episodes_per_instance = episodes_per_instance
        self.discrete_svpg = discrete_svpg

        self.freeze_discriminator = freeze_discriminator
        self.freeze_agent = freeze_agent

        self.train_svpg = train_svpg
        
        self.agent_eval_frequency = max_env_timesteps * nagents 

        self.seed = seed
        self.svpg_timesteps = 0
        self.agent_timesteps = 0
        self.agent_timesteps_since_eval = 0

        self.discriminator_rewarder = DiscriminatorRewarder(reference_env=self.reference_env,
                                                            randomized_env_id=randomized_env_id,
                                                            discriminator_batchsz=discriminator_batchsz,
                                                            reward_scale=reward_scale,
                                                            load_discriminator=load_discriminator,
                                                            )

        if not self.freeze_agent:
            self.replay_buffer = ReplayBuffer()
        else:
            self.replay_buffer = None

        self.svpg = SVPG(nagents=nagents,
                         nparams=self.nparams,
                         max_step_length=max_step_length,
                         svpg_rollout_length=svpg_rollout_length,
                         svpg_horizon=svpg_horizon,
                         temperature=temperature,
                         discrete=self.discrete_svpg,
                         kld_coefficient=0.0)

        if particle_path != "":
            logger.info("Loading particles from: {}".format(particle_path))
            self.svpg.load(directory=particle_path)

        self.simulation_instances_full_horizon = np.ones((self.nagents,
                                                          self.svpg_horizon,
                                                          self.svpg.svpg_rollout_length,
                                                          self.svpg.nparams)) * -1

    def select_action(self, agent_policy):
        """Select an action based on SVPG policy, where an action is the delta in each dimension.
        Update the counts and statistics after training agent,
        rolling out policies, and calculating simulator reward.
        """
        if self.svpg_timesteps >= self.initial_svpg_steps:
            # Get sim instances from SVPG policy
            simulation_instances = self.svpg.step()

            index = self.svpg_timesteps % self.svpg_horizon
            self.simulation_instances_full_horizon[:, index, :, :] = simulation_instances

        else:
            # Creates completely randomized environment
            simulation_instances = np.ones((self.nagents,
                                            self.svpg.svpg_rollout_length,
                                            self.svpg.nparams)) * -1

        assert (self.nagents, self.svpg.svpg_rollout_length, self.svpg.nparams) == simulation_instances.shape

        # Create placeholders for trajectories
        randomized_trajectories = [[] for _ in range(self.nagents)]
        reference_trajectories = [[] for _ in range(self.nagents)]

        # Create placeholder for rewards
        rewards = np.zeros(simulation_instances.shape[:2])
        
        # Discriminator debugging
        randomized_discrim_score_mean = 0
        reference_discrim_score_mean  = 0
        randomized_discrim_score_median = 0
        reference_discrim_score_median  = 0

        # Reshape to work with vectorized environments
        simulation_instances = np.transpose(simulation_instances, (1, 0, 2))

        # Create environment instances with vectorized env, and rollout agent_policy in both
        for t in range(self.svpg.svpg_rollout_length):
            agent_timesteps_current_iteration = 0
            logging.info('Iteration t: {}/{}'.format(t, self.svpg.svpg_rollout_length))  

            reference_trajectory = self.rollout_agent(agent_policy)

            self.randomized_env.randomize(randomized_values=simulation_instances[t])
            randomized_trajectory = self.rollout_agent(agent_policy, reference=False)

            for i in range(self.nagents):
                agent_timesteps_current_iteration += len(randomized_trajectory[i])

                reference_trajectories[i].append(reference_trajectory[i])
                randomized_trajectories[i].append(randomized_trajectory[i])
                
                self.agent_timesteps += len(randomized_trajectory[i])
                self.agent_timesteps_since_eval += len(randomized_trajectory[i])

                simulator_reward = self.discriminator_rewarder.calculate_rewards(randomized_trajectories[i][t])
                rewards[i][t] = simulator_reward

                logger.info('Setting: {}, Score: {}'.format(simulation_instances[t][i], simulator_reward))

            if not self.freeze_discriminator:
                # flatten and combine all randomized and reference trajectories for discriminator
                flattened_randomized = [randomized_trajectories[i][t] for i in range(self.nagents)]
                flattened_randomized = np.concatenate(flattened_randomized)

                flattened_reference = [reference_trajectories[i][t] for i in range(self.nagents)]
                flattened_reference = np.concatenate(flattened_reference)

                randomized_discrim_score_mean, randomized_discrim_score_median, randomized_discrim_score_sum = \
                    self.discriminator_rewarder.get_score(flattened_randomized)
                reference_discrim_score_mean, reference_discrim_score_median, reference_discrim_score_sum = \
                    self.discriminator_rewarder.get_score(flattened_reference)

                # Train discriminator based on state action pairs for agent env. steps
                # TODO: Train more?
                self.discriminator_rewarder.train_discriminator(flattened_reference, flattened_randomized,
                                                                iterations=agent_timesteps_current_iteration)

                randomized_discrim_score_mean, randomized_discrim_score_median, randomized_discrim_score_sum = \
                    self.discriminator_rewarder.get_score(flattened_randomized)
                reference_discrim_score_mean, reference_discrim_score_median, reference_discrim_score_sum = \
                    self.discriminator_rewarder.get_score(flattened_reference)

        # Calculate discriminator based reward, pass it back to SVPG policy
        if self.svpg_timesteps >= self.initial_svpg_steps:
            if self.train_svpg:
                self.svpg.train(rewards)

            for dimension in range(self.nparams):
                self.sampled_regions[dimension] = np.concatenate([
                    self.sampled_regions[dimension], simulation_instances[:, :, dimension].flatten()
                ])           

        solved_reference = info = None
        if self.agent_timesteps_since_eval > self.agent_eval_frequency:
            self.agent_timesteps_since_eval %= self.agent_eval_frequency
            logger.info("Evaluating for {} episodes afer timesteps: {} (SVPG), {} (Agent)".format(
                self.randomized_eval_episodes * self.nagents, self.svpg_timesteps, self.agent_timesteps))

            agent_reference_eval_rewards = []
            agent_randomized_eval_rewards = []

            final_dist_ref = []
            final_dist_rand = []

            for _ in range(self.randomized_eval_episodes):
                rewards_ref, dist_ref = evaluate_policy(nagents=self.nagents,
                                                        env=self.reference_env,
                                                        agent_policy=agent_policy,
                                                        replay_buffer=None,
                                                        eval_episodes=1,
                                                        max_steps=self.max_env_timesteps,
                                                        return_rewards=True,
                                                        add_noise=False,
                                                        log_distances=self.log_distances)

                full_random_settings = np.ones((self.nagents, self.nparams)) * -1
                self.randomized_env.randomize(randomized_values=full_random_settings)

                rewards_rand, dist_rand = evaluate_policy(nagents=self.nagents,
                                                          env=self.randomized_env,
                                                          agent_policy=agent_policy,
                                                          replay_buffer=None,
                                                          eval_episodes=1,
                                                          max_steps=self.max_env_timesteps,
                                                          return_rewards=True,
                                                          add_noise=False,
                                                          log_distances=self.log_distances)

                agent_reference_eval_rewards += list(rewards_ref)
                agent_randomized_eval_rewards += list(rewards_rand)
                final_dist_ref += [dist_ref]
                final_dist_rand += [dist_rand]

            evaluation_criteria_reference = agent_reference_eval_rewards
            evaluation_criteria_randomized = agent_randomized_eval_rewards

            if self.log_distances:
                evaluation_criteria_reference = final_dist_ref
                evaluation_criteria_randomized = final_dist_rand

            solved_reference = check_solved(self.reference_env_id, evaluation_criteria_reference)
            solved_randomized = check_solved(self.randomized_eval_env_id, evaluation_criteria_randomized)

            info = {
                'solved': str(solved_reference),
                'solved_randomized': str(solved_randomized),
                'svpg_steps': self.svpg_timesteps,
                'agent_timesteps': self.agent_timesteps,
                'final_dist_ref_mean': np.mean(final_dist_ref),
                'final_dist_ref_std': np.std(final_dist_ref),
                'final_dist_ref_median': np.median(final_dist_ref),
                'final_dist_rand_mean': np.mean(final_dist_rand),
                'final_dist_rand_std': np.std(final_dist_rand),
                'final_dist_rand_median': np.median(final_dist_rand),
                'agent_reference_eval_rewards_mean': np.mean(agent_reference_eval_rewards),
                'agent_reference_eval_rewards_std': np.std(agent_reference_eval_rewards),
                'agent_reference_eval_rewards_median': np.median(agent_reference_eval_rewards),
                'agent_reference_eval_rewards_min': np.min(agent_reference_eval_rewards),
                'agent_reference_eval_rewards_max': np.max(agent_reference_eval_rewards),
                'agent_randomized_eval_rewards_mean': np.mean(agent_randomized_eval_rewards),
                'agent_randomized_eval_rewards_std': np.std(agent_randomized_eval_rewards),
                'agent_randomized_eval_rewards_median': np.median(agent_randomized_eval_rewards),
                'agent_randomized_eval_rewards_min': np.min(agent_randomized_eval_rewards),
                'agent_randomized_eval_rewards_max': np.max(agent_randomized_eval_rewards),
                'randomized_discrim_score_mean': str(randomized_discrim_score_mean),
                'reference_discrim_score_mean': str(reference_discrim_score_mean),
                'randomized_discrim_score_median': str(randomized_discrim_score_median),
                'reference_discrim_score_median': str(reference_discrim_score_median),

            }

            agent_hard_eval_rewards, final_dist_hard = evaluate_policy(nagents=self.nagents,
                                                                       env=self.hard_env,
                                                                       agent_policy=agent_policy,
                                                                       replay_buffer=None,
                                                                       eval_episodes=1,
                                                                       max_steps=self.max_env_timesteps,
                                                                       return_rewards=True,
                                                                       add_noise=False,
                                                                       log_distances=self.log_distances)
            info_hard = {
                'final_dist_hard_mean': np.mean(final_dist_hard),
                'final_dist_hard_std': np.std(final_dist_hard),
                'final_dist_hard_median': np.median(final_dist_hard),
                'agent_hard_eval_rewards_median': np.median(agent_hard_eval_rewards),
                'agent_hard_eval_rewards_mean': np.mean(agent_hard_eval_rewards),
                'agent_hard_eval_rewards_std': np.std(agent_hard_eval_rewards),
            }

            info.update(info_hard)

        self.svpg_timesteps += 1
        return solved_reference, info

    def rollout_agent(self, agent_policy, reference=True, eval_episodes=None):
        """Rolls out agent_policy in the specified environment
        """
        if reference:
            if eval_episodes is None:
                eval_episodes = self.episodes_per_instance
            trajectory = evaluate_policy(nagents=self.nagents,
                                         env=self.reference_env,
                                         agent_policy=agent_policy,
                                         replay_buffer=None,
                                         eval_episodes=eval_episodes,
                                         max_steps=self.max_env_timesteps,
                                         freeze_agent=True,
                                         add_noise=False,
                                         log_distances=self.log_distances)
        else:
            trajectory = evaluate_policy(nagents=self.nagents,
                                         env=self.randomized_env,
                                         agent_policy=agent_policy,
                                         replay_buffer=self.replay_buffer,
                                         eval_episodes=self.episodes_per_instance,
                                         max_steps=self.max_env_timesteps,
                                         freeze_agent=self.freeze_agent,
                                         add_noise=True,
                                         log_distances=self.log_distances)

        return trajectory

    def sample_trajectories(self, batch_size):
        indices = np.random.randint(0, len(self.extracted_trajectories['states']), batch_size)

        states = self.extracted_trajectories['states']
        actions = self.extracted_trajectories['actions']
        next_states = self.extracted_trajectories['next_states']

        trajectories = []
        for i in indices:
            trajectories.append(np.concatenate(
                [
                    np.array(states[i]),
                    np.array(actions[i]),
                    np.array(next_states[i])
                ], axis=-1))
        return trajectories

