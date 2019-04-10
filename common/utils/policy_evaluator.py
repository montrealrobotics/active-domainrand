import json

import numpy as np
import gym


class PolicyEvaluator:
    def __init__(self, env_id, seed, policy, eval_file_path):
        self.env = gym.make(env_id)
        self.env.seed(seed)
        self.policy = policy
        self.eval_file = open(eval_file_path, mode='w')

    def evaluate(self, iteration, episodes=10, debug=True):

        episodes_stats = []
        cumulative_reward = 0.0

        for _ in range(episodes):
            obs = self.env.reset()

            steps = 0
            total_reward = 0.0
            done = False

            while not done:
                action = self.policy.select_action(np.array(obs))
                obs, reward, done, _ = self.env.step(action)

                # stats
                steps += 1
                total_reward += reward
                cumulative_reward += reward

                if debug:
                    self.env.render()

            episodes_stats.append({
                'steps': steps,
                'reward': total_reward
            })

        json.dump({
            'iteration': iteration,
            'reward': cumulative_reward,
            'episodes': episodes,
            'stats': episodes_stats
        }, self.eval_file, indent=2, sort_keys=True)

        self.eval_file.flush()

        self.env.close()

        return cumulative_reward / episodes

    def close(self):
        self.eval_file.close()

