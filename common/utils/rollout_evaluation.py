import numpy as np

LUNAR_LANDER_SOLVED_SCORE = 200.0
ERGO_SOLVED_DISTANCE = 0.025
PUSHER_SOLVED_DISTANCE = 0.25  # Radius=0.17


def evaluate_policy(nagents, env, agent_policy, replay_buffer, eval_episodes, max_steps, freeze_agent=True,
                    return_rewards=False, add_noise=False, log_distances=True, 
                    gail_rewarder=None, noise_scale=0.1, min_buffer_len=1000):
    """Evaluates a given policy in a particular environment, 
    returns an array of rewards received from the evaluation step.
    """

    states = [[] for _ in range(nagents)]
    actions = [[] for _ in range(nagents)]
    next_states = [[] for _ in range(nagents)]
    rewards = [[] for _ in range(nagents)]
    ep_rewards = []
    final_dists = []

    for ep in range(eval_episodes):
        agent_total_rewards = np.zeros(nagents)
        state = env.reset()

        done = [False] * nagents
        add_to_buffer = [True] * nagents
        steps = 0
        training_iters = 0

        while not all(done) and steps <= max_steps:
            action = agent_policy.select_action(np.array(state))

            if add_noise:
                action = action + np.random.normal(0, noise_scale, size=action.shape)
                action = action.clip(-1, 1)

            next_state, reward, done, info = env.step(action)
            if gail_rewarder is not None:
                reward = gail_rewarder.get_reward(np.concatenate([state, action], axis=-1))

            for i, st in enumerate(state):
                if add_to_buffer[i]:
                    states[i].append(st)
                    actions[i].append(action[i])
                    next_states[i].append(next_state[i])
                    rewards[i].append(reward[i])
                    agent_total_rewards[i] += reward[i]
                    training_iters += 1

                    if replay_buffer is not None:
                        done_bool = 0 if steps + 1 == max_steps else float(done[i])
                        replay_buffer.add((state[i], next_state[i], action[i], reward[i], done_bool))

                if done[i]:
                    # Avoid duplicates
                    add_to_buffer[i] = False

                    if log_distances:
                        final_dists.append(info[i]['goal_dist'])

            state = next_state
            steps += 1

        # Train for total number of env iterations
        if not freeze_agent and len(replay_buffer.storage) > min_buffer_len:
            agent_policy.train(replay_buffer=replay_buffer, iterations=training_iters)

        ep_rewards.append(agent_total_rewards)

    if return_rewards:
        return np.array(ep_rewards).flatten(), np.array(final_dists).flatten()

    trajectories = []
    for i in range(nagents):
        trajectories.append(np.concatenate(
            [
                np.array(states[i]),
                np.array(actions[i]),
                np.array(next_states[i])
            ], axis=-1))

    return trajectories


def check_solved(env_name, criteria):
    if env_name.find('Lunar') != -1:
        return np.median(criteria) > LUNAR_LANDER_SOLVED_SCORE
    elif env_name.find('Ergo') != -1:
        return np.median(criteria) < ERGO_SOLVED_DISTANCE
    else:
        return np.median(criteria) < PUSHER_SOLVED_DISTANCE


def check_new_best(env_name, new, current):
    if env_name.find('Lunar') != -1:
        return new > current
    else:
        return new < current
