from common.envs import LunarLanderRandomized
from common.envs.randomized_vecenv import make_vec_envs

def _create_envs(seed, reference_env_id='LunarLanderDefault-v0', 
    randomized_env_id='LunarLanderRandomized-v0'):
    
    reference_env = make_vec_envs(reference_env_id, seed, num_processes=3)
    randomized_env = make_vec_envs(randomized_env_id, seed, num_processes=3)

    return reference_env, randomized_env


reference_env, randomized_env = _create_envs(1)
obs = randomized_env.reset()
print(randomized_env.get_current_params())

for _ in range(3):
    randomized_env.randomize(randomized_values=[['random'], ['random'], ['random']])
    print(randomized_env.get_current_params())

print("2D Lunar Lander Randomization")
reference_env, randomized_env = _create_envs(1, randomized_env_id='LunarLanderRandomized2D-v0')
obs = randomized_env.reset()
print(randomized_env.get_current_params())

for _ in range(3):
    randomized_env.randomize(randomized_values=[['random', 'random'], ['random', 'random'], ['random', 'random']])
    print(randomized_env.get_current_params())

print("2D - Setting One Value")
randomized_env.randomize(randomized_values=[[0.0, 'random'], [0.5, 'random'], [1.0, 'random']])
print(randomized_env.get_current_params())

print("2D - Setting Both Values")
randomized_env.randomize(randomized_values=[[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
print(randomized_env.get_current_params())