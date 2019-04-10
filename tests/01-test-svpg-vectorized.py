import numpy as np
from common.svpg.svpg import SVPG
from common.envs.randomized_vecenv import make_vec_envs

def _create_envs(seed, nagents, reference_env_id='LunarLanderDefault-v0', 
    randomized_env_id='LunarLanderRandomized-v0'):
    
    reference_env = make_vec_envs(reference_env_id, seed, nagents)
    randomized_env = make_vec_envs(randomized_env_id, seed, nagents)

    return reference_env, randomized_env

nagents = 3
svpg = SVPG(nagents)
reference_env, randomized_env = _create_envs(seed=123, nagents=nagents)

simulation_settings = svpg.step()
assert (nagents, svpg.svpg_rollout_length, svpg.nparams) == simulation_settings.shape

simulation_settings = np.transpose(simulation_settings, (1, 0, 2))

for t in range(svpg.svpg_rollout_length):
    print("Current Timestep: {}".format(t))
    print([simulation_settings[t]])
    randomized_env.randomize(randomized_values=simulation_settings[t])
    print(randomized_env.get_current_params())


