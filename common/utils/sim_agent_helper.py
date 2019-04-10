from common.agents.svpg_simulator_agent import SVPGSimulatorAgent

def generate_simulator_agent(args):
    return SVPGSimulatorAgent(
        reference_env_id=args.reference_env_id, 
        randomized_env_id=args.randomized_env_id,
        randomized_eval_env_id=args.randomized_eval_env_id,
        agent_name=args.agent_name,
        nagents=args.nagents,
        nparams=args.nparams,
        temperature=args.temperature, 
        svpg_rollout_length=args.svpg_rollout_length,
        svpg_horizon=args.svpg_horizon,
        max_step_length=args.max_step_length,
        reward_scale=args.reward_scale,
        initial_svpg_steps=args.initial_svpg_steps,
        max_env_timesteps=args.max_env_timesteps,
        episodes_per_instance=args.episodes_per_instance,
        discrete_svpg=args.discrete_svpg,
        load_discriminator=args.load_discriminator, 
        freeze_discriminator=args.freeze_discriminator, 
        freeze_agent=args.freeze_agent, 
        seed=args.seed,
        particle_path=args.particle_path,
    )