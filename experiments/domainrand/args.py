import argparse
import logging

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='Domain Randomization Driver')

    subparsers = parser.add_subparsers(help='sub-command help', dest='subparser_name')

    lunar_subparser = subparsers.add_parser('lunar', help='lunar lander subparser')
    pusher_subparser = subparsers.add_parser('pusher', help='puhser-3dof subparser')
    ergo_subparser = subparsers.add_parser('ergo', help='ergoreacher subparser')
    backlash_subparser = subparsers.add_parser('backlash', help='ergoreacher with backlash subparser')
    ergosix_subparser = subparsers.add_parser('ergosix', help='ergoreacher 6dpf subparser')

    lunar_subparser.add_argument("--randomized-env-id", default="LunarLanderDefault-v0", 
        type=str, help="Name of the reference environment")
    lunar_subparser.add_argument("--reference-env-id", default="LunarLanderDefault-v0", 
        type=str, help="Name of the randomized environment")
    lunar_subparser.add_argument("--randomized-eval-env-id", default="LunarLanderRandomized-v0", 
        type=str, help="Name of the randomized environment")
    lunar_subparser.add_argument("--nparams", default=1, type=int, help="Number of randomization parameters")
    lunar_subparser.add_argument("--eval-randomization-discretization", default=50, type=int, help="number of eval points")
    lunar_subparser.add_argument("--max-env-timesteps", default=1000, type=int, 
            help="environment timeout")
    lunar_subparser.add_argument("--plot-frequency", default=5, type=int, help="how often to plot / log")
    lunar_subparser.add_argument("--nagents", default=10, type=int, 
            help="Number of SVPG particle")

    pusher_subparser.add_argument("--randomized-env-id", default="Pusher3DOFDefault-v0",
        type=str, help="Name of the reference environment")
    pusher_subparser.add_argument("--reference-env-id", default="Pusher3DOFDefault-v0", 
        type=str, help="Name of the randomized environment")
    pusher_subparser.add_argument("--randomized-eval-env-id", default="Pusher3DOFRandomized-v0", 
        type=str, help="Name of the randomized environment")
    pusher_subparser.add_argument("--nparams", default=2, type=int, help="Number of randomization parameters")
    pusher_subparser.add_argument("--eval-randomization-discretization", default=20, type=int, help="number of eval points")
    pusher_subparser.add_argument("--max-env-timesteps", default=100, type=int, 
            help="environment timeout")
    pusher_subparser.add_argument("--plot-frequency", default=5, type=int, help="how often to plot / log")
    pusher_subparser.add_argument("--nagents", default=10, type=int, 
            help="Number of SVPG particle")

    ergo_subparser.add_argument("--randomized-env-id", default="ErgoReacher4DOFDefault-v0", 
        type=str, help="Name of the reference environment")
    ergo_subparser.add_argument("--reference-env-id", default="ErgoReacher4DOFDefault-v0", 
        type=str, help="Name of the randomized environment")
    ergo_subparser.add_argument("--randomized-eval-env-id", default="ErgoReacher4DOFRandomizedEasy-v0", 
        type=str, help="Name of the randomized environment")
    ergo_subparser.add_argument("--nparams", default=8, type=int, help="Number of randomization parameters")
    ergo_subparser.add_argument("--eval-randomization-discretization", default=5, type=int, help="number of eval points")
    ergo_subparser.add_argument("--max-env-timesteps", default=100, type=int, 
            help="environment timeout")
    ergo_subparser.add_argument("--plot-frequency", default=50, type=int, help="how often to plot / log")
    ergo_subparser.add_argument("--nagents", default=10, type=int, 
            help="Number of SVPG particle")

    backlash_subparser.add_argument("--randomized-env-id", default="ErgoReacherRandomizedBacklashEasy-v0", 
        type=str, help="Name of the reference environment")
    backlash_subparser.add_argument("--reference-env-id", default="ErgoReacher-DualGoal-Easy-Default-Headless-v0", 
        type=str, help="Name of the randomized environment")
    backlash_subparser.add_argument("--randomized-eval-env-id", default="ErgoReacherRandomizedBacklashEasy-v0", 
        type=str, help="Name of the randomized environment")
    backlash_subparser.add_argument("--nparams", default=8, type=int, help="Number of randomization parameters")
    backlash_subparser.add_argument("--eval-randomization-discretization", default=20, type=int, help="number of eval points")
    backlash_subparser.add_argument("--max-env-timesteps", default=200, type=int, 
            help="environment timeout")
    backlash_subparser.add_argument("--plot-frequency", default=50, type=int, help="how often to plot / log")
    backlash_subparser.add_argument("--nagents", default=10, type=int, 
            help="Number of SVPG particle")

    ergosix_subparser.add_argument("--randomized-env-id", default="ErgoReacher-6Dof-Default-Headless-v0", 
        type=str, help="Name of the reference environment")
    ergosix_subparser.add_argument("--reference-env-id", default="ErgoReacher-6Dof-Default-Headless-v0", 
        type=str, help="Name of the randomized environment")
    ergosix_subparser.add_argument("--randomized-eval-env-id", default="ErgoReacher-6Dof-Randomized-Headless-v0", 
        type=str, help="Name of the randomized environment")
    ergosix_subparser.add_argument("--nparams", default=12, type=int, help="Number of randomization parameters")
    ergosix_subparser.add_argument("--eval-randomization-discretization", default=20, type=int, help="number of eval points")
    ergosix_subparser.add_argument("--max-env-timesteps", default=100, type=int, 
            help="environment timeout")
    ergosix_subparser.add_argument("--plot-frequency", default=5, type=int, help="how often to plot / log")
    ergosix_subparser.add_argument("--nagents", default=10, type=int, 
            help="Number of SVPG particle")

    for subparser in [lunar_subparser, pusher_subparser, ergo_subparser, backlash_subparser, ergosix_subparser]:
        subparser.add_argument("--experiment-name", type=str, 
            choices=['bootstrapping', 'unfreeze-policy'])
        subparser.add_argument("--experiment-prefix", default="experiment", type=str, help="Any custom string to attach")
        subparser.add_argument("--agent-name", default="baseline", type=str, 
            help="Which Agent to benchmark")
        subparser.add_argument("--temperature", default=10.0, type=float, 
            help="SVPG temperature")
        subparser.add_argument("--svpg-rollout-length", default=5, type=int, 
            help="length of one svpg particle rollout")
        subparser.add_argument("--svpg-horizon", default=25, type=int, 
            help="how often to fully reset svpg particles")

        subparser.add_argument("--max-step-length", default=0.05, 
            type=float, help="step length / delta in parameters; If discrete, this is fixed, If continuous, this is max.")

        subparser.add_argument("--reward-scale", default=1.0, type=float, 
            help="reward multipler for discriminator")
        subparser.add_argument("--initial-svpg-steps", default=0, type=float, 
            help="number of svpg steps to take before updates")
        subparser.add_argument("--max-agent-timesteps", default=1e6, type=float, 
            help="max iterations, counted in terms of AGENT env steps")
        subparser.add_argument("--episodes-per-instance", default=1, type=int, 
            help="number of episodes to rollout the agent for per sim instance")

        subparser.add_argument("--kld-coefficient", default=0.00, type=float, help="kld coefficient for particles")
        subparser.add_argument("--discrete-svpg", action="store_true", help="discrete SVPG")
        subparser.add_argument("--continuous-svpg", action="store_true", help="continuous SVPG")
        subparser.add_argument("--save-particles", action="store_true", help="store the particle policies")
        subparser.add_argument("--particle-path", default="", type=str, help="where to load particles from")
        subparser.add_argument("--freeze-svpg", action="store_true", help="Freeze SVPG or not")

        subparser.add_argument("--pretrain-discriminator", help="pretrain discriminator or not")
        subparser.add_argument("--load-discriminator", action="store_true", help="load discriminator or not")
        subparser.add_argument("--load-agent", action="store_true", help="load an agent or not")
        subparser.add_argument("--freeze-discriminator", action="store_true", help="freeze discriminator (no training)")
        subparser.add_argument("--freeze-agent", action="store_true", help="freeze agent (no training)")

        subparser.add_argument("--seed", default=123, type=int)
        subparser.add_argument("--use-bootstrapping-results", action="store_true", help="where to look when running batch-reward-anaylsis")

    return parser.parse_args()

def check_args(args):
    experiment_name = args.experiment_name

    assert args.nagents > 2, "TODO: Weird bug"
    assert args.discrete_svpg or args.continuous_svpg and not (args.discrete_svpg and args.continuous_svpg), "Specify continuous OR discrete"

    if experiment_name == 'batch-reward-anaylsis':
        assert args.load_agent
        assert args.episodes_per_instance >= 5, "Need to run atleast 5+ runs when doing reward plots"
        return
    elif experiment_name.find('reward') != -1:
        assert args.episodes_per_instance > 1, "Probably want more than just one eval_episode for evaluation?"
    elif experiment_name == 'bootstrapping':
        assert args.load_discriminator, "Need to load discriminator"
        assert args.freeze_agent == False, "Need to unfreeze agent"

    assert args.svpg_rollout_length < 25, "Rollout length likely too long - SVPG will likely need more frequent feedback"
    assert args.svpg_horizon > 10, "Horizon likely too short for consistency - might reset SVPG to random positions too frequently"
    assert args.episodes_per_instance > 0, "Must provide episodes_per_instance"

    if args.pretrain_discriminator:
        assert args.load_discriminator == True, "If pretraining, you should also load"

    if args.discrete_svpg:
        assert args.max_step_length < 0.1, "Step length for discrete_svpg too large"

    if args.initial_svpg_steps >= args.max_agent_timesteps:
        logger.warning("YOU WILL NOT TRAIN THE SVPG AGENT")

    if not args.freeze_discriminator and not args.load_discriminator:
        logger.warning("YOU ARE TRAINING THE DISCRIMINATOR FROM SCRATCH")

    if not args.load_agent:
        logger.warning("YOU ARE TRAINING THE AGENT POLICY FROM SCRATCH")

    if args.randomized_env_id == args.reference_env_id:
        logger.warning("REFERENCE AND RANDOMIZED IDs ARE SAME")