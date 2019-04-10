#!/usr/bin/env bash
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=2   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham
#SBATCH --mem=36000M        # memory per node
#SBATCH --time=1-12:00      # time (DD-HH:MM)
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --mail-user=noreply@domain.com
#SBATCH --mail-type=ALL

echo "Configuring Slurm Job Environment - $SLURM_JOB_ID"
source activate rl-local
cd ~/coding/diffsim

export PYTHONPATH="${PYTHONPATH}:`pwd`/coding"
# python -m experiments.domainrand.experiment_driver lunar --experiment-name=gail-baseline  --initial-svpg-steps=1e6 --freeze-svpg --prerecorded-trajectories --expert-trajectories-file="reference_trajectories_trained_16" --continuous-svpg --randomized-env-id="LunarLanderRandomized-v0" --experiment-prefix="gailbaseline16" --seed=1 &
# python -m experiments.domainrand.experiment_driver lunar --experiment-name=gail-baseline  --initial-svpg-steps=1e6 --freeze-svpg --prerecorded-trajectories --expert-trajectories-file="reference_trajectories_trained_16" --continuous-svpg --randomized-env-id="LunarLanderRandomized-v0" --experiment-prefix="gailbaseline16" --seed=2   

python -m experiments.domainrand.experiment_driver lunar --experiment-name=adaptive-randomization --particle-path="saved-models/particles/" --reward-scale=-1.0 --kld-coefficient=0.01 --prerecorded-trajectories --expert-trajectories-file="reference_trajectories_trained_16" --continuous-svpg --randomized-env-id="LunarLanderRandomized-v0" --experiment-prefix="adrplus16" --seed=2 &

python -m experiments.domainrand.experiment_driver lunar --experiment-name=adaptive-randomization --particle-path="saved-models/particles/" --reward-scale=-1.0 --kld-coefficient=0.01 --prerecorded-trajectories --expert-trajectories-file="reference_trajectories_trained_16" --continuous-svpg --randomized-env-id="LunarLanderRandomized-v0" --experiment-prefix="adrplus16" --seed=3
