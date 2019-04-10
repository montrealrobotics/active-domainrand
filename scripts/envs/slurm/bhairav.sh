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
export LD_LIBRARY_PATH=/Tmp/glx:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/mehtabha/.mujoco/mjpro150/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so.1.10
Xvfb :$SLURM_JOB_ID -screen 0 84x84x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:$SLURM_JOB_ID