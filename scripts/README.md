# TL;DR Scripting

There are two scripts that are helpful to execute the experiments
as decribed on [README](../README.md):

1. `scripts/multiseed.sh` for executing an experiment with multiple seeds.
2. `scripts/with_seed.sh` for executing one experiment with one seed.

The abbreviated names for each experiment are defined on the `experiments.sh`
script located on this folder.

We currently support the following experiments:

**Baselines**
- `baseline_pure`
- `baseline_fulldr`

**Unfreeze Policy**
- `unfreeze_policy_pretrained`
- `unfreeze_policy_scratch`

**Unfreeze Discriminator**
- `unfreeze_discriminator_pretrained`
- `unfreeze_discriminator_scratch`

**SVPG 2D Full**
- `svpg2d_ours`
- `svpg2d_fulldr`

#### Examples

Use `multiseed.sh` to execute an experiment with multiple, consecutive seeds.
The syntax for `multiseed.sh` is as follow:

```bash
scripts/multiseed.sh [environment] [user] [experiment] [starting seed] [number of seeds]
```

For instance:

```bash
scripts/multiseed.sh bluewire manfred svpg2d_fulldr 0 5
```

executes 5 seeds `[0, 1, 2, 3, 4]` of the `svpg2d_fulldr` experiment
using `manfred.sh` configuration for the `bluewire` environment.

Alternatively, you can use `with_seed.sh` to run an experiment with only 1 seed.
The syntax for `with_seed.sh` is as follows:

```bash
scripts/with_seed.sh [environment] [user] [experiment] [seed]
```

Then,

```bash
scripts/with_seed.sh slurm bhairav svpg2d_ours 1234
```

executes `svpg2d_ours` experiment with `seed=1234` using Bhairav's slurm configuration.

### NOTE
 **ALWAYS!!!!** execute the scripts from the repo main folder.

# Custom Configurations

This section explains how (and why you need) to create per user/per
environment configuration to run the experiments scripts.

## Environments

As we currently have multiple places where we can run our experiments
(slurm, bluewire, uberduck, etc), and we may be adding more soon (e.g.,
AWS) the particularities of each environment are quite different.
Therefore, we need to isolate them from our main scripting.

There are currently 3 folders to group each
user's particular settings for any of those environments.

```
scripts
    - bluewire    (Manfred's PC at home)
    - slurm       (Mila's cluster)
    - uberduck    (Lab's computer)
```

## Users

Create a `[env]\[user].sh` file to configure your particular setting in the `[env]` environment.

For instance, this is Bhairav's configuration for MILA's Slurm at `slurm\bhairav.sh`

```bash
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
source activate ml
cd ~/coding/diffsim

export PYTHONPATH="${PYTHONPATH}:`pwd`/coding"
export LD_LIBRARY_PATH=/Tmp/glx:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/mehtabha/.mujoco/mjpro150/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so.1.10
Xvfb :$SLURM_JOB_ID -screen 0 84x84x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:$SLURM_JOB_ID

```

Hence, if e.g., Bhairav wants to run 5 seeds (starting at 0) on slurm of the `svpg2d_ours` on slurm,
he would have to execute, from the main `diffsim` folder, the following command:

```
scripts/multiseed.sh slurm bhairav svpg2d_ours 0 5
```