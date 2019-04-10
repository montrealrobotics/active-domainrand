#!/usr/bin/env bash

# =============== Pusher 3 DoF ============

pusher_3dof_baseline() {
    python scripts/launch.py --main=unfreeze_policy --setting=pusher --prefix=true-baseline --seeds 5 \
        with --freeze-discriminator \
             --agent-name=baseline \
             --initial-svpg-steps=1e6 \
             --continuous-svpg \
             --freeze-svpg
}

pusher_3dof_full_dr() {
    python scripts/launch.py --main=unfreeze_policy --setting=pusher --prefix=fulldr-baseline --seeds 5 \
        with --randomized-env-id="Pusher3DOFRandomized-v0" \
             --freeze-discriminator \
             --agent-name=baseline-full-dr \
             --initial-svpg-steps=1e6 \
             --continuous-svpg \
             --freeze-svpg
}

pusher_3dof_ours() {
    python scripts/launch.py --main=unfreeze_policy --setting=pusher --prefix=ours-pusher-3p --seeds 5 \
        with --randomized-env-id="Pusher3DOFRandomized-v0" \
             --agent-name=ours-pusher \
             --continuous-svpg
}

pusher_3dof_ours_5p() {
    python scripts/launch.py --main=unfreeze_policy --setting=pusher --prefix=ours-pusher-5p --seeds 5 \
        with --randomized-env-id="Pusher3DOFRandomized-v0" \
             --agent-name=ours-pusher-5p \
             --continuous-svpg \
             --nagents=5
}