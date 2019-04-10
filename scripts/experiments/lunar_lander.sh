#!/usr/bin/env bash


# =============== Lunar Lander ============

lunar_lander_baseline() {
    python scripts/launch.py --main=unfreeze_policy --setting=lunar --prefix=true-baseline --seeds 5 \
        with --freeze-discriminator \
             --agent-name=baseline \
             --initial-svpg-steps=1e6 \
             --continuous-svpg \
             --freeze-svpg
}

lunar_lander_full_dr() {
    python scripts/launch.py --main=unfreeze_policy --setting=lunar --prefix=fulldr-baseline --seeds 5 \
        with --randomized-env-id="LunarLanderRandomized-v0" \
             --freeze-discriminator \
             --agent-name=baseline-full-dr \
             --initial-svpg-steps=1e6 \
             --continuous-svpg \
             --freeze-svpg
}

lunar_lander_expert_813() {
    python scripts/launch.py --main=unfreeze_policy --setting=lunar --prefix=e813-baseline --seeds 5 \
        with --randomized-env-id="LunarLanderRandomized-RandomM813-v0" \
             --freeze-discriminator \
             --agent-name=expert-813 \
             --initial-svpg-steps=1e6 \
             --continuous-svpg \
             --freeze-svpg
}

lunar_lander_expert_811() {
    python scripts/launch.py --main=unfreeze_policy --setting=lunar --prefix=e811-baseline --seeds 5 \
        with --randomized-env-id="LunarLanderRandomized-RandomM811-v0" \
             --freeze-discriminator \
             --agent-name=expert-811 \
             --initial-svpg-steps=1e6 \
             --continuous-svpg \
             --freeze-svpg
}

lunar_lander_ours_1d() {
    python scripts/launch.py --main=unfreeze_policy --setting=lunar --prefix=ours-lunar1d --seeds 5 \
        with --randomized-env-id="LunarLanderRandomized-v0" \
             --agent-name=ours-lunar1d \
             --continuous-svpg
}

lunar_lander_ours_1d_5p() {
    python scripts/launch.py --main=unfreeze_policy --setting=lunar --prefix=ours-lunar1d-5p --seeds 5 \
        with --randomized-env-id="LunarLanderRandomized-v0" \
             --agent-name=ours-lunar1d-5p \
             --continuous-svpg \
             --nagents=5
}

lunar_lander_ours_1d_5p_6_20() {
    python scripts/launch.py --main=unfreeze_policy --setting=lunar --prefix=ours-lunar1d-5p-620 --seeds 1 \
        with --randomized-env-id="LunarLanderRandomized-RandomM620-v0" \
             --agent-name=ours-lunar1d-5p-620 \
             --continuous-svpg \
             --nagents=5
}