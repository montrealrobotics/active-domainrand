from gym.envs.registration import register

from common.envs.config import CONFIG_PATH
from common.envs.lunar_lander import LunarLanderRandomized
import os.path as osp

# Needed because of gym.space error in normal LunarLander-v2
register(
    id='LunarLanderDefault-v0',
    entry_point='common.envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'common/envs/config/LunarLanderRandomized/default.json'}
)

register(
    id='LunarLander10-v0',
    entry_point='common.envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'common/envs/config/LunarLanderRandomized/10.json'}
)

register(
    id='LunarLander16-v0',
    entry_point='common.envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'common/envs/config/LunarLanderRandomized/16.json'}
)

register(
    id='LunarLanderRandomized-v0',
    entry_point='common.envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'common/envs/config/LunarLanderRandomized/random_820.json'}
)

register(
    id='LunarLanderRandomized2D-v0',
    entry_point='common.envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'common/envs/config/LunarLanderRandomized/random2D_820.json'}
)

register(
    id='LunarLanderRandomized-RandomM811-v0',
    entry_point='common.envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'common/envs/config/LunarLanderRandomized/random_811.json'}
)

register(
    id='LunarLanderRandomized-RandomM812-v0',
    entry_point='common.envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'common/envs/config/LunarLanderRandomized/random_812.json'}
)

register(
    id='LunarLanderRandomized-RandomM813-v0',
    entry_point='common.envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'common/envs/config/LunarLanderRandomized/random_813.json'}
)

register(
    id='LunarLanderRandomized-RandomM1720-v0',
    entry_point='common.envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'common/envs/config/LunarLanderRandomized/random_1720.json'}
)

register(
    id='LunarLanderRandomized-RandomM620-v0',
    entry_point='common.envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'common/envs/config/LunarLanderRandomized/random_620.json'}
)

register(
    id='PusherDefault-v0',
    entry_point='common.envs.pusher:PusherRandomizedEnv',
    max_episode_steps=100,
    kwargs={'config': 'common/envs/config/PusherRandomized/default.json'}
)

register(
    id='Pusher3DOFDefault-v0',
    entry_point='common.envs.pusher3dof:PusherEnv3DofEnv',
    max_episode_steps=100,
    kwargs={'config': 'common/envs/config/Pusher3DOFRandomized/default.json'}
)

register(
    id='Pusher3DOFRandomized-v0',
    entry_point='common.envs.pusher3dof:PusherEnv3DofEnv',
    max_episode_steps=100,
    kwargs={'config': 'common/envs/config/Pusher3DOFRandomized/fulldr.json'}
)

register(
    id='Pusher3DOFRandomizedEasy-v0',
    entry_point='common.envs.pusher3dof:PusherEnv3DofEnv',
    max_episode_steps=100,
    kwargs={'config': 'common/envs/config/Pusher3DOFRandomized/fulldr-easy.json'}
)

register(
    id='Pusher3DOFHard-v0',
    entry_point='common.envs.pusher3dof:PusherEnv3DofEnv',
    max_episode_steps=100,
    kwargs={'config': 'common/envs/config/Pusher3DOFRandomized/hard.json'}
)

register(
    id='Pusher3DOFUberHard-v0',
    entry_point='common.envs.pusher3dof:PusherEnv3DofEnv',
    max_episode_steps=100,
    kwargs={'config': 'common/envs/config/Pusher3DOFRandomized/fulldr-toohard.json'}
)

# Pusher Generalization Environments
for i in range(3):
    for j in range(3):
        register(
            id='Pusher3DOFGeneralization{}{}-v0'.format(i, j),
            entry_point='common.envs.pusher3dof:PusherEnv3DofEnv',
            max_episode_steps=100,
            kwargs={'config': 'common/envs/config/Pusher3DOFGeneralization/{}{}.json'.format(i, j)}
        )

register(
    id='ErgoReacher4DOFRandomizedEasy-v0',
    entry_point='common.envs.ergoreacher:ErgoReacherRandomizedEnv',
    max_episode_steps=100,
    kwargs={
        'config': osp.join(
            CONFIG_PATH,
            'ErgoReacherRandomized',
            'easy-4dof.json'
        ),
        'headless': True,
        'simple': True,
        'goal_halfsphere': True
    }
)

register(
    id='ErgoReacher4DOFRandomizedHard-v0',
    entry_point='common.envs.ergoreacher:ErgoReacherRandomizedEnv',
    max_episode_steps=100,
    kwargs={
        'config': osp.join(
            CONFIG_PATH,
            'ErgoReacherRandomized',
            'hard-4dof.json'
        ),
        'headless': True,
        'simple': True,
        'goal_halfsphere': True
    }
)

register(
    id='ErgoReacher4DOFRandomizedHardVisual-v0',
    entry_point='common.envs.ergoreacher:ErgoReacherRandomizedEnv',
    max_episode_steps=300,
    kwargs={
        'config': osp.join(
            CONFIG_PATH,
            'ErgoReacherRandomized',
            'hard-4dof.json'
        ),
        'headless': False,
        'simple': True,
        'goal_halfsphere': True
    }
)


register(
    id='ErgoReacher4DOFDefault-v0',
    entry_point='common.envs.ergoreacher:ErgoReacherRandomizedEnv',
    max_episode_steps=100,
    kwargs={
        'config': osp.join(
            CONFIG_PATH,
            'ErgoReacherRandomized',
            'default-4dof.json'
        ),
        'headless': True,
        'simple': True,
        'goal_halfsphere': True
    }
)

register(
    id='ErgoReacherRandomizedBacklashEasy-v0',
    entry_point='common.envs.ergoreacherbacklash:ErgoReacherRandomizedBacklashEnv',
    max_episode_steps=200,
    kwargs={
        'config': osp.join(
            CONFIG_PATH,
            'ErgoReacherRandomizedBacklash',
            'fulldr-easy.json'
        ),
        'headless': True,
        'simple': True,
        'goal_halfsphere': False,
        'double_goal': True
    }
)

register(
    id='ErgoReacherRandomizedBacklashHard-v0',
    entry_point='common.envs.ergoreacherbacklash:ErgoReacherRandomizedBacklashEnv',
    max_episode_steps=200,
    kwargs={
        'config': osp.join(
            CONFIG_PATH,
            'ErgoReacherRandomizedBacklash',
            'fulldr-hard.json'
        ),
        'headless': True,
        'simple': True,
        'goal_halfsphere': False,
        'double_goal': True
    }
)

for headlessness in ["Headless", "Graphical"]:
    for randomization in ["Default", "Randomized"]:

        register(
            id='ErgoReacher{}-{}-v0'.format(randomization, headlessness),
            entry_point='common.envs.ergoreacher:ErgoReacherRandomizedEnv',
            max_episode_steps=100,
            kwargs={
                'config': osp.join(
                    CONFIG_PATH,
                    'ErgoReacherRandomized',
                    'default-4dof.json' if randomization == "Default" else 'fulldr-4dof.json'
                ),
                'headless': True if headlessness == "Headless" else False,
                'simple': True
            }
        )

        register(
            id='ErgoReacher-Halfdisk-{}-{}-v0'.format(randomization, headlessness),
            entry_point='common.envs.ergoreacher:ErgoReacherRandomizedEnv',
            max_episode_steps=100,
            kwargs={
                'config': osp.join(
                    CONFIG_PATH,
                    'ErgoReacherRandomized',
                    'default-4dof.json' if randomization == "Default" else 'fulldr-4dof.json'
                ),
                'headless': True if headlessness == "Headless" else False,
                'simple': True,
                'goal_halfsphere': True
            }
        )

        register(
            id='ErgoReacher-Halfdisk-Backlash-{}-{}-v0'.format(randomization, headlessness),
            entry_point='common.envs.ergoreacherbacklash:ErgoReacherRandomizedBacklashEnv',
            max_episode_steps=100,
            kwargs={
                'config': osp.join(
                    CONFIG_PATH,
                    'ErgoReacherRandomizedBacklash',
                    'default-4dof.json' if randomization == "Default" else 'fulldr-4dof.json'
                ),
                'headless': True if headlessness == "Headless" else False,
                'simple': True,
                'goal_halfsphere': True
            }
        )

        register(
            id='ErgoReacher-DualGoal-{}-{}-v0'.format(randomization, headlessness),
            entry_point='common.envs.ergoreacherbacklash:ErgoReacherRandomizedBacklashEnv',
            max_episode_steps=200,
            kwargs={
                'config': osp.join(
                    CONFIG_PATH,
                    'ErgoReacherRandomizedBacklash',
                    'default-4dof.json' if randomization == "Default" else 'fulldr-4dof.json'
                ),
                'headless': True if headlessness == "Headless" else False,
                'simple': True,
                'goal_halfsphere': True,
                'double_goal': True
            }
        )

        register(
            id='ErgoReacher-DualGoal-Easy-{}-{}-v0'.format(randomization, headlessness),
            entry_point='common.envs.ergoreacherbacklash:ErgoReacherRandomizedBacklashEnv',
            max_episode_steps=200,
            kwargs={
                'config': osp.join(
                    CONFIG_PATH,
                    'ErgoReacherRandomizedBacklash',
                    'default-4dof.json' if randomization == "Default" else 'fulldr-4dof.json'
                ),
                'headless': True if headlessness == "Headless" else False,
                'simple': True,
                'goal_halfsphere': False,
                'double_goal': True
            }
        )

        register(
            id='ErgoReacher-6Dof-{}-{}-v0'.format(randomization, headlessness),
            entry_point='common.envs.ergoreacher:ErgoReacherRandomizedEnv',
            max_episode_steps=100,
            kwargs={
                'config': osp.join(
                    CONFIG_PATH,
                    'ErgoReacherRandomized',
                    'default-6dof.json' if randomization == "Default" else 'fulldr-6dof.json'
                ),
                'headless': True if headlessness == "Headless" else False,
                'simple': False,
                'goal_halfsphere': False
            }
        )

register(
    id='HumanoidRandomizedEnv-v0',
    entry_point='common.envs.humanoid:HumanoidRandomizedEnv',
    max_episode_steps=1000,
    kwargs={
        'config': osp.join(
            CONFIG_PATH,
            'HumanoidRandomized',
            'default.json'
        ),
        'xml_name': 'humanoid.xml'
    }
)

register(
    id='HalfCheetahRandomizedEnv-v0',
    entry_point='common.envs.half_cheetah:HalfCheetahRandomizedEnv',
    max_episode_steps=1000,
    kwargs={
        'config': osp.join(
            CONFIG_PATH,
            'HalfCheetahRandomized',
            'default.json'
        ),
        'xml_name': 'half_cheetah.xml'
    }
)
