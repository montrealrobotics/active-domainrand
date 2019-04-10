import logging

import gym
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper

from common.envs.wrappers import RandomizedEnvWrapper

"""File Description:
Creates a vectorized environment with RandomizationEnvWrapper, which helps
for fast / general Domain Randomization.
The main thing to note here is unlike the OpenAI vectorized env,
the step command does not automatically reset.

We also provide simple helper functions to randomize environments
"""

logger = logging.getLogger(__name__)


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        env = RandomizedEnvWrapper(env, seed + rank)

        env.seed(seed + rank)
        obs_shape = env.observation_space.shape  # TODO: is this something we can remove

        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes):
    envs = [make_env(env_name, seed, i)
            for i in range(num_processes)]
    envs = RandomizedSubprocVecEnv(envs)
    return envs


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space, env.unwrapped.randomization_space))
            elif cmd == 'get_dimension_name':
                remote.send(env.unwrapped.dimensions[data].name)
            elif cmd == 'rescale_dimension':
                dimension = data[0]
                array = data[1]
                rescaled = env.unwrapped.dimensions[dimension]._rescale(array)
                remote.send(rescaled)
            elif cmd == 'randomize':
                randomized_val = data
                env.randomize(randomized_val)
                remote.send(None)
            elif cmd == 'get_current_randomization_values':
                values = []
                for dim in env.unwrapped.dimensions:
                    values.append(dim.current_value)

                remote.send(values)
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class RandomizedSubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    # TODO: arg spaces is no longer used. Remove?
    def __init__(self, env_fns, spaces=None):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space, randomization_space = self.remotes[0].recv()
        self.randomization_space = randomization_space
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        logger.debug('[step] => SENDING')
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        logger.debug('[step] => SENT')
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        logger.debug('[step] => WAITING')
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        logger.debug('[step] => DONE')
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def randomize(self, randomized_values):
        self._assert_not_closed()

        logger.debug('[randomize] => SENDING')
        for remote, val in zip(self.remotes, randomized_values):
            remote.send(('randomize', val))
        results = [remote.recv() for remote in self.remotes]  # TODO: why creating the array if you're not gonna use it
        logger.debug('[randomize] => SENT')
        self.waiting = False

    def get_current_params(self):
        logger.debug('[get_current_randomization_values] => SENDING')
        for remote in self.remotes:
            remote.send(('get_current_randomization_values', None))
        result = [remote.recv() for remote in self.remotes]
        logger.debug('[get_current_randomization_values] => SENT')
        return np.stack(result)

    def get_dimension_name(self, dimension):
        logger.debug('[get_dimension_name] => SENDING')
        self.remotes[0].send(('get_dimension_name', dimension))
        result = self.remotes[0].recv()
        logger.debug('[get_dimension_name] => SENT')
        return result

    def rescale(self, dimension, array):
        logger.debug('[rescale_dimension] => SENDING')
        data = (dimension, array)
        self.remotes[0].send(('rescale_dimension', data))
        result = self.remotes[0].recv()
        logger.debug('[rescale_dimension] => SENT')
        return result

    def reset(self):
        self._assert_not_closed()
        logger.debug('[reset] => SENDING')
        for remote in self.remotes:
            remote.send(('reset', None))
        result = [remote.recv() for remote in self.remotes]
        logger.debug('[reset] => SENT')
        return np.stack(result)

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
