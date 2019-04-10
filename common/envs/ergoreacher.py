from gym_ergojr.envs import ErgoReacherEnv
import numpy as np


class ErgoReacherRandomizedEnv(ErgoReacherEnv):
    def __init__(self, **kwargs):
        self.dimensions = []  # this will be 8 elements long after wrapper init
        self.config_file = kwargs.get('config')

        del kwargs['config']

        super().__init__(**kwargs)

        # # these two are affected by the DR
        # self.max_force
        # self.max_vel

    def step(self, action):
        observation, reward, done, info = super().step(action)
        info = {'goal_dist': self.dist.query()}
        return observation, reward, False, info  # I'm not crazy excited about the lack of early stopping, but alright

    def _update_randomized_params(self):
        # these are used automatically in the `step` function
        self.max_force = np.zeros(6, np.float32)
        self.max_vel = np.zeros(6, np.float32)

        if self.simple:
            self.max_force[[1, 2, 4, 5]] = [x.current_value for x in self.dimensions[:4]]
            self.max_vel[[1, 2, 4, 5]] = [x.current_value for x in self.dimensions[4:]]
        else:
            self.max_force[:] = [x.current_value for x in self.dimensions[:6]]
            self.max_vel[:] = [x.current_value for x in self.dimensions[6:]]
