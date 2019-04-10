from gym_ergojr.envs import ErgoReacherHeavyEnv
import numpy as np


class ErgoReacherRandomizedBacklashEnv(ErgoReacherHeavyEnv):
    def __init__(self, **kwargs):
        self.dimensions = []  # this will be 8 elements long after wrapper init
        self.config_file = kwargs.get('config')

        del kwargs['config']

        super().__init__(**kwargs)

        # # these three are affected by the DR
        # self.max_force
        # backlash + self.force_urdf_reload

    def step(self, action):
        observation, reward, done, info = super().step(action)
        info = {'goal_dist': self.dist.query()}
        return observation, reward, False, info

    def _update_randomized_params(self):
        # the self.max_force is used automatically in the step function,
        # but for the backlash to take effect, self.reset() has to be called
        self.max_force = np.zeros(6, np.float32)
        backlash = np.zeros(6, np.float32)

        if self.simple:
            self.max_force[[0, 3]] = [1000, 1000] # setting these to default

            self.max_force[[1, 2, 4, 5]] = [x.current_value for x in self.dimensions[:4]]

            # The values coming into the backlash from the JSON are from -2.302585*4 = -9.2103 to 0
            # ...so that when we do e^[-9.2103,0] we get [0.0001,1]
            backlash[[1, 2, 4, 5]] = [np.power(np.e, x.current_value) for x in self.dimensions[4:]]
            self.update_backlash(backlash)
        else:
            raise NotImplementedError("just ping me and I'll write this if need be")
            # reason I haven't written this yet is because
            # the 6dof+backlash task is wayyy too hard

        self.reset()
