import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as et

import mujoco_py

PLANE_LOCATION_Z = -0.325


# TODO: this class is not Thread-Safe
class PusherRandomizedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pusher.xml', frame_skip=5)

        # randomization
        self.reference_path = os.path.join(os.path.dirname(mujoco_env.__file__), "assets", 'pusher.xml')
        self.reference_xml = et.parse(self.reference_path)
        self.config_file = kwargs.get('config')
        self.dimensions = []
        self._locate_randomize_parameters()

    def _locate_randomize_parameters(self):
        self.root = self.reference_xml.getroot()
        self.geom = self.root.find("./default/geom[@friction]")
        roll_link = self.root.find(".//body[@name='r_wrist_roll_link']")
        self.wrist = roll_link.findall("./geom[@type='capsule']")
        self.tips = roll_link.findall("./body[@name='tips_arm']/geom")
        self.object_body = self.root.find(".//body[@name='object']")
        self.object_body_geom = self.object_body.findall('./geom')
        self.goal_body = self.root.find(".//body[@name='goal']/geom")

    def _update_randomized_params(self):
        xml = self._create_xml()
        self._re_init(xml)

    def _re_init(self, xml):
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done
        if self.viewer:
            self.viewer.update_sim(self.sim)

    def _create_xml(self):
        # TODO: I might speed this up, but I think is insignificant w.r.t to the model/sim creation...
        self._randomize_friction()
        self._randomize_density()
        self._randomize_size()

        return et.tostring(self.root, encoding='unicode', method='xml')

    # TODO: I'm making an assumption here that 3 places after the comma are good enough, are they?
    def _randomize_friction(self):
        self.geom.set('friction', '{:3f} 0.1 0.1'.format(self.dimensions[0].current_value))

    def _randomize_density(self):
        self.geom.set('density', '{:3f}'.format(self.dimensions[1].current_value))

    def _randomize_size(self):
        size = self.dimensions[2].current_value

        # grabber
        grabber_width = size * 2
        self.wrist[0].set('fromto', '0 -{:3f} 0. 0.0 +{:3f} 0'.format(grabber_width, grabber_width))
        self.wrist[1].set('fromto', '0 -{:3f} 0. {:3f} -{:3f} 0'.format(grabber_width, grabber_width, grabber_width))
        self.wrist[2].set('fromto', '0 +{:3f} 0. {:3f} +{:3f} 0'.format(grabber_width, grabber_width, grabber_width))
        self.tips[0].set('pos', '{:3f} -{:3f} 0.'.format(grabber_width, grabber_width))
        self.tips[1].set('pos', '{:3f} {:3f} 0.'.format(grabber_width, grabber_width))

        # object
        # self.object_body.set('pos', '0.45 -0.05 {:3f}'.format(PLANE_LOCATION_Z + size))
        # for geom in self.object_body_geom:
        #     geom.set('size', "{:3f} {:3f} {:3f}".format(size, size, size))

        # goal
        # TODO: maybe a constant here? 1.6 is 0.08 / 0.05, the goal diam shrinks with the object diam
        # self.goal_body.set('size', "{:3f} 0.001 0.1".format(size * 1.6))

    def step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate([
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])