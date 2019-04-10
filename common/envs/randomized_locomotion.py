import os

import json
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as et

import mujoco_py


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos

    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


# TODO: this class is not Thread-Safe
class RandomizedLocomotionEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, kwargs.get('xml_name'), frame_skip=5)

        # randomization
        self.reference_path = os.path.join(os.path.dirname(mujoco_env.__file__), "assets", kwargs.get('xml_name'))
        self.reference_xml = et.parse(self.reference_path)
        self.config_file = kwargs.get('config')
        self.dimensions = []
        self.dimension_map = []
        self.suffixes = []
        self._locate_randomize_parameters()

    def _locate_randomize_parameters(self):
        self.root = self.reference_xml.getroot()
        with open(self.config_file, mode='r') as f:
            config = json.load(f)

        check_suffixes = config.get('suffixes', False)

        for entry in config['dimensions']:
            name = entry["name"]
            self.dimension_map.append([])
            for geom in config["geom_map"][name]:
                self.dimension_map[-1].append(self.root.find(".//geom[@name='{}']".format(geom)))

            if check_suffixes:
                suffix = config['suffixes'].get(name, "")
                self.suffixes.append(suffix)
            else:
                self.suffixes.append("")

    def _create_xml(self):
        for i, bodypart in enumerate(self.dimensions):
            for geom in self.dimension_map[i]:
                suffix = self.suffixes[i]
                value = "{:3f} {}".format(self.dimensions[i].current_value, suffix)
                geom.set('size', '{}'.format(value))

        return et.tostring(self.root, encoding='unicode', method='xml')

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