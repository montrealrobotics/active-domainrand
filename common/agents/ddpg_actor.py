import os
import torch
import logging
from common.agents.ddpg import Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGActor(object):
    def __init__(self, state_dim, action_dim, max_action=1, agent_name="baseline", load_agent=True, model_path=None):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.agent_name = agent_name
        self.model_path = model_path

        if load_agent:
            self._load()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy()

    def _load(self):
        if self.model_path is not None:
            logging.info('Loading DDPG from: {}'.format(self.model_path))
            self.actor.load_state_dict(torch.load(self.model_path, map_location=device))
        else:
            cur_dir = os.getcwd()
            full_path = os.path.join(cur_dir, 'saved-models/policy/baseline_actor.pth')
            self.actor.load_state_dict(torch.load(full_path, map_location=device))
