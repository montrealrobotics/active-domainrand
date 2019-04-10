import numpy as np
import torch.nn as nn


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


orthogonal_init = lambda m: init(module=m,
                                 weight_init=nn.init.orthogonal_,
                                 bias_init=lambda x: nn.init.constant_(x, 0),
                                 gain=np.sqrt(2))
