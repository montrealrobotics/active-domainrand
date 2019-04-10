import torch
import torch.nn as nn

class MLPDiscriminator(nn.Module):
    """Discriminator class based on Feedforward Network
    Input is a state-action-state' transition
    Output is probability that it was from a reference trajectory
    """
    def __init__(self, state_dim, action_dim):
        super(MLPDiscriminator, self).__init__()
        
        self.l1 = nn.Linear((state_dim + action_dim + state_dim), 128)
        self.l2 = nn.Linear(128, 128)
        self.logic = nn.Linear(128, 1)
        
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    # Tuple of S-A-S'
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.logic(x)
        return torch.sigmoid(x)

class GAILMLPDiscriminator(nn.Module):
    """Discriminator class based on Feedforward Network
    Input is a state-action-state' transition
    Output is probability that it was from a reference trajectory
    """
    def __init__(self, state_dim, action_dim):
        super(GAILMLPDiscriminator, self).__init__()
        self.l1 = nn.Linear((state_dim + action_dim), 128)
        self.l2 = nn.Linear(128, 128)
        self.logic = nn.Linear(128, 1)
        
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    # Tuple of S-A-S'
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.logic(x)
        return torch.sigmoid(x)


class LSTMDiscriminator(nn.Module):
    """Discriminator class based on Feedforward Network
    Input is a sequence of state-action-state' transitions
    Output is probability that it was from a reference trajectory
    """
    def __init__(self, state_dim, batch_size, hidden_dim):
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=1)
        self.state_dim = state_dim
        
        self.hidden_dim = hidden_dim
        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.hidden = self._init_hidden()

    def _init_hidden(self):
         return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, trajectory):
        self.hidden = self._init_hidden()

        predictions, (ht, ct) = self.lstm(trajectory, self.hidden)
        output = self.hidden2out(ht[-1])
        return torch.sigmoid(output)
