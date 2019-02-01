import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        hidden_sizes = [32, 32, 32]
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.m_ = nn.Linear(hidden_sizes[2], hidden_sizes[2])
        self.m = nn.Linear(hidden_sizes[2], action_size)
        self.s_ = nn.Linear(hidden_sizes[2], hidden_sizes[2])
        self.s = nn.Linear(hidden_sizes[2], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        s_ = self.s_(x)
        s_ = F.relu(s_)
        out_sigma = self.s(s_)
        sigmasq = out_sigma*out_sigma

        m_ = self.m_(x)
        m_ = F.relu(m_)

        out_mu = self.m(m_)

        return out_mu, sigmasq