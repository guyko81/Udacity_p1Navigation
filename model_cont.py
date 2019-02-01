import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        hidden_sizes = [512, 256, 128, 64, 32]
        self.bn1 = nn.BatchNorm1d(state_size + action_size)
        self.fc1 = nn.Linear(state_size + action_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])

        self.V_ = nn.Linear(hidden_sizes[4], hidden_sizes[4])
        self.V = nn.Linear(hidden_sizes[4], 1)
        self.A_ = nn.Linear(hidden_sizes[4], hidden_sizes[4])
        self.A = nn.Linear(hidden_sizes[4], 1)


    def forward(self, state, action):
        """Build a network that maps state, action -> values."""
        inp = torch.cat((state.float(), action.float()), dim=1)
        x = self.bn1(inp)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1)
        x = self.fc3(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0.1)
        x = self.fc4(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0.1)
        x = self.fc5(x)
        x = F.relu(x)
        
        V_ = self.V_(x)
        V_ = F.relu(V_)
        #V = self.V(V_)
        out_sigma = self.V(V_)
        sigmasq = out_sigma*out_sigma

        A_ = self.A_(x)
        A_ = F.relu(A_)
        #A = self.A(A_)
        out_mu = self.A(A_)

        #out = V + A - torch.mean(A, dim=1, keepdim=True)

        return out_mu, sigmasq