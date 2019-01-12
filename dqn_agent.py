import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from policy import Policy
from state import State

import torch
import torch.nn.functional as F
import torch.optim as optim

import random

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 1        # how often to update the network
UPDATE_EVERY2 = 1
LR2 = 5e-4
num_of_batch_step = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.embedding_size = 16

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.policy_network = QNetwork(state_size, action_size, seed).to(device)

        self.qnetwork_optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=LR2)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.t_step2 = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                for b in range(num_of_batch_step):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_network.eval()
        with torch.no_grad():
            action_policy_mu, _ = self.policy_network(state)
            action_policy = F.softmax(action_policy_mu, dim=1)
        self.policy_network.train()

        action_policy = action_policy.data.cpu().numpy().astype(float)
        action_policy /= action_policy.sum(axis=1)

        return np.argmax(np.random.multinomial(1, action_policy[0])).astype(int)

        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get expected predicted Q values (for next states) from target model and the action policy network
        self.policy_network.eval()
        action_policy_next_mu, _ = self.policy_network(next_states)
        action_policy_next = F.softmax(action_policy_next_mu.detach(), dim=1).unsqueeze(1)
        self.policy_network.train()

        self.qnetwork_target.eval()
        Q_target_mu, Q_target_sigma = self.qnetwork_target(next_states)
        self.qnetwork_target.train()
        
        Q_targets_next = torch.matmul(action_policy_next, Q_target_mu.detach().unsqueeze(2)).squeeze().unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))


        # Get expected Q values and std from local model
        self.qnetwork_local.eval()
        Q_local_mu, Q_local_sigma = self.qnetwork_local(states)
        Q_expected_mu = Q_local_mu.gather(1, actions)
        Q_expected_sigma = Q_local_sigma.gather(1, actions)
        self.qnetwork_local.train()
        
        def GAUSS_NLL(mu, sigmasq, target):
            log_likelihood = torch.distributions.Normal(mu, sigmasq+1e-4).log_prob(target)
            loss = torch.mean(-log_likelihood)
            return loss
        
        # Compute loss
        loss = GAUSS_NLL(Q_expected_mu, Q_expected_sigma, Q_targets)
        # Minimize the loss
        self.qnetwork_optimizer.zero_grad()
        loss.backward()
        self.qnetwork_optimizer.step()


        self.t_step2 = (self.t_step2 + 1) % UPDATE_EVERY2
        if self.t_step2 == 0:

            # Action policy
            Q_max = Q_local_mu.detach().max(1)[1] # maximum Q value
            self.policy_network.eval()
            policy, _ = self.policy_network(states)
            self.policy_network.train()

            policy_loss = F.cross_entropy(policy, Q_max)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()


            # Exploration policy, calculated on next state with target Q network
            Q_explore_max = Q_target_sigma.detach().max(1)[1] # maximum entropy
            self.policy_network.eval()
            policy_explore, _ = self.policy_network(next_states)
            self.policy_network.train()

            policy_explore_loss = F.cross_entropy(policy_explore, Q_explore_max)

            self.policy_optimizer.zero_grad()
            policy_explore_loss.backward()
            self.policy_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)