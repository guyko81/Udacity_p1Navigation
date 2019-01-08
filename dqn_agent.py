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
UPDATE_EVERY2 = 2
LR2 = 5e-4

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
        self.policy_network = Policy(state_size, action_size, seed).to(device)

        self.qnetwork_explore = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_explore_target = QNetwork(state_size, action_size, seed).to(device)
        self.policy_network_explore = Policy(state_size, action_size, seed).to(device)

        self.state_network = State(state_size, self.embedding_size, seed).to(device)
        self.static_state_network = State(state_size, self.embedding_size, seed).to(device)

        self.qnetwork_optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=LR2)
        self.qnetwork_explore_optimizer = optim.Adam(self.qnetwork_explore.parameters(), lr=LR)
        self.policy_explore_optimizer = optim.Adam(self.policy_network_explore.parameters(), lr=LR2)
        self.state_network_optimizer = optim.Adam(self.state_network.parameters(), lr=LR2)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.t_step2 = 0
        
        # Freeze the embedding model
        self.static_state_network.eval()

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
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
        self.policy_network_explore.eval()
        with torch.no_grad():
            action_policy = F.softmax(self.policy_network(state), dim=1)
            action_policy_explore = F.softmax(self.policy_network_explore(state), dim=1)
        self.policy_network.train()
        self.policy_network_explore.train()
        
        action_policy = action_policy.data.cpu().numpy().astype(float)
        action_policy /= np.sum(action_policy)

        action_policy_explore = action_policy_explore.data.cpu().numpy().astype(float)
        action_policy_explore /= np.sum(action_policy_explore)

        # Averaging the 2 policies in the log-odds space
        #action_policy_ln = np.log(action_policy/(1-action_policy))
        #action_policy_explore_ln = np.log(action_policy_explore/(1-action_policy_explore))
        #action_policy_ln_avg = (action_policy_ln + action_policy_explore_ln)/2
        #action_policy_avg = 1/(1+np.exp(-action_policy_ln_avg))
        action_policy_avg = action_policy * action_policy_explore
        action_policy_avg /= np.sum(action_policy_avg)

        return np.argmax(np.random.multinomial(1, action_policy_avg[0])).astype(int)
        
        """
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_policy.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        """
        
        """
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(int)
        else:
            return random.choice(np.arange(self.action_size)).astype(int)
        """
        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        action_policy_next = F.softmax(self.policy_network(next_states).detach(), dim=1).unsqueeze(1)
        Q_targets_next = torch.matmul(action_policy_next, self.qnetwork_target(next_states).detach().unsqueeze(2)).squeeze().unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.qnetwork_optimizer.zero_grad()
        loss.backward()
        self.qnetwork_optimizer.step()


        
        # Exploration reward
        state_prediction = self.state_network(next_states)
        state_embedding = self.static_state_network(next_states).detach()
        rewards_explore = ((state_prediction - state_embedding)**2).mean(dim=1)

        # Get max predicted Q values (for next states) from target model for exploration model
        #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        action_policy_explore_next = F.softmax(self.policy_network_explore(next_states).detach(), dim=1).unsqueeze(1)
        Q_targets_explore_next = torch.matmul(action_policy_explore_next, self.qnetwork_explore_target(next_states).detach().unsqueeze(2)).squeeze().unsqueeze(1)

        # Compute Q targets for current states 
        Q_targets_explore = rewards_explore + (gamma * Q_targets_explore_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected_explore = self.qnetwork_explore(states).gather(1, actions)

        # Compute loss
        loss_explore = F.mse_loss(Q_expected_explore, Q_targets_explore)
        # Minimize the loss
        self.qnetwork_explore_optimizer.zero_grad()
        loss_explore.backward(retain_graph=True) # we want to keep the rewards_explore values for later
        self.qnetwork_explore_optimizer.step()

        

        # State prediction optimization
        rewards_explore_loss = rewards_explore.mean(dim=0)
        self.state_network_optimizer.zero_grad()
        rewards_explore_loss.backward()
        self.state_network_optimizer.step()


        self.t_step2 = (self.t_step2 + 1) % UPDATE_EVERY2
        if self.t_step2 == 0:

            # Action policy
            Q_targets_max = self.qnetwork_local(states).detach().max(1)[1]
            policy = self.policy_network(states)

            policy_loss = F.cross_entropy(policy, Q_targets_max)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()


            # Exploration policy
            Q_targets_explore_max = self.qnetwork_explore(states).detach().max(1)[1]
            policy_explore = self.policy_network_explore(states)

            policy_explore_loss = F.cross_entropy(policy_explore, Q_targets_explore_max)

            self.policy_explore_optimizer.zero_grad()
            policy_explore_loss.backward()
            self.policy_explore_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     
        self.soft_update(self.qnetwork_explore, self.qnetwork_explore_target, TAU)   

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