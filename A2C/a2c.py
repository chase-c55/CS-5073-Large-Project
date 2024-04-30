# This a2c implementation was inspired by Github user Lucasc-99's Actor-Critic RL Algorithm in PyTorch. See the repo here: https://github.com/Lucasc-99/Actor-Critic

import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_node_actions, num_color_actions, hidden_size, gamma=0.99, learning_rate=0.01):
        super(ActorCritic, self).__init__()
        
        # Set hyperparameters
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Define the actor network
        self.actor_node = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_node_actions)
        )
        
        self.actor_color = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_color_actions)
        )
        
        # Define the critic network
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(list(self.actor_node.parameters()) + list(self.actor_color.parameters()), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
    
    def actor(self, state):
        node_logits = self.actor_node(state)
        color_logits = self.actor_color(state)
        return node_logits, color_logits

    def select_action(self, state):
        # Convert state to tensor
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        else:
            state = state.clone().detach()

        # Convert state to float
        state = state.float()
        
        # Get action logits from the actor network
        node_logits, color_logits = self.actor(state)
        
        # Convert action logits to probabilities
        node_probs = nn.functional.softmax(node_logits, dim=-1)
        color_probs = nn.functional.softmax(color_logits, dim=-1)

        # Create categorical distributions
        node_dist = Categorical(node_probs)
        color_dist = Categorical(color_probs)
        
        # Sample actions from the distributions
        node_action = node_dist.sample()
        color_action = color_dist.sample()
        
        # Get the log probabilities of the sampled actions
        node_log_prob = node_dist.log_prob(node_action)
        color_log_prob = color_dist.log_prob(color_action)
        
        # Get the value estimate from the critic network
        value = self.critic(state)
        
        return (node_action, color_action), node_log_prob + color_log_prob, value
    
    def get_value(self, state):
        if state is None:
            return torch.tensor(0, dtype=torch.float32)
        else:
            # Convert state to tensor if it's not already a tensor
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32)
            else:
                state = state.clone().detach()

            # Convert state to float
            state = state.float()
            
            return self.critic(state)
    
    def update(self, advantage, log_prob, value):
        # Compute actor loss
        actor_loss = -(log_prob * advantage.detach()).mean()
        
        # Compute critic loss
        critic_loss = nn.functional.mse_loss(value, advantage.detach())
        
        # Zero gradients for actor and critic optimizers
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        # Perform backward pass for actor loss
        actor_loss.backward()
        
        # Perform backward pass for critic loss
        critic_loss.backward()
        
        # Update actor and critic parameters
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def evaluate_action(self, state, action):
        # Convert state to tensor if it's not already a tensor
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        else:
            state = state.clone().detach()

        # Convert state to float
        state = state.float()

        # Get action logits from the actor network
        node_logits, color_logits = self.actor(state)

        # Convert action logits to probabilities
        node_probs = nn.functional.softmax(node_logits, dim=-1)
        color_probs = nn.functional.softmax(color_logits, dim=-1)

        # Create categorical distributions
        node_dist = Categorical(node_probs)
        color_dist = Categorical(color_probs)

        # Get the log probabilities of the given actions
        node_action, color_action = action
        node_log_prob = node_dist.log_prob(node_action)
        color_log_prob = color_dist.log_prob(color_action)

        # Get the value estimate from the critic network
        value = self.critic(state)

        return node_log_prob, color_log_prob, value
