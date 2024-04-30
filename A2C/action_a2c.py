# simulates an action in a graph coloring state space, based on both node and color action parameters. Inspired by Chase's action.py code, but specialized to fit A2C.

import torch
import torch.nn as nn
from torch.distributions import Categorical

def select_action(model, state, device):
    # Convert state to tensor
    if not torch.is_tensor(state):
        state = torch.tensor(state, dtype=torch.float32, device=device)
    else:
        state = state.to(device).float()  # Convert state to Float data type

    # Get action logits from the actor network
    node_action_logits, color_action_logits = model.actor(state)

    # Apply softmax to obtain valid action probabilities
    node_action_probs = nn.functional.softmax(node_action_logits, dim=-1)
    color_action_probs = nn.functional.softmax(color_action_logits, dim=-1)

    # Convert action probabilities to categorical distributions
    node_dist = Categorical(node_action_probs)
    color_dist = Categorical(color_action_probs)

    # Sample an action from the distributions
    node_action = node_dist.sample()
    color_action = color_dist.sample()

    # Get the log probabilities of the sampled actions
    node_log_prob = node_dist.log_prob(node_action)
    color_log_prob = color_dist.log_prob(color_action)

    # Get the value estimate from the critic network
    value = model.critic(state)

    return (node_action, color_action), node_log_prob + color_log_prob, value

def evaluate_action(model, state, action, device):
    # Convert state to tensor
    if not torch.is_tensor(state):
        state = torch.tensor(state, dtype=torch.float32, device=device)
    else:
        state = state.to(device).float()  # Convert state to Float data type

    # Get action logits from the actor network
    node_logits, color_logits = model.actor(state)

    # Convert action logits to probabilities
    node_probs = nn.functional.softmax(node_logits, dim=-1)
    color_probs = nn.functional.softmax(color_logits, dim=-1)

    # Convert action probabilities to categorical distributions
    node_dist = Categorical(node_probs)
    color_dist = Categorical(color_probs)

    # Get the log probabilities of the given actions
    node_action, color_action = action
    node_log_prob = node_dist.log_prob(node_action)
    color_log_prob = color_dist.log_prob(color_action)

    # Get the value estimate from the critic network
    value = model.critic(state)

    return node_log_prob, color_log_prob, value
