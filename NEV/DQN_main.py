#!/usr/bin/env python3

import time
import pickle
import csv
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# We are going to place everythng in its own py file
from dqn import *
from replay import *
from action import *
from plotter import *
from optimizer import *
# from graph_color_env import *
from DQN_graph_color_env import *

env = GraphColoring()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


# training
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state["graph"].flatten(order="C")) + len(state["node_colors"])

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

num_episodes = 10000
# if torch.cuda.is_available():
#     num_episodes = 600
# else:
#     num_episodes = 50


# initial variables that needed to be "zeroed out"
steps_done = 0
episode_durations = []
reward_history = []
avg_reward = []

OUTPUT_CONTINUOUS = True

csv_file = open('output.csv', 'w')
writer = csv.writer(csv_file)
writer.writerow(["episode", "steps", "reward", "num_colors"])

# main loop
for i_episode in range(num_episodes):
    print("#" * 80)
    print("On episode: ", i_episode)
    # Initialize the environment and get it's state
    obs, info = env.reset()
    state = torch.tensor(
        obs["graph"].flatten(order="C"), dtype=torch.int64, device=device
    )
    state = torch.concat(
        (state, torch.tensor(obs["node_colors"], dtype=torch.int64, device=device))
    )
    for t in count():
        # Pick the action and increment the steps taken
        action = select_action(
            state, env, steps_done, policy_net, EPS_START, EPS_END, EPS_DECAY, device
        )

        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation["graph"].flatten(order="C"),
                dtype=torch.int64,
                device=device,
            )
            next_state = torch.concat(
                (
                    next_state,
                    torch.tensor(
                        observation["node_colors"], dtype=torch.int64, device=device
                    ),
                )
            )

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model(
            policy_net, target_net, memory, optimizer, device, GAMMA, BATCH_SIZE
        )

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            reward_history.append(reward)
            
            # avg_reward.append(np.mean(reward_history))
            if OUTPUT_CONTINUOUS:
              draw_graph(observation["graph"], observation["node_colors"])
              plot_durations(reward_history, episode_durations, avg_reward)

            num_colors_used = len(set(observation['node_colors']))
            print(
                f"Steps: {t} Reward: {reward.item()} Num Colors: {num_colors_used}"
            )
            writer.writerow([i_episode, t, reward.item(), num_colors_used])
            break

# with open(f"dqn-{int(time.time())}.pt", "wb") as f:
#     torch.save(policy_net, f)  # TODO: Is this right?

csv_file.close()
print("Complete!")
plot_durations(reward_history, episode_durations, avg_reward, show_result=True)
