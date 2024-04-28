import pickle
import pandas as pd
import torch
from tqdm import tqdm
from itertools import count

from dqn import *
from replay import *
from action import *
from plotter import *
from optimizer import *
from DQN_graph_color_env import *

BATCH_SIZE = 64
NUM_EPISODES = 10_000
DISCOUNT_FACTOR = 1.0
MEMORY_SIZE = 10000
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
OUTPUT_CONTINUOUS = False
NUM_PERMUTATIONS = 1000

# Original best without mutation
with open("NEV/Results/Chase/best_individual_v2.pkl", "rb") as f:
    best_individual = pickle.load(f)

print("Best Individual:")
for item in best_individual.items():
    print(item)

env = GraphColoring()
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state["graph"].flatten(order="C")) + len(state["node_colors"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


policy_net = DQN(
    n_observations, n_actions, best_individual["layers"], best_individual["activations"]
).to(device)
target_net = DQN(
    n_observations, n_actions, best_individual["layers"], best_individual["activations"]
).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=best_individual["lr"], amsgrad=True)
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0
episode_durations = []
reward_history = []
avg_reward = []

for i_episode in tqdm(range(NUM_EPISODES), desc="Episodes", colour="blue"):
    state, info = env.reset()
    graph_tensor = torch.tensor(
        state["graph"].flatten(order="C"), dtype=torch.int64, device=device
    )
    node_colors_tensor = torch.tensor(
        state["node_colors"], dtype=torch.int64, device=device
    )
    state = torch.concat((graph_tensor, node_colors_tensor))
    for t in count():
        action = select_action(
            state, env, steps_done, policy_net, EPS_START, EPS_END, EPS_DECAY, device
        )
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device) * 10
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_graph_tensor = torch.tensor(
                observation["graph"].flatten(order="C"),
                dtype=torch.int64,
                device=device,
            )
            next_node_colors_tensor = torch.tensor(
                observation["node_colors"], dtype=torch.int64, device=device
            )
            next_state = torch.concat((next_graph_tensor, next_node_colors_tensor))

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model(
            policy_net,
            target_net,
            memory,
            optimizer,
            device,
            DISCOUNT_FACTOR,
            BATCH_SIZE,
        )

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * best_individual[
                "tau"
            ] + target_net_state_dict[key] * (1 - best_individual["tau"])
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            reward_history.append(reward)

            avg_reward.append(np.mean(reward_history))
            if OUTPUT_CONTINUOUS:
                plot_durations(reward_history, episode_durations, avg_reward)

            # num_colors_used = len(set(observation['node_colors']))
            break

plot_durations(reward_history, episode_durations, avg_reward, True)

torch.save(policy_net, "./trained_model.pt")
print("Saved Model")

trained_policy_net = torch.load("./trained_model.pt")
env = GraphColoring()
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state["graph"].flatten(order="C")) + len(state["node_colors"])

import csv

csv_file = open("permutations_DQN.csv", "w")
writer = csv.writer(csv_file)
writer.writerow(["iteration", "node_order", "steps"])

# Default node order
node_order = [i for i in range(len(env.graph))]

try:
    done = False
    for iteration in tqdm(
        range(NUM_PERMUTATIONS), desc="Permutation Tests", colour="blue"
    ):

        graph_tensor = torch.tensor(
            state["graph"].flatten(order="C"), dtype=torch.int64, device=device
        )
        node_colors_tensor = torch.tensor(
            state["node_colors"], dtype=torch.int64, device=device
        )
        state = torch.concat((graph_tensor, node_colors_tensor))

        for step in count(1):
            action = select_action(
                state,
                env,
                step,
                trained_policy_net,
                eps_start=0.05,
                eps_end=0.05,
                eps_decay=EPS_DECAY,
                device=device,
            )
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device) * 10
            # print(action, reward)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_graph_tensor = torch.tensor(
                    observation["graph"].flatten(order="C"),
                    dtype=torch.int64,
                    device=device,
                )
                next_node_colors_tensor = torch.tensor(
                    observation["node_colors"], dtype=torch.int64, device=device
                )
                next_state = torch.concat((next_graph_tensor, next_node_colors_tensor))

            state = next_state
            if done:
                writer.writerow([iteration, node_order, step + 1])
                state, info = env.reset()
                node_order = env.permute()
                # print("Steps needed: ", step)
                break

except Exception as e:
    print("Exception: ", e)
finally:
    csv_file.close()
