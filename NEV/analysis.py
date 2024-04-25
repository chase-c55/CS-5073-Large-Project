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
NUM_EPISODES = 10000
DISCOUNT_FACTOR = 1.0
MEMORY_SIZE = 10000
EPS_START = 0.9 # FIXME: Change these?
EPS_END = 0.05
EPS_DECAY = 1000
OUTPUT_CONTINUOUS = True

gen_data_df = pd.read_csv("NEV/generation_data.csv")

with open("NEV/best_individual-1713946820.pkl",'rb') as f:
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


policy_net = DQN(n_observations, n_actions, best_individual['layers'], best_individual['activations']).to(device)
target_net = DQN(n_observations, n_actions, best_individual['layers'], best_individual['activations']).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=best_individual['lr'], amsgrad=True)
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
                device=device
            )
            next_node_colors_tensor = torch.tensor(
                observation["node_colors"],
                dtype=torch.int64,
                device=device
            )
            next_state = torch.concat(
                (next_graph_tensor, next_node_colors_tensor)
            )

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model(
            policy_net, target_net, memory, optimizer, device, DISCOUNT_FACTOR, BATCH_SIZE
        )

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * best_individual['tau'] + target_net_state_dict[key] * (1 - best_individual['tau'])
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