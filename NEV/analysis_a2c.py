import pickle
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
from itertools import count

from a2c import ActorCritic
from action_a2c import *                # select_action
from plotter_a2c import *               # plot_durations
from A2C_graph_color_env import *   # draw_graph, permute_graph, calculate_reward, color_node, GraphColoring

# set hyperparameters
BATCH_SIZE = 64
NUM_EPISODES = 10_000
DISCOUNT_FACTOR = 1.0
OUTPUT_CONTINUOUS = False
NUM_PERMUTATIONS = 1000
HIDDEN_SIZE = 128
MAX_STEPS = 1000

# initialize environment
env = GraphColoring()
n_node_actions = env.node_action_space.n
n_color_actions = env.color_action_space.n
state, info = env.reset()
n_observations = len(state["graph"].flatten(order="C")) + len(state["node_colors"])

# get device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Initialize the ActorCritic model
model = ActorCritic(n_observations, n_node_actions, n_color_actions, HIDDEN_SIZE).to(device)
optimizer = optim.Adam(model.parameters())

# Initialize variables for tracking training progress
steps_done = 0
episode_durations = []
reward_history = []
avg_reward = []

# Training loop
for i_episode in tqdm(range(NUM_EPISODES), desc="Episodes", colour="blue"):
    # Reset the environment
    state, info = env.reset()
    graph_tensor = torch.tensor(
        state["graph"].flatten(order="C"), dtype=torch.float, device=device
    )
    node_colors_tensor = torch.tensor(
        state["node_colors"], dtype=torch.float, device=device
    )
    state = torch.cat((graph_tensor, node_colors_tensor))
    
    # Episode loop
    for t in count():
        action, log_prob, value = select_action(model, state, device)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step: {t}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        # Manage state and reward
        reward_tensor = torch.tensor([reward], device=device)
        reward_history.append(reward_tensor.item())  # Collecting raw reward for each step
        done = terminated or truncated

        if done:
            episode_durations.append(t + 1)
            avg_reward.append(np.mean(reward_history))  # Update average reward based on all collected rewards
            print(f"Episode {i_episode}: Duration = {t + 1}, Avg Reward = {avg_reward[-1]}")
            break

# Plot the training results
print("Plotting training results...")
print(f"reward_history: {reward_history}")
print(f"episode_durations: {episode_durations}")
print(f"avg_reward: {avg_reward}")
plot_durations(reward_history, episode_durations, avg_reward, True)

# Save the trained model
torch.save(model, "./trained_model.pt")
print("Saved Model")

# Load the trained model for permutation testing
trained_model = torch.load("./trained_model.pt")
env = GraphColoring()
n_node_actions = env.node_action_space.n
n_color_actions = env.color_action_space.nstate, info = env.reset()
n_observations = len(state["graph"].flatten(order="C")) + len(state["node_colors"])

# Prepare the CSV file for writing permutation test results
import csv

csv_file = open("permutations_A2C.csv", "w")
writer = csv.writer(csv_file)
writer.writerow(["iteration", "node_order", "steps"])

# Default node order
node_order = [i for i in range(len(env.graph))]

# Permutation testing loop
try:
    for iteration in tqdm(range(NUM_PERMUTATIONS), desc="Permutation Tests", colour="blue"):
        state, info = env.reset()
        graph_tensor = torch.tensor(state["graph"].flatten(order="C"), dtype=torch.int64, device=device)
        node_colors_tensor = torch.tensor(state["node_colors"], dtype=torch.int64, device=device)
        state = torch.concat((graph_tensor, node_colors_tensor))
        
        for step in count(1):
            action, log_prob, value = select_action(trained_model, state, device)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            print(f"Iteration: {iteration}, Step: {step}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            
            reward = torch.tensor([reward], device=device) * 10
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_graph_tensor = torch.tensor(observation["graph"].flatten(order="C"), dtype=torch.int64, device=device)
                next_node_colors_tensor = torch.tensor(observation["node_colors"], dtype=torch.int64, device=device)
                next_state = torch.concat((next_graph_tensor, next_node_colors_tensor))
            
            state = next_state
            
            if done or step >= MAX_STEPS:
                writer.writerow([iteration, node_order, step + 1])
                print(f"Permutation {iteration} completed at step {step}.")
                break

except Exception as e:
    print(f"Exception during permutation testing: {e}")
finally:
    csv_file.close()
    print("Permutation testing data saved and file closed.")