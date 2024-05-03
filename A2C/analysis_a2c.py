# main driver file, based on and inspired by Chase's analysis.py file, specialized for A2C.

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
print("training loop started")
for i_episode in tqdm(range(NUM_EPISODES), desc="Episodes", colour="blue"):
    state, info = env.reset()

    # Debug: Print the shapes of graph and node colors tensors
    #print(f"Graph shape: {state['graph'].shape}, Node colors shape: {state['node_colors'].shape}")

    # If the graph is already a tensor and flat, you don't need to flatten again
    if not torch.is_tensor(state["graph"]):
        graph_tensor = torch.tensor(state["graph"], dtype=torch.float, device=device)
    else:
        graph_tensor = state["graph"].to(device).float()

    if not torch.is_tensor(state["node_colors"]):
        node_colors_tensor = torch.tensor(state["node_colors"], dtype=torch.float, device=device)
    else:
        node_colors_tensor = state["node_colors"].to(device).float()

    state = torch.cat((graph_tensor.flatten(), node_colors_tensor))  # Flattening only when needed

    
    # Episode loop
    for t in count():
        #print(f"Episode loop iteration: {t}")
        # Select an action using the ActorCritic model
        prev_state = state.clone()
        #print("Selecting action...")
        if random.random() < 0.1:  # 10% chance of selecting a random action
            node_action = torch.randint(0, n_node_actions, (1,), device=device, dtype=torch.long)
            color_action = torch.randint(0, n_color_actions, (1,), device=device, dtype=torch.long)
            action = (node_action, color_action)
            node_log_prob, color_log_prob, value = evaluate_action(model, state, action, device)
            log_prob = node_log_prob + color_log_prob
        else:
            action, log_prob, value = select_action(model, state, device)
            #print(f"Action selected: {action}")      
        
        # Take a step in the environment
        #print("Taking a step in the environment...")
        observation, reward, terminated, truncated, info = env.step(action)
        #print(f"Step taken. Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        reward = torch.tensor([reward], device=device) * 10
        done = terminated or truncated
        
        # Prepare the next state
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
        
        # Compute the advantage and update the model
        advantage = reward + (1 - done) * DISCOUNT_FACTOR * model.get_value(next_state).detach() - value
        model.update(advantage, log_prob, value)
        
        # Update the current state
        state = next_state
        #print(f"prev_state: {prev_state}")
        #print(f"state: {state}")
        #print(f"done: {done}")
        #if state is None or prev_state is None:
        #    print("Either prev_state or state is None. Breaking the loop.")
        #    break
        #else:
        #    # Check if the state has not changed
        #    if torch.all(prev_state.eq(state)):
        #       #print("State has not changed after action. Breaking the loop.")
        #        break
        
        # Check if the episode is done
        if done:
            #print("Episode is done. Breaking the loop.")
            episode_durations.append(t + 1)
            reward_history.append(reward)
            avg_reward.append(np.mean([reward.item() for reward in reward_history]))

            if OUTPUT_CONTINUOUS:
                print("Plotting durations...")  # Print a message before plotting
                plot_durations(reward_history, episode_durations, avg_reward)
            
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

csv_file_path = 'testing_results.csv'
csv_file = open(csv_file_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Episode', 'Average Reward', 'Episode Duration'])

# Default node order
node_order = [i for i in range(len(env.graph))]

# Permutation testing loop
try:
    done = False
    for iteration in tqdm(
        range(NUM_PERMUTATIONS), desc="Permutation Tests", colour="blue"
    ):
        # Prepare the initial state
        graph_tensor = torch.tensor(
            state["graph"].flatten(order="C"), dtype=torch.int64, device=device
        )
        node_colors_tensor = torch.tensor(
            state["node_colors"], dtype=torch.int64, device=device
        )
        state = torch.concat((graph_tensor, node_colors_tensor))
        
        # Step loop
        for step in count(1):
            # Select an action using the trained model
            #print("Selecting action...")
            if random.random() < 0.1:  # 10% chance of selecting a random action
                node_action = torch.randint(0, n_node_actions, (1,), device=device, dtype=torch.long)
                color_action = torch.randint(0, n_color_actions, (1,), device=device, dtype=torch.long)
                action = (node_action, color_action)
                log_prob, value = model.evaluate_action(state, action)
            else:
                action, log_prob, value = select_action(model, state, device)
            #print(f"Action selected: {action}")            
            # Take a step in the environment
            #print("Taking a step in the environment...")
            observation, reward, terminated, truncated, _ = env.step(action.item())
            #print(f"Step taken. Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            reward = torch.tensor([reward], device=device) * 10
            done = terminated or truncated
            
            # Prepare the next state
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
            
            # Update the current state
            state = next_state
            
            # Check if the episode is done
            if done or step >= MAX_STEPS:
                csv_writer.writerow([i_episode, avg_reward[-1], episode_durations[-1]])
                state, info = env.reset()
                node_order = env.permute()
                break

except Exception as e:
    print(f"Exception during permutation testing: {e}")
finally:
    csv_file.close()
    print("Permutation testing data saved and file closed.")