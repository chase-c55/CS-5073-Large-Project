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

from tqdm import tqdm
import numpy as np
import random

env = GraphColoring()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Parameter domains
POPULATION_SIZE = 12
NUM_GENERATIONS = 100
MUTATION_RATE = 0.15
TAU_DOMAIN = [0.001, 0.005, 0.01, 0.02, 0.05]
LR_DOMAIN = [0.001, 0.002, 0.005, 0.008, 0.01]
LAYERS_DOMAIN = [1, 2, 3, 4]
NEURONS_DOMAIN = [2, 4, 8, 16, 32, 64]
ACTIVATIONS_DOMAIN = [F.relu, F.softmax, torch.sigmoid, F.leaky_relu, torch.tanh]
SELECTION_CUT = 0.5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
BATCH_SIZE = 64
NUM_EPISODES = 1000
DISCOUNT_FACTOR = 1.0
MEMORY_SIZE = 10000
OUTPUT_CONTINUOUS = True

def create_individual():
    num_layers = random.choice(LAYERS_DOMAIN)
    layers = random.choices(NEURONS_DOMAIN, k=num_layers)
    activations = random.choices(ACTIVATIONS_DOMAIN, k=num_layers + 1)
    tau = random.choice(TAU_DOMAIN)
    lr = random.choice(LR_DOMAIN)
    return {
        'layers': layers,
        'activations': activations,
        'tau': tau,
        'lr': lr
    }


def crossover(parent_1, parent_2):
    # Choose a random crossover point
    cut = random.randint(1, min(len(parent_1['layers']), len(parent_2['layers'])))
    child_1 = {
        'layers': parent_1['layers'][:cut] + parent_2['layers'][cut:],
        'activations': parent_1['activations'][:cut+1] + parent_2['activations'][cut+1:],
        'tau': random.choice([parent_1['tau'], parent_2['tau']]),
        'lr': random.choice([parent_1['lr'], parent_2['lr']])
    }
    child_2 = {
        'layers': parent_2['layers'][:cut] + parent_1['layers'][cut:],
        'activations': parent_2['activations'][:cut+1] + parent_1['activations'][cut+1:],
        'tau': random.choice([parent_1['tau'], parent_2['tau']]),
        'lr': random.choice([parent_1['lr'], parent_2['lr']])
    }
    return child_1, child_2


def mutate(individual):
    if random.random() < MUTATION_RATE:
        individual['tau'] = random.choice(TAU_DOMAIN)
    if random.random() < MUTATION_RATE:
        individual['lr'] = random.choice(LR_DOMAIN)
    if random.random() < MUTATION_RATE:
        individual['layers'] = random.choices(NEURONS_DOMAIN, k=random.choice(LAYERS_DOMAIN))
    if random.random() < MUTATION_RATE:
        individual['activations'] = random.choices(ACTIVATIONS_DOMAIN, k=len(individual['layers']) + 1)
    return individual

def select_top_fifty(rewards, population):
    # Sort the population by the rewards
    sorted_population = sorted(zip(rewards, population), key=lambda x: x[0], reverse=True)
    # Select the top half of the population
    return [individual for _, individual in sorted_population[:len(population) // 2]]

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state["graph"].flatten(order="C")) + len(state["node_colors"])

population = [create_individual() for _ in range(POPULATION_SIZE)]

generation_metrics = []

# Logging to CSV
csv_file = open('generation_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['generation', 'individual_index', 'fitness', 'tau', 'lr', 'layers', 'activations'])


for generation in tqdm(range(NUM_GENERATIONS), desc="Generations"):
    population_rewards = []
    for index, individual in enumerate(tqdm(population, desc="Individuals", leave=False)):
        policy_net = DQN(n_observations, n_actions, individual['layers'], individual['activations']).to(device)
        target_net = DQN(n_observations, n_actions, individual['layers'], individual['activations']).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.AdamW(policy_net.parameters(), lr=individual['lr'], amsgrad=True)
        memory = ReplayMemory(MEMORY_SIZE)

        steps_done = 0
        episode_durations = []
        reward_history = []
        avg_reward = []

        for i_episode in tqdm(range(NUM_EPISODES), desc="Episodes", leave=False):
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
                reward = torch.tensor([reward], device=device)
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
                    ] * individual['tau'] + target_net_state_dict[key] * (1 - individual['tau'])
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    reward_history.append(reward)


                    num_colors_used = len(set(observation['node_colors']))
                    # print(
                    #     f"Steps: {t} Reward: {reward.item()} Num Colors: {num_colors_used}"
                    # )
                    break

        # Save the individual's performance
        fitness = np.mean(reward_history)
        population_rewards.append(fitness)
        csv_writer.writerow([generation, index, fitness, individual['tau'], individual['lr'], individual['layers'], individual['activations']])

        
    population = select_top_fifty(population_rewards, population)
    # Shuffle the population
    random.shuffle(population)
    # Crossover the top half of the population
    for i in range(0, len(population), 2):
        population.extend(crossover(population[i], population[i + 1]))

    for individual in population:
        mutate(individual)

    generation_metrics.append(max(population_rewards))

# Save the best individual
best_individual = population[0]
with open(f"best_individual-{int(time.time())}.pkl", "wb") as f:
    pickle.dump(best_individual, f)

csv_file.close()

# Plotting generation vs best fitness
plt.figure(figsize=(10, 5))
plt.plot(generation_metrics, label='Best Fitness per Generation')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Best Fitness Evolution Over Generations')
plt.legend()
plt.savefig('fitness_evolution.png')