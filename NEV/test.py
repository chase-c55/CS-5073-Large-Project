import random
import numpy as np
import torch
import torch.nn.functional as F

TAU_DOMAIN = [0.001, 0.005, 0.01, 0.02, 0.05]
LR_DOMAIN = [0.001, 0.002, 0.005, 0.008, 0.01]
LAYERS_DOMAIN = [1, 2, 3, 4]
NEURONS_DOMAIN = [2, 4, 8, 16, 32, 64]
ACTIVATIONS_DOMAIN = [F.relu, F.softmax, torch.sigmoid, F.leaky_relu, torch.tanh]
POPULATION_SIZE = 12

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


population = [create_individual() for _ in range(POPULATION_SIZE)]

print(population)

# shuffle the population
random.shuffle(population)
print(population)