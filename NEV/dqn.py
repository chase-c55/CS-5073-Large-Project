import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable

class DQN(nn.Module):
    def __init__(
            self, 
            n_observations: int, 
            n_actions: int, 
            layers: List[int], 
            activation_funcs: List[Callable[[torch.Tensor], torch.Tensor]]
            ):
        
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        previous_neurons = n_observations
        
        # Creating hidden layers
        for neurons in layers:
            self.layers.append(nn.Linear(previous_neurons, neurons))
            previous_neurons = neurons
        
        # Output layer
        self.layers.append(nn.Linear(previous_neurons, n_actions))
        
        # Check if the correct number of activation functions has been provided
        if len(activation_funcs) != len(layers) + 1:  # +1 for the output layer
            raise ValueError("Number of activation functions should match the number of layers + 1 (including the output layer)")
        
        self.activation_funcs = activation_funcs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()  # Ensure input is float
        
        # Apply each activation function to its corresponding layer
        for layer, activation in zip(self.layers[:-1], self.activation_funcs[:-1]):
            x = activation(layer(x))

        if self.activation_funcs[-1] == torch.nn.functional.softmax:
            return self.activation_funcs[-1](self.layers[-1](x), dim=1)
        
        # Output layer with its activation function
        return self.activation_funcs[-1](self.layers[-1](x))