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
        
        previous_neurons = n_observations
        for neurons in layers:
            self.layers.append(nn.Linear(previous_neurons, neurons))
            previous_neurons = neurons
        self.layers.append(nn.Linear(previous_neurons, n_actions))
        
        if len(activation_funcs) != len(layers) + 1:
            raise ValueError("Number of activation functions should match the number of layers + 1 (including the output layer)")
        
        self.activation_funcs = activation_funcs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        for layer, activation in zip(self.layers[:-1], self.activation_funcs[:-1]):
            x = activation(layer(x))

        # Ensuring softmax is applied correctly across the batch
        final_layer = self.layers[-1](x)
        output_activation = self.activation_funcs[-1]
        if output_activation == F.softmax:
            return output_activation(final_layer, dim=0)  # Apply softmax over the correct dimension
        else:
            return output_activation(final_layer)
