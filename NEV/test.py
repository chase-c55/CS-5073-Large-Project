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

# Example usage
n_observations = 10
n_actions = 2
layers = [64, 64, 128, 256, 128]  # Number of neurons in each hidden layer
activation_funcs = [F.relu, F.relu, F.relu, F.relu, torch.nn.functional.softmax, F.relu]  # Activation functions for each layer, including output

net = DQN(n_observations, n_actions, layers, activation_funcs)

print("#"*50)
print(net)
print("#"*50)
print(net(torch.rand(1, 10)))
print("#"*50)
print(net.forward(torch.rand(1, 10)))
print("#"*50)
print(net.layers)
print("#"*50)