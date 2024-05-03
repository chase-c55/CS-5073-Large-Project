# Credit to Chase, this env script was based on graph_color_env and specialized for the A2C environment.

import random
import numpy as np
from typing import Tuple
import networkx as nx
import gymnasium as gym
import matplotlib.pyplot as plt
MAX_STEPS = 1000


def draw_graph(graph: np.ndarray, node_colors: np.ndarray):
    plt.figure(1, clear=True)
    nx_graph = nx.from_numpy_array(graph)
    nx.draw(
        nx_graph,
        with_labels=True,
        font_weight="bold",
        node_color=node_colors,
        font_color="white",
    )
    plt.savefig("./graph.png")


def permute_graph(graph: np.ndarray, permutation: list) -> np.ndarray:
    new_graph = np.zeros(np.array(graph).shape)
    for i in range(len(graph)):
        for j in range(len(graph)):
            new_graph[permutation[i]][permutation[j]] = graph[i][j]
    return new_graph


def calculate_reward(graph: np.ndarray, node_colors: np.ndarray, old_node_colors: np.ndarray) -> Tuple[float, bool]:
    """Calculate the reward for the current graph coloring"""
    reward = 0
    done = False
    num_nodes = len(node_colors)
    correctly_colored_edges = 0

    # Reward/penalty for edges
    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            if graph[i][j] == 1:
                if node_colors[i] != node_colors[j] and node_colors[i] != 0 and node_colors[j] != 0:
                    correctly_colored_edges += 1
                else:
                    reward -= 10  # Penalize for each incorrect edge

    reward += correctly_colored_edges * 10  # Reward for each correct edge

    # Check if all nodes are colored and correctly so
    if correctly_colored_edges == np.sum(graph) / 2:
        reward += 100  # Large reward for complete and correct coloring
        done = True  # Terminate the episode if the graph is correctly colored

    return reward, done


def color_node(node_colors: np.ndarray, node: int, color: int):
    """Color a node in the graph"""
    node_colors[node] = color + 1


graph_5x5 = np.array(
    [
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0],
    ]
)

graph_8x8 = np.array(
    [
        [0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 1],
        [1, 1, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 1],
        [0, 1, 0, 0, 1, 1, 1, 0],
    ]
)

graph_12x12 = np.array(
    [
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    ]
)

graph_16x16 = np.array(
    [
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    ]
)

# TODO: Read this in from a file
input_graph_file = input("Enter the name of the graph file: ")

if input_graph_file == "":
    graph = graph_5x5
else:
    in_graph = nx.read_adjlist(input_graph_file, nodetype=int)
    graph = nx.to_numpy_array(in_graph)

colors_response = input("Enter max number of colors to use (default = # of nodes): ")

if colors_response == "":
    max_colors = len(graph)
else:
    max_colors = int(colors_response)

# Start the node's uncolored (0)
node_colors = np.array([0 for i in range(len(graph))])

# Our action space consists of each node with each possible color
node_actions = [i for i in range(len(graph))]
color_actions = [i for i in range(max_colors)]

# We observe both the graph and its node colors
observations = graph, node_colors


class GraphColoring(gym.Env):
    def __init__(self):
        self.num_steps = 0
        self.graph = graph
        self.node_action_space = gym.spaces.Discrete(len(graph))
        self.color_action_space = gym.spaces.Discrete(max_colors)
        self.observations = observations
        self.observation_space = gym.spaces.Dict(
            {
                "graph": gym.spaces.Box(
                    low=0, high=1, shape=(len(graph), len(graph)), dtype=np.int64
                ),
                "node_colors": gym.spaces.Box(
                    low=0, high=max_colors, shape=(len(graph),), dtype=np.int64
                ),
            }
        )

    def observation(self):
        """Returns an observation of the environment"""
        return {"graph": self.graph, "node_colors": self.node_colors}

    def reset(self):
        """Resets the node colors all to 0 (uncolored) and returns an observation"""
        self.node_colors = np.zeros(len(graph))
        return {"graph": self.graph, "node_colors": self.node_colors}, {}

    def step(self, action: int):
        """Takes a step in the environment given an action"""        
        #print(f"Action: {action}")
        self.num_steps += 1
        done = False

        old_node_colors = np.copy(self.node_colors)
        node_action, color_action = action
        color_node(self.node_colors, node_action, color_action)
        #print(f"Node colors after action: {self.node_colors}")

        reward, done = calculate_reward(self.graph, self.node_colors, old_node_colors)
        #print(f"Done: {done}")

        info = {}

        return self.observation(), reward, done, False, info

    def render(self):
        """Renders the current graph with coloring by outputting to a file

        Returns:
            np.ndarray, np.array: the current graph, the current node colors
        """

        plt.figure(1, clear=True)
        draw_graph(self.graph, self.node_colors, "colored_graph.png")
        return self.graph, self.node_colors

    def permute(self) -> list[int]:
        """Permutes the graph and returns the new node order

        Returns:
            list: the new node order
        """

        node_order = [i for i in range(len(graph))]
        self.graph = graph  # Resets graph so new node order is consistent

        random.shuffle(node_order)
        self.graph = permute_graph(self.graph, node_order)
        self.node_colors = np.zeros(len(self.graph))
        return node_order 
