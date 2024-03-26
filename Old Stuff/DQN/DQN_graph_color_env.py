import random
import numpy as np
from typing import Tuple
import networkx as nx
import gymnasium as gym
import matplotlib.pyplot as plt

REWARD_PER_EDGE = 5


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


def calculate_reward(
    graph: np.ndarray,
    node_colors: np.ndarray,
    old_node_colors: np.ndarray,
) -> Tuple[float, bool]:
    """Calculate the reward for the current graph coloring"""

    reward = -1
    properly_colored = True
    num_adjacent_nodes = 1
    num_colors_used = len(set(node_colors)) - 1  # Don't count 0 as a color
    color_factor = num_colors_used if num_colors_used > 0 else 1  # Avoid divide by 0
    for i in range(len(node_colors)):
        if node_colors[i] != old_node_colors[i]:  # If the color changed
            for j in range(len(graph)):  # Iterate through connected nodes
                if graph[i][j] == 1:  # If j is adjacent
                    num_adjacent_nodes += 1
                    if old_node_colors[i] != 0:  # If it was already colored
                        if node_colors[j] != 0:  # If its neighbor is colored
                            if (
                                old_node_colors[i] != node_colors[j]
                            ):  # If they were correctly colored
                                if node_colors[i] == node_colors[j]:  # Same color now
                                    reward -= 10
                                elif node_colors[i] == 0:  # Took off correct coloring
                                    reward -= 5
                                else:
                                    pass
                            else:  # If they were incorrectly colored
                                if (
                                    node_colors[i] != node_colors[j]
                                    and node_colors[i] != 0
                                ):  # Now correctly colored
                                    reward += 5
                                else:  # Uncolored an incorrectly colored
                                    pass
                        else:  # Neighbor not colored
                            if node_colors[i] == 0:  # Uncolored correctly colored
                                reward -= 5
                            else:  # Changed color
                                pass
                    else:  # If it wasn't colored
                        if node_colors[j] == 0:  # If its neighbor isn't colored
                            reward += 5
                        else:  # If its neighbor was colored
                            if node_colors[i] == node_colors[j]:  # Now same color
                                reward -= 10
                            else:  # Now different colors
                                reward += 5

    # # TODO: Could change this to work on a just a triangle since it's symmetric
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i][j] == 1:
                # If the nodes are connected, we want them to have different colors
                if (
                    node_colors[i] != node_colors[j]
                    and node_colors[i] != 0
                    and node_colors[j] != 0
                ):
                    pass  # reward += np.exp(max_colors - color_factor)
                elif (
                    node_colors[i] == node_colors[j]
                    and node_colors[i] != 0
                    and node_colors[j] != 0
                ):
                    properly_colored = False
                    # reward -= 1.5 ** (max_colors - color_factor)
                else:
                    properly_colored = False
    return reward / color_factor, properly_colored


def color_node(node_colors: np.ndarray, action: int):
    """Color a node in the graph"""
    node = action // max_colors
    color = action % max_colors
    node_colors[node] = color


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

max_colors = len(graph)  # TODO: Make this an input parameter
# Start the node's uncolored (0)
node_colors = np.array([0 for i in range(len(graph))])
draw_graph(graph, node_colors)

# node_order = [i for i in range(len(graph))]
# print("Old node order: ", node_order)
# print(graph)

# random.shuffle(node_order)
# print("New node order: ", node_order)
# permuted_graph = permute_graph(graph, node_order)
# print(permuted_graph)

# Our action space consists of each node with each possible color
actions = [i for i in range(max_colors * len(graph))]

# We observe both the graph and its node colors
observations = graph, node_colors


class GraphColoring(gym.Env):
    def __init__(self):
        self.num_steps = 0
        self.actions = actions
        self.action_space = gym.spaces.Discrete(max_colors * len(graph))
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
        return {"graph": self.graph, "node_colors": self.node_colors}

    def reset(self):
        """Reset the node colors to all 0s"""
        self.graph = graph
        self.node_colors = np.zeros(len(graph))
        return self.observation(), {}

    def step(self, action):
        self.num_steps += 1
        old_node_colors = np.copy(self.node_colors)
        color_node(self.node_colors, action)

        reward, done = calculate_reward(self.graph, self.node_colors, old_node_colors)

        info = {}

        return self.observation(), reward, done, False, info
