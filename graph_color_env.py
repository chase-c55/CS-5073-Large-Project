import random
import numpy as np
import networkx as nx
import gymnasium as gym
from typing import Tuple
import matplotlib.pyplot as plt


def draw_graph(graph: np.ndarray, node_colors: np.ndarray, name: str = "graph.png"):
    """Draws a given graph by outputing it to a file

    Args:
        graph (np.ndarray): the graph to be drawn
        node_colors (np.ndarray): the list of node colors
        name (str, optional): the filename to write to. Defaults to "graph.png".
    """

    plt.figure(1, clear=True)
    nx_graph = nx.from_numpy_array(graph)
    nx.draw(
        nx_graph,
        with_labels=True,
        font_weight="bold",
        node_color=node_colors,
        font_color="white",
    )
    plt.savefig("./" + name)


def permute_graph(graph: np.ndarray, permutation: list) -> np.ndarray:
    """Creates a random permutation of a given graph

    Args:
        graph (np.ndarray): the graph to be permuted
        permutation (list): the new list of node orders

    Returns:
        np.ndarray: the permuted graph
    """

    new_graph = np.zeros(np.array(graph).shape)

    # Permutes the graph by reassigning the rows and columns
    for i in range(len(graph)):
        for j in range(len(graph)):
            new_graph[permutation[i]][permutation[j]] = graph[i][j]
    return new_graph


def calculate_reward(
    graph: np.ndarray,
    node_colors: np.ndarray,
    old_node_colors: np.ndarray,
) -> Tuple[float, bool]:
    """Calculates the reward for a graph based on the latest color action and checks if the graph is properly colored

    Args:
        graph (np.ndarray): the graph
        node_colors (np.ndarray): the list of node colors
        old_node_colors (np.ndarray): the list of previous node colors

    Returns:
        Tuple[float, bool]: the reward and a bool indicating if it the graph is properly colored
    """

    reward = -1  # Starts at -1 to discourage taking extra steps
    properly_colored = True
    num_colors_used = len(set(node_colors)) - 1  # Don't count 0 as a color
    color_factor = num_colors_used if num_colors_used > 0 else 1  # Avoids divide by 0

    for i in range(len(node_colors)):
        if node_colors[i] != old_node_colors[i]:  # If the color changed
            for j in range(len(graph)):  # Iterate through connected nodes
                if graph[i][j] == 1:  # If j is adjacent
                    if old_node_colors[i] != 0:  # If it was already colored
                        if node_colors[j] != 0:  # If its neighbor is colored
                            if (
                                old_node_colors[i] != node_colors[j]
                            ):  # If they were correctly colored
                                if (
                                    node_colors[i] == node_colors[j]
                                ):  # If same color now
                                    reward -= 10
                                elif node_colors[i] == 0:  # If removed correct coloring
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
                            if (
                                node_colors[i] == 0
                            ):  # Uncolored previously correctly colored
                                reward -= 5
                            else:  # Changed color
                                pass
                    else:  # If it wasn't colored
                        if node_colors[j] == 0:  # If its neighbor isn't colored
                            reward += 5
                        else:  # If its neighbor was colored
                            if node_colors[i] == node_colors[j]:  # If now same color
                                reward -= 10
                            else:  # If now different colors
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
                    pass
                elif (
                    node_colors[i] == node_colors[j]
                    and node_colors[i] != 0
                    and node_colors[j] != 0
                ):
                    properly_colored = False
                else:
                    properly_colored = False

    # Divide reward by number of colors used to encourage using fewer colors
    return reward / color_factor, properly_colored


def color_node(node_colors: np.ndarray, action: int):
    """Colors a given node of a graph given an action

    Args:
        node_colors (np.ndarray): the list of node colors
        action (int): the action to take
    """

    # To find which node to color, we divide the action by the number of colors and take the floor.
    node = action // max_colors
    # To find which color to color the node, we take the action mod the number of colors.
    color = action % max_colors

    node_colors[node] = color


graph_4x4 = np.array(
    [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ]
)

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


graph = graph_8x8  # TODO: Could read this in from a file
max_colors = len(graph)  # TODO: Make this an input parameter

# Start the node's uncolored (0)
node_colors = np.array([0 for i in range(len(graph))])

# Our action space consists of each node with each possible color
actions = [i for i in range(max_colors * len(graph))]

# We observe both the graph and its node colors
observations = graph, node_colors


class GraphColoring(gym.Env):
    """This class implements a gymnasium environment for graph coloring"""

    def __init__(self):
        self.num_steps = 0
        self.graph = graph
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
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(len(graph) * len(graph) + len(graph),), dtype=np.int64
        )

    def observation(self):
        """Returns an observation of the environment"""
        return np.concatenate((self.graph.flatten(), self.node_colors))

    def reset(self):
        """Resets the node colors all to 0 (uncolored) and returns an observation"""
        self.node_colors = np.zeros(len(graph))
        return self.observation(), {}

    def render(self):
        """Renders the current graph with coloring by outputting to a file

        Returns:
            np.ndarray, np.array: the current graph, the current node colors
        """

        plt.figure(1, clear=True)
        draw_graph(self.graph, self.node_colors, "colored_graph.png")
        return self.graph, self.node_colors

    def step(self, action: int):
        """Takes a step in the environment given an action"""
        self.num_steps += 1
        old_node_colors = np.copy(self.node_colors)
        color_node(self.node_colors, action)

        reward, done = calculate_reward(self.graph, self.node_colors, old_node_colors)

        info = {}

        return self.observation(), reward, done, False, info

    def permute(self) -> list[int]:
        """Permutes the graph and returns the new node order

        Returns:
            list: the new node order
        """

        node_order = [i for i in range(len(graph))]
        self.graph = graph  # Resets graph so new node order is consistent

        random.shuffle(node_order)
        self.graph = permute_graph(self.graph, node_order)
        return node_order
