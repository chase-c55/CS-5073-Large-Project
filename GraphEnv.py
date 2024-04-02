import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GraphColoringEnv(gym.Env):
    """A simple graph coloring environment for reinforcement learning."""
    metadata = {'render.modes': ['human']}

    def __init__(self, graph_matrix):
        super(GraphColoringEnv, self).__init__()
        self.graph = graph_matrix
        self.num_nodes = graph_matrix.shape[0]
        self.action_space = spaces.Discrete(self.num_nodes * self.num_nodes)  # Node, Color
        self.observation_space = spaces.Box(low=0, high=self.num_nodes, shape=(self.num_nodes,), dtype=np.int32)

        self.state = np.zeros(self.num_nodes, dtype=np.int32)  
        self.reset()

    def reset(self):
        self.state = np.zeros(self.num_nodes, dtype=np.int32)  # Reset coloring
        return self.state
    
    def check_conflict(self, node_id):
        # Iterate through all edges to check for conflict
        for i in range(self.num_nodes):
            # Check if there is an edge between node_id and i
            if self.graph[node_id, i] == 1:  # Assuming graph is represented by an adjacency matrix
                # Check if the adjacent node i has the same color as node_id
                if self.state[node_id] == self.state[i]:
                    return True  # Conflict found
        return False  # No conflict

    def step(self, actions):
        self.state = np.zeros(self.num_nodes, dtype=np.int32)  # Assuming 0 indicates uncolored

        terminated = False  
        reward = 0  

        # Apply the colors to each node
        for node_id, color_id in enumerate(actions):
            # Assuming the actions tensor is valid and does not require validation for this example
            # Update the node's color directly
            self.state[node_id] = color_id

        # After all nodes are colored, check the entire graph for conflicts
        conflict_count = 0
        for node_id in range(self.num_nodes):
            if self.check_conflict(node_id):
                conflict_count += 1

        # Calculate reward based on the number of conflicts
        if conflict_count > 0:
            reward = -conflict_count  # Penalize based on the number of conflicts
        else:
            reward = self.num_nodes  # Reward for a completely correct coloring

        # Determine if the episode is done
        # This could be based on achieving a correct coloring or some other criterion
        if conflict_count == 0:
            terminated = True

        truncated = False  # This would depend on your episode truncation logic
        info = {"conflict_count": conflict_count}  # Provide conflict count for debugging

        return self.state, reward, terminated, truncated, info

    
    def render(self, mode='human'):
        pass

