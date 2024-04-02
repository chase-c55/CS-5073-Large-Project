import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.optim as optim
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from GNN import GCN
from GraphEnv import GraphColoringEnv
from graphs import choose_graph
import random

def greedy_coloring(graph):
    n = len(graph)
    result = [-1] * n

    # Sort the vertices based on their degree
    sorted_vertices = sorted(range(n), key=lambda x: len(graph[x]), reverse=True)

    # Assign the smallest available color to each vertex
    for vertex in sorted_vertices:
        available_colors = set(range(n))
        for neighbor in graph[vertex]:
            if result[neighbor] in available_colors:
                available_colors.remove(result[neighbor])
        result[vertex] = min(available_colors)

    # Calculate the chromatic number
    chromatic_number = max(result) + 1

    return chromatic_number



def visualize_graph(G, color):
    plt.figure(figsize=(4,4))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                     node_color=color, cmap="Set2", node_size=500, font_size=8, font_color="white")
    plt.show()
    
def setup_data(graph_matrix):
    graph = nx.from_numpy_array(graph_matrix) 

    #print(list(graph.edges()))
    #print(nx.to_dict_of_lists(graph))

    # Find the indices of nonzero elements in the adjacency matrix (edges)
    row, col = np.where(graph_matrix)

    edge_index_np = np.vstack((row, col))  # Stack row and col into a numpy array
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)  # Convert to tensor

    num_nodes = graph_matrix.shape[0]
    node_features = torch.eye(num_nodes)

    # Update the Data object
    data = Data(x=node_features, edge_index=edge_index, num_nodes=num_nodes, num_edges=edge_index.shape[1])

    print(data)
    print(data.edge_index)
    print(data.edge_index.t())
    print(data.x)

    num_features = data.x.shape[1]
    num_classes = 4
    print("Num Features: ", num_features)
    print("Num Colors Available: ", num_classes)

    return data, num_features, num_classes, graph, num_nodes

def train(model, data, color, criterion, optimizer):
    epsilon = 0.1
    terminated = False
    while not terminated:
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        print("Out: ", out)
        input("Press Enter to continue...")
        
        loss = criterion(out, color)
        loss.backward()
        optimizer.step()
        
        if random.random() < epsilon:
            action = torch.randint(0, num_classes, (num_nodes,))
        else:
            action = torch.argmax(out, dim=1)
        print("Action: ", action)
        
        state, reward, terminated, _, _ = env.step(action)
        color = torch.tensor(state, dtype=torch.long) 
        #visualize_graph(graph, color)
        print("Reward: ", reward)
        print("Terminated: ", terminated)
        print("Coloring: ", color)
        print("Loss: ", loss.item())
        print("===")

    return color, model

graph_matrix = choose_graph(8)

graph = nx.from_numpy_array(graph_matrix)
print("Graph:", graph)

print("Chromatic Number:", greedy_coloring(graph))

data, num_features, num_classes, graph, num_nodes = setup_data(graph_matrix)


hidden_channels = 16
model = GCN(num_features, num_classes, hidden_channels=16)
print(model)

# Initial colors randomly assigned
color = torch.randint(0, num_classes, (num_nodes,))
visualize_graph(graph, color)


model = GCN(num_features, num_classes, hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

env = GraphColoringEnv(graph_matrix)

state = env.reset()

terminated = False

color, model = train(model, data, color, criterion, optimizer)

out = model(data.x, data.edge_index)

color = torch.argmax(out, dim=1)

visualize_graph(graph, color)

# save the graph with a color attribute indicating its coloring
for i in range(num_nodes):
    graph.nodes[i]["color"] = color[i].item()

# save the graph to a gml file
nx.write_gml(graph, "graph.gml")



