import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
data = pd.read_csv('permutations_A2C.csv')

# Plotting the number of steps per iteration in a line plot
plt.figure(figsize=(10, 5))
plt.plot(data['steps'], label='Number of Steps')
plt.title('A2C Graph Coloring Problem: Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.legend()
plt.show()

# Creating a histogram of the steps to see the distribution
plt.figure(figsize=(10, 5))
plt.hist(data['steps'], bins=30, alpha=0.75, color='blue')
plt.title('Distribution of Steps')
plt.xlabel('Steps')
plt.ylabel('Frequency')
plt.show()
