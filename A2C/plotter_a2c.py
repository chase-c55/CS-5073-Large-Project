import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_durations(reward, episode_durations, avg_reward, show_result=False):
    print("Plotting durations...")
    plt.figure(1)

    print("Converting to tensors...")
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    reward_t = torch.tensor(reward, dtype=torch.float)
    avg_reward_t = torch.tensor(avg_reward, dtype=torch.float)
    
    print("Calculating running average...")
    running_avg = [[i + 100, sum(episode_durations[i : i + 100]) / 100] for i in range(max(0, len(episode_durations) - 100))]
    
    print("Plotting...")
    plt.clf()
    plt.plot(durations_t.numpy(), label="Num Steps")
    if running_avg:
        plt.plot(*zip(*running_avg), label="Avg Steps")
    plt.plot(reward_t.numpy(), label="Reward")
    plt.plot(avg_reward_t.numpy(), label="Avg Reward")
    
    if show_result:
        plt.title("Result")
        if running_avg:
            plt.scatter(len(running_avg), running_avg[-1][1], label=f"Final Avg Steps = {running_avg[-1][1]:.2f}")
    else:
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Reward / Duration")
    
    plt.legend()

    print("Showing plot...")
    plt.show()