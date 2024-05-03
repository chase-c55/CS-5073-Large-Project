import matplotlib.pyplot as plt
import torch
import numpy as np


# other than plt, all variables are local or parameters
# TODO: Add a flag to dump the plot to disk
def plot_durations(reward, episode_durations, avg_reward, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    reward_t = torch.tensor(reward, dtype=torch.float)
    avg_reward = torch.tensor(avg_reward, dtype=torch.float)
    
    running_avg = [[i + 100, sum(episode_durations[i : i + 100]) / 100] for i in range(len(episode_durations) - 100)]
    

    plt.clf()
    plt.plot(durations_t.numpy(), label="Num Steps")
    plt.plot(reward_t.numpy(), label="Reward 10x")
    plt.plot(*zip(*running_avg), label = "Avg Steps")
    plt.plot(avg_reward.numpy(), label="Avg Reward")
    
    if show_result:
        plt.title("Evolved DQN Results over 10,000 Epochs")
        plt.scatter(len(running_avg), running_avg[-1][1], label=f"Final Avg Steps = {running_avg[-1][1]}", c="black")
    else:
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Reward / Duration")
    
    plt.legend()

    # plt.show()
    plt.savefig("./Figure.png")
