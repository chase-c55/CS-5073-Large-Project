import matplotlib.pyplot as plt
import torch


# other than plt, all variables are local or parameters
# TODO: Add a flag to dump the plot to disk
def plot_durations(reward, episode_durations, avg_reward, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    reward_t = torch.tensor(reward, dtype=torch.float)
    avg_reward = torch.tensor(avg_reward, dtype=torch.float)
    if show_result:
        plt.clf()
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Reward / Duration")
    
    plt.plot(durations_t.numpy(), label="Num Steps")
    plt.plot(reward_t.numpy(), label="Reward")
    plt.plot(avg_reward.numpy(), label="Avg Reward")
    plt.legend()

    # plt.show()
    plt.savefig("./Figure.png")
