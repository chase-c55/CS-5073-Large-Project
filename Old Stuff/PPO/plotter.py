import csv
import matplotlib.pyplot as plt


permuations_csv_files = [
    ("PPO_200k_16x_50k_8_nodes_v2.csv", "16 Permuations - 50k"),
    # ("PPO_200k_8x_50k_8_nodes.csv", "8 Permutation - 50k"),
    # ("PPO_200k_4x_50k_8_nodes.csv", "4 Permutations - 50k"),
    # ("PPO_200k_8_nodes.csv", "Original - 200k"),
]

nodes_csv_files = [
    ("PPO_200000_steps_8_nodes.csv", "200k Steps - 8 Nodes"),
    ("PPO_200000_steps_12_nodes.csv", "200k Steps - 12 Nodes"),
    ("PPO_200000_steps_16_nodes.csv", "200k Steps - 16 Nodes"),
]

plt.style.use("tableau-colorblind10")


def plot_dqn(input_file_name: str, plt_title: str, out_file_name: str):
    """Plots the results of a DQN run.

    Args:
        input_file_name (str): name of the csv file to read from
        plt_title (str): title of the plot
        out_file_name (str): name of the file to save the plot to
    """

    rewards = []
    steps = []
    with open(input_file_name, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            reward = float(row["reward"]) if row["reward"] != "" else 0
            ep_mean_len = float(row["steps"]) if row["steps"] != "" else 0
            rewards.append(reward)
            steps.append(ep_mean_len)

    steps = [steps[i] for i in range(len(steps)) if steps[i] != 0]
    rewards = [rewards[i] * 10 for i in range(len(rewards)) if rewards[i] != 0]
    avg_steps = [sum(steps[i : i + 100]) / 100 for i in range(len(steps) - 100)]

    plt.figure(clear=True)
    plt.plot(steps, label="Steps")
    plt.plot(rewards, label="Reward (10x)")
    plt.plot(avg_steps, label="Avg Steps")
    plt.plot(
        len(steps),
        avg_steps[-1],
        marker="o",
        label="Final Avg Steps: " + str(avg_steps[-1]),
    )

    plt.xlabel("Episodes")
    plt.legend()
    plt.title(plt_title)
    plt.savefig(out_file_name)


def plot_permutations(permuations_csv_files: list):
    plt.figure(1, clear=True)
    plt.figure(2, clear=True)
    for file_name, label_name in permuations_csv_files:
        ep_reward_means = []
        ep_len_means = []
        permutation_breakkpoints = []
        with open(file_name, newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=",")
            for i, row in enumerate(reader):
                reward = (
                    float(row["rollout/ep_rew_mean"])
                    if row["rollout/ep_rew_mean"] != ""
                    else 0
                )
                ep_mean_len = (
                    float(row["rollout/ep_len_mean"])
                    if row["rollout/ep_len_mean"] != ""
                    else 0
                )
                ep_reward_means.append(reward)
                ep_len_means.append(ep_mean_len)
                if row["time/iterations"] == "1":
                    permutation_breakkpoints.append(i + 1)

        ep_len_means = [
            ep_len_means[i] for i in range(len(ep_len_means)) if ep_len_means[i] != 0
        ]
        ep_reward_means = [
            ep_reward_means[i]
            for i in range(len(ep_reward_means))
            if ep_reward_means[i] != 0
        ]

        permutation_breakkpoints.append(len(ep_len_means)) # add the final endpoint
        
        for i in reversed(permutation_breakkpoints):
            plt.figure(1)
            plt.plot(ep_len_means[:i])
            plt.figure(2)
            plt.plot(ep_reward_means[:i])

    plt.figure(1)
    plt.xlabel("Time Steps (x2048)")
    plt.ylabel("Episodes Length Mean")
    plt.yscale("log")
    plt.xscale("linear")
    plt.text(len(ep_len_means) + 30, ep_len_means[-1], str(ep_len_means[-1]))
    plt.title("Episode Length vs Time Steps for 16 Graph Permutations")
    plt.savefig("Plots/permutations_ep_length.png")

    plt.figure(2)
    plt.xlabel("Time Steps (x2048)")
    plt.ylabel("Episode Reward Mean")
    plt.yscale("log")
    plt.xscale("linear")
    plt.title("Reward vs Time Steps for 16 Graph Permutations")
    plt.savefig("Plots/permutations_ep_reward.png")


def plot_8_12_16_nodes_len_rew(nodes_csv_files: list):
    """Plot the episode length and reward for 8, 12, and 16 nodes.

    Args:
        nodes_csv_files (list): a list of tuples containing the file name and label
    """

    plt.figure(1, clear=True)
    plt.figure(2, clear=True)
    for file_name, label_name in nodes_csv_files:
        ep_reward_means = []
        ep_len_means = []
        with open(file_name, newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=",")
            for row in reader:
                reward = (
                    float(row["rollout/ep_rew_mean"])
                    if row["rollout/ep_rew_mean"] != ""
                    else 0
                )
                ep_mean_len = (
                    float(row["rollout/ep_len_mean"])
                    if row["rollout/ep_len_mean"] != ""
                    else 0
                )
                ep_reward_means.append(reward)
                ep_len_means.append(ep_mean_len)

        ep_len_means = [
            ep_len_means[i] for i in range(len(ep_len_means)) if ep_len_means[i] != 0
        ]
        ep_reward_means = [
            ep_reward_means[i]
            for i in range(len(ep_reward_means))
            if ep_reward_means[i] != 0
        ]

        plt.figure(1)
        plt.plot(ep_len_means, label=label_name)
        plt.text(103, ep_len_means[-1], str(ep_len_means[-1]))
        plt.figure(2)
        plt.plot(ep_reward_means, label=label_name)

    plt.figure(1)
    plt.yscale("log")
    plt.ylabel("Episodes Length Mean")
    plt.xlabel("Time Steps (x2048)")
    plt.legend()
    plt.title("Episode Length vs Time Steps for 8, 12, 16 Nodes")
    plt.savefig("Plots/8_12_16_nodes_vs_length.png")

    plt.figure(2)
    plt.ylabel("Episode Reward Mean")
    plt.xlabel("Time Steps (x2048)")
    plt.legend()
    plt.title("Episode Reward vs Time Steps for 8, 12, 16 Nodes")
    plt.savefig("Plots/8_12_16_nodes_vs_reward.png")


def make_steps_histogram(input_file_name: str, title: str, out_file_name: str):
    """Makes a histogram of the steps taken to solve a graph.

    Args:
        file_name (str): the name of the csv file to read from
    """

    steps = []
    with open(input_file_name, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            steps.append(int(row["steps"]))

    plt.figure(1, clear=True)
    max_steps = max(steps)
    range = max_steps if max_steps < 15_000 else 15_000
    plt.hist(steps, bins=20, range=(0, range), color="purple")

    mean = round(sum(steps) / len(steps))
    median = steps[len(steps) // 2]
    plt.text(range * 0.7, 200, f"Mean Steps: {mean}", color="red")
    plt.vlines(mean, 0, 410, colors="red", linewidth=2)
    plt.text(range * 0.7, 220, f"Median Steps: {median}", color="blue")
    plt.vlines(median, 0, 410, colors="blue", linewidth=2)
    plt.text(range * 0.7, 240, f"Max Steps: {max_steps}")
    plt.text(range * 0.7, 260, f"Min Steps: {min(steps)}")

    plt.ylabel("Frequency")
    plt.xlabel("Number of Steps")
    plt.title(title)
    plt.savefig(out_file_name)


if __name__ == "__main__":
    plot_permutations(permuations_csv_files)
    make_steps_histogram(
        "1000_perm_steps_200k_8_nodes.csv",
        "Steps Taken to Solve - 1k Permutations",
        "Plots/perm_steps_histogram_orig.png",
    )
    make_steps_histogram(
        "1000_perm_steps_200k_16_perm_50k_8_nodes.csv",
        "Steps Taken to Solve w/ Extra Training - 1k Permutations",
        "Plots/perm_steps_histogram_16_50k.png",
    )
    plot_8_12_16_nodes_len_rew(nodes_csv_files)
    plot_dqn(
        "DQN_5000_steps_8_nodes.csv",
        "DQN Episode Length and Reward over 5000 Episodes - 8 Nodes",
        "Plots/dqn_5000_8_nodes.png",
    )
    plot_dqn(
        "DQN_10000_steps_5_nodes.csv",
        "DQN Episode Length and Reward over 10000 Episodes - 5 Nodes",
        "Plots/dqn_10000_5_nodes.png",
    )
