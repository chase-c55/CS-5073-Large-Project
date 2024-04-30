import ast
import pandas as pd
import matplotlib.pyplot as plt


def create_permutations_histogram(file_name: str, output_file_name: str):
    perm_data_df = pd.read_csv(file_name)
    steps = perm_data_df["steps"]

    min_steps = min(steps)
    max_steps = max(steps)
    range = max_steps - min_steps
    mean = round(sum(steps) / len(steps))
    median = steps[len(steps) // 2]
    std_div = pd.Series.std(steps)

    plt.hist(steps)
    plt.text(range * 0.7, 280, f"{'Mean:':<13}{mean:>7}", color="red")
    plt.text(range * 0.7, 260, f"{'Median:':<13}{median:>6}", color="blue")
    plt.text(range * 0.7, 240, f"{'Min:':<16}{min_steps:>8}")
    plt.text(range * 0.7, 220, f"{'Max:':<13}{max_steps:>8}")
    plt.text(range * 0.7, 200, f"{'Std Dev:':<12}{std_div:>5.2f}")

    plt.vlines(mean, 0, 410, colors="red", linewidth=2)
    plt.vlines(median, 0, 410, colors="blue", linewidth=2)
    plt.title("DQN - Histogram of Steps for 1000 Permutations")
    plt.xlabel("Steps")
    plt.ylabel("Number of Permutations")
    plt.savefig(output_file_name)
    plt.show()


if __name__ == "__main__":
    create_permutations_histogram(
        "NEV/Results/Chase/permutations_DQN_v2.csv", "DQN_histogram_v2"
    )
