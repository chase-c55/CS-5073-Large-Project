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
    plt.show()
    plt.savefig(output_file_name)


if __name__ == "__main__":
    create_permutations_histogram(
        "NEV/Results/Chase/permutations_DQN_v2.csv", "DQN_histogram_v2"
    )

    gen_data_df = pd.read_csv("NEV/Results/Chase/generation_data_v2.csv")

    gen_layers = [
        data
        for data in zip(
            gen_data_df["generation"].values,
            gen_data_df["layers"].values,
            gen_data_df["activations"].values,
        )
    ]
    layers_counts = []
    neurons_counts = []
    active_func_counts = []
    for gen, layers, active_funcs in gen_layers:
        layers_counts.append([gen, len(ast.literal_eval(layers))])
        for layer in ast.literal_eval(layers):
            neurons_counts.append([gen, int(layer)])
        for active_func in active_funcs.replace("[", "").replace("]", "").split(","):
            active_func_counts.append([gen, active_func.strip()])

    plt.scatter(*zip(*layers_counts))
    plt.ylabel("Number of Hidden Layers")
    plt.xlabel("Generation")
    plt.yticks([1, 2, 3, 4])
    plt.title("Number of Hidden Layers over Generations")
    plt.show()

    plt.scatter(*zip(*neurons_counts))
    plt.xlabel("Generation")
    plt.ylabel("Nodes in Layer")
    plt.yticks([2, 4, 8, 16, 32, 64])
    plt.title("Number of Nodes in Hidden Layers over Generations")
    plt.show()

    plt.scatter(*zip(*active_func_counts))
    plt.xlabel("Generation")
    plt.ylabel("Activation Function")
    plt.title("Activation Functions over Generations")
    plt.show()

    plt.scatter(gen_data_df["generation"], gen_data_df["lr"], s=15)
    plt.xlabel("Generation")
    plt.ylabel("Learning Rate")
    plt.yticks([0.001, 0.002, 0.005, 0.008, 0.01])
    plt.title("Learning Rate Values over Generations")
    plt.show()

    plt.scatter(gen_data_df["generation"], gen_data_df["tau"])
    plt.xlabel("Generation")
    plt.ylabel("Tau")
    plt.yticks([0.001, 0.005, 0.01, 0.02, 0.05])
    plt.title("Tau Values over Generations")
    plt.show()
