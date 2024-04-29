import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def tau_diversity(df):
    return df['tau'].value_counts()


def lr_diversity(df):
    return df['lr'].value_counts()


def layers_diversity(df):
    return df['layers'].value_counts()


def activations_diversity(df):
    return df['activations'].value_counts()


def overall_diversity_summary(df):
    diversity_summary = {
        'tau_diversity': tau_diversity(df),
        'lr_diversity': lr_diversity(df),
        'layers_diversity': layers_diversity(df),
        'activations_diversity': activations_diversity(df)
    }
    return diversity_summary


def plot_tau_diversity(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='tau', data=df, palette='viridis')
    plt.title('Diversity of Tau Values')
    plt.xlabel('Tau')
    plt.ylabel('Frequency')
    plt.savefig('plots/tau_diversity.png')
    plt.close()

def plot_lr_diversity(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='lr', data=df, palette='viridis')
    plt.title('Diversity of Learning Rate (lr) Values')
    plt.xlabel('Learning Rate')
    plt.ylabel('Frequency')
    plt.savefig('plots/lr_diversity.png')
    plt.close()


# Read your data into a DataFrame
data = pd.read_csv('Results/Chase/generation_data_v2.csv')

# Print the overall diversity summary
print(overall_diversity_summary(data))

diversity_summary = overall_diversity_summary(data)

# Convert dict of df to a DataFrame
for key, value in diversity_summary.items():
    diversity_summary[key] = value.reset_index()
    diversity_summary[key].columns = ['Value', 'Frequency']

# Print the diversity summary
for key, value in diversity_summary.items():
    print(key)
    print(value)

# Save the diversity summary to a markdown file
for key, value in diversity_summary.items():
    value.to_markdown(f'Results/Chase/{key}.md', index=False)


# Plot the diversity of tau values
plot_tau_diversity(data)

# Plot the diversity of learning rates
plot_lr_diversity(data)


