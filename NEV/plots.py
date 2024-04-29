import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Domains
TAU_DOMAIN = [0.001, 0.005, 0.01, 0.02, 0.05]
LR_DOMAIN = [0.001, 0.002, 0.005, 0.008, 0.01]
LAYERS_DOMAIN = [1, 2, 3, 4]
NEURONS_DOMAIN = [2, 4, 8, 16, 32, 64]
ACTIVATIONS_DOMAIN = ["relu", "softmax", "sigmoid", "leaky_relu", "tanh"]


def plot_layers(df_layers):
    # Create a counts dictionary which contains count lists for each generation over time, one list in the dict for each option in the layers domain
    counts = {layer: [] for layer in LAYERS_DOMAIN}

    df_layers = df_layers.copy()

    # Convert the string representations of lists in 'layers' column to actual lists
    df_layers['layers_list'] = df_layers['layers'].apply(ast.literal_eval)

    # Calculate the number of layers for each individual
    df_layers['num_layers'] = df_layers['layers_list'].apply(len)

    # Iterate over each generation in the dataset
    for generation in sorted(df_layers['generation'].unique()):
        # Filter the dataframe for the current generation
        gen_df = df_layers[df_layers['generation'] == generation]
        
        # Calculate counts for each layer option in LAYERS_DOMAIN
        for layer in LAYERS_DOMAIN:
            # Count the number of individuals that have the current number of layers
            count = (gen_df['num_layers'] == layer).sum()
            counts[layer].append(count)

    # Plotting
    plt.figure(figsize=(10, 6))
    for layer in LAYERS_DOMAIN:
        plt.plot(counts[layer], label=f'Layers = {layer}')

    plt.title('Number of Individuals by Layer Count over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Count of Individuals')
    plt.legend()
    plt.grid(True)
    # Save plot
    plt.savefig("plots/Layers.png")
    plt.close()

def plot_tau_counts(df_tau):
    # Create a counts dictionary which contains count lists for each generation over time,
    # one list in the dict for each tau value in the TAU_DOMAIN
    counts = {tau: [] for tau in TAU_DOMAIN}

    # Iterate over each generation in the dataset
    for generation in sorted(df_tau['generation'].unique()):
        # Filter the dataframe for the current generation
        gen_df = df_tau[df_tau['generation'] == generation]
        
        # Calculate counts for each tau value in TAU_DOMAIN
        for tau in TAU_DOMAIN:
            # Count the number of individuals that have the current tau value
            count = (gen_df['tau'] == tau).sum()
            counts[tau].append(count)

    # Plotting
    plt.figure(figsize=(10, 6))
    for tau in TAU_DOMAIN:
        plt.plot(counts[tau], label=f'Tau = {tau}')

    plt.title('Count of Individuals by Tau Value over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Count of Individuals')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/Taus.png")
    plt.close()

def plot_lr_counts(df_lr):
    # Create a counts dictionary which contains count lists for each generation over time,
    # one list in the dict for each learning rate value in the LR_DOMAIN
    counts = {lr: [] for lr in LR_DOMAIN}

    # Iterate over each generation in the dataset
    for generation in sorted(df_lr['generation'].unique()):
        # Filter the dataframe for the current generation
        gen_df = df_lr[df_lr['generation'] == generation]
        
        # Calculate counts for each learning rate value in LR_DOMAIN
        for lr in LR_DOMAIN:
            # Count the number of individuals that have the current learning rate
            count = (gen_df['lr'] == lr).sum()
            counts[lr].append(count)

    # Plotting
    plt.figure(figsize=(10, 6))
    for lr in LR_DOMAIN:
        plt.plot(counts[lr], label=f'LR = {lr}')

    plt.title('Count of Individuals by Learning Rate Value over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Count of Individuals')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/LearningRates.png")
    plt.close()

def parse_activation_string(activation_str):
    # Remove the square brackets
    clean_str = activation_str.strip('[]')
    # Split the string by commas
    activations = clean_str.split(', ')
    # Return the list of activation function names
    return activations

def plot_activation_counts(df_activations):
    # Initialize the dictionary to hold counts for each activation
    counts = {activation: [] for activation in ACTIVATIONS_DOMAIN}
    
    # Get the unique generations
    generations = sorted(df_activations['generation'].unique())
    
    # Loop through each generation
    for generation in generations:
        # Filter the dataframe for the current generation
        df_gen = df_activations[df_activations['generation'] == generation]
        
        # Count occurrences of each activation in this generation
        gen_counts = {activation: 0 for activation in ACTIVATIONS_DOMAIN}
        for index, row in df_gen.iterrows():
            # Use the custom parser to get a list of activations
            activations_list = parse_activation_string(row['activations'])
            
            # Increment count for each activation found
            for activation in activations_list:
                if activation in gen_counts:
                    gen_counts[activation] += 1
        
        # Append the counts of each activation to the corresponding list in the counts dictionary
        for activation in ACTIVATIONS_DOMAIN:
            counts[activation].append(gen_counts[activation])
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    for activation in ACTIVATIONS_DOMAIN:
        plt.plot(generations, counts[activation], label=activation)
    
    plt.title('Activation Counts by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Count')
    plt.legend(title='Activations')
    plt.grid(True)
    plt.savefig("plots/Activations.png")
    plt.close()

def create_heatmap(df_neurons):
    df = df_neurons.copy()
    
    # Check if 'layers' needs conversion from string to list
    if isinstance(df.iloc[0]['layers'], str):
        df.loc[:, 'layers'] = df['layers'].apply(ast.literal_eval)
    
    df.loc[:, 'configuration'] = df['layers'].apply(lambda x: ', '.join(map(str, x)))
    
    heatmap_data = df.pivot_table(index='configuration', columns='generation', aggfunc='size', fill_value=0)
    
    # Normalize data by row to show relative frequencies within each configuration across generations
    normalized_heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(normalized_heatmap_data, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Normalized Frequency of Layer Configurations per Generation')
    plt.ylabel('Layer Configuration')
    plt.xlabel('Generation')
    plt.xticks(rotation=45)  # Rotates the generation labels for better visibility
    plt.yticks(rotation=0)   # Ensures configuration labels are horizontal for readability
    plt.tight_layout()       # Adjust layout to make room for label rotations
    plt.savefig("plots/Normalized_Heatmap.png")
    plt.close()

def create_configuration_barchart(df_neurons, top_n=30):
    df = df_neurons.copy()
    
    if isinstance(df.iloc[0]['layers'], str):
        df.loc[:, 'layers'] = df['layers'].apply(ast.literal_eval)
    
    df.loc[:, 'configuration'] = df['layers'].apply(lambda x: ', '.join(map(str, x)))
    
    config_counts = df['configuration'].value_counts()
    
    if top_n:
        config_counts = config_counts.nlargest(top_n)
    
    config_counts = config_counts.sort_values()
    
    # Plotting the bar chart
    plt.figure(figsize=(10, max(8, 0.4 * len(config_counts))))  # Scale height based on number of items
    bars = plt.barh(config_counts.index, config_counts, color='skyblue')
    plt.title('Frequency of Layer Configurations Across All Data')
    plt.xlabel('Frequency')
    plt.ylabel('Configuration')
    plt.tick_params(axis='both', which='major', labelsize=10)  # Adjust font size here
    plt.tight_layout()
    plt.savefig("plots/Configuration.png")
    plt.close()

def create_config_heatmap(df, top_n=20):

    df = df.copy()

    # Standardize the layer configurations and convert them to strings
    df['config'] = df['layers'].apply(lambda x: ', '.join(map(str, sorted(eval(x)))))

    # Count the frequency of each configuration
    config_counts = df['config'].value_counts().to_frame()
    config_counts.columns = ['Frequency']

    # Prepare data for heatmap: one column for frequency, rows for each config
    config_counts['Configuration'] = config_counts.index

    # Select the top_n configurations
    top_config_counts = config_counts.nlargest(top_n, 'Frequency')

    # Apply a logarithmic transformation to the frequency to reduce skewness
    top_config_counts['Log Frequency'] = np.log1p(top_config_counts['Frequency'])

    # Create a heatmap
    plt.figure()
    sns.heatmap(top_config_counts[['Log Frequency']].sort_values(by='Log Frequency', ascending=False),
                annot=True, fmt=".2f", cmap='viridis', cbar=True,
                yticklabels=top_config_counts.sort_values(by='Log Frequency', ascending=False)['Configuration'])
    plt.title(f'Frequency of Top {top_n} Neural Network Configurations (Log Scale)')
    #plt.xlabel('Log Frequency')
    plt.ylabel('Configuration')
    plt.tight_layout()
    plt.savefig("plots/Config_Heatmap.png")
    plt.close()


def main():
    # Load the Results/Chase/generation_data_v2.csv file
    df = pd.read_csv('Results/Chase/generation_data_v2.csv')
    #print(df.head())

    # print the columns of the dataframe
    keys = df.columns

    # Print the number of layers from teh first row
    layer_0 = df['layers'][3]

    # Convert string to list "[1, 2, 3]" to [1,2,3]
    layer_0 = ast.literal_eval(layer_0)

    # Create new dataframes for layers with layers, generation, fitness, and individual_index
    df_layers = df[['layers', 'generation', 'fitness', 'individual_index']]

    # Create a new dataframe for tau with tau, generation, fitness, and indiviudal_index
    df_tau = df[['tau', 'generation', 'fitness', 'individual_index']]

    # Create a new dataframe for learning rate with learning rate, generation, fitness, and individual_index
    df_lr = df[['lr', 'generation', 'fitness', 'individual_index']]

    # Create a new dataframe for activations with activations, generation, fitness, and individual_index
    df_activations = df[['activations', 'generation', 'fitness', 'individual_index']]

    # Create a new dataframe for neurons with neurons, generation, fitness, and individual_index
    df_neurons = df[['layers', 'generation', 'fitness', 'individual_index']]


    plot_layers(df_layers)

    plot_tau_counts(df_tau)

    plot_lr_counts(df_lr)

    plot_activation_counts(df_activations)

    create_heatmap(df_neurons)

    create_configuration_barchart(df_neurons)



    create_config_heatmap(df_neurons)


if __name__ == "__main__":
    main()