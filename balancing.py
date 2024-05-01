from sklearn.datasets import fetch_covtype
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

import preprocessing


def plot_distribution(x, y, class_names, dataset_name):

    # Convert the dataset into a DataFrame
    dataset_df = pd.DataFrame(data=x, columns=class_names)

    # Add the target variable 'Cover_Type' to the DataFrame
    dataset_df['Cover_Type'] = y

    # Count occurrences of each class label
    class_distribution = dataset_df['Cover_Type'].value_counts()

    # Sort the class distribution by the class label (index) instead of count
    class_distribution_sorted = class_distribution.sort_index()

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    class_distribution_sorted.plot(kind='bar')
    plt.title(dataset_name + '-dataset klassdistribution')
    plt.xlabel('Klassnamn')
    plt.ylabel('Frekvens')
    plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines
    plt.tight_layout()
    plt.show()


def undersample_dataset(dataset_name, random_state=42):
    x, y, class_names = preprocessing.load_dataset(dataset_name, None, random_state)

    # Calculate the frequency of each class
    class_frequencies = Counter(y)

    # Determine the minimum frequency
    min_frequency = min(class_frequencies.values())

    # Set the undersampling strategy to match the minimum class frequency
    undersample_strategy = {class_label: min_frequency for class_label in class_frequencies}

    undersampler = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=random_state)

    # Apply the undersampler to the dataset
    x_resampled, y_resampled = undersampler.fit_resample(x, y)

    return x_resampled, y_resampled

