# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-05-02

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import evaluation


def load_dataset(dataset_name, samples, random_state=42):
    """
    Load a specified dataset. Return the data, labels and class name.

    :param dataset_name: Name of the dataset to load (iris or mnist)
    :param samples: Number of samples.
    :param random_state: Controls shuffling of the dataset.

    :return: x: feature data of the dataset,
    :return: y: target or label data corresponding to x and tha name of the classes in the dataset.
    """

    if dataset_name == 'MNIST':
        mnist = fetch_openml('mnist_784', version=1)
        x = mnist.data / 255.0
        y = mnist.target.astype(int)
        class_names = [str(i) for i in range(10)]
    elif dataset_name == 'Bank marketing':
        bank_marketing = fetch_openml('bank-marketing', version=8)
        x = bank_marketing.data
        y = bank_marketing.target

        class_names = ['Ja', 'Nej']  # Anpassa detta enligt ditt behov

    else:
        raise ValueError("Datasetet kändes inte igen.")

    if samples is not None:
        x, y = resample(x, y, n_samples=samples, random_state=random_state, replace=False)

    return x, y, class_names


def process_dataset(dataset_name, test_size, samples, random_state):
    """
    Processes the given dataset using a specified mode
    :param dataset_name: The name of the dataset
    :param test_size: The amount of data that is used for testing
    :param samples: Number of samples
    :param random_state: Controls shuffling of the dataset before splitting.
    """
    # Load dataset.
    x, y, class_names = load_dataset(dataset_name, samples, random_state)

    evaluation.plot_distribution(x, y, class_names, dataset_name)

    # Split dataset into train and test data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    return x_train, x_test, y_train, y_test, class_names
