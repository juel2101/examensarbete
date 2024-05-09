# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-05-02

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def load_dataset(dataset_name, samples, random_state):
    """
    Load and preprocess specified dataset.

    :param dataset_name: Name of the dataset to load.
    :param samples: Number of samples.
    :param random_state: Controls shuffling of the dataset.

    :return: x: feature data of the dataset.
    :return: y: target or label data corresponding to x and tha name of the classes in the dataset.
    :return class_names: names of the classes in the dataset.
    """

    if dataset_name == 'MNIST':
        mnist = fetch_openml('mnist_784', version=1)
        x = mnist.data / 255.0
        y = mnist.target.astype(int)
        class_names = [str(i) for i in range(10)]
    elif dataset_name == 'Bank Marketing':
        bank_marketing = fetch_openml('bank-marketing', version=8)
        x = bank_marketing.data
        y = bank_marketing.target
        class_names = ['Ja', 'Nej']

    else:
        raise ValueError("Datasetet kändes inte igen.")

    if samples is not None:
        x, y = resample(x, y, n_samples=samples, random_state=random_state, replace=False)

    return x, y, class_names


def process_dataset(dataset_name, test_size, samples, random_state=123):
    """
    Processes the given dataset.

    :param dataset_name: The name of the dataset.
    :param test_size: The amount of data that is used for testing.
    :param samples: Number of samples.
    :param random_state: Controls shuffling of the dataset before splitting.

    :return x_train: Training data features.
    :return x_test: Testing data features.
    :return y_train: Training data labels.
    :return y_test: Testing data labels.
    :return class_names: Name of classes in the dataset.
    """
    # Load dataset.
    x, y, class_names = load_dataset(dataset_name, samples, random_state)

    # Split dataset into train and test data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    return x_train, x_test, y_train, y_test, class_names
