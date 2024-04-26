# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-04-26

from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_dataset(dataset_name):
    """
    Load a specified dataset. Return the data, labels and class name.

    :param dataset_name: Name of the dataset to load (iris or mnist)

    :return: x: feature data of the dataset,
    :return: y: target or label data corresponding to x and tha name of the classes in the dataset.
    """
    if dataset_name == 'coverType':
        data = fetch_covtype()
        x = data.data
        y = data.target
        class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir',
                       'Krummholz']
    elif dataset_name == 'MNIST':
        mnist = fetch_openml('mnist_784', version=1)
        x = mnist.data / 255.0
        y = mnist.target.astype(int)
        class_names = [str(i) for i in range(10)]
    elif dataset_name == 'Iris':
        dataset = datasets.load_iris()
        x = dataset.data
        y = dataset.target
        class_names = dataset.target_names
    else:
        raise ValueError("Datasetet kändes inte igen.")
    return x, y, class_names


def process_dataset(dataset_name, test_size, random_state):
    """
    Processes the given dataset using a specified model.

    :param dataset_name: The name of the dataset
    :param test_size: The amount of data that is used for testing
    :param random_state: Controls shuffling of the dataset before splitting.
    """
    # Load dataset.
    x, y, class_names = load_dataset(dataset_name)

    # Split dataset into train and test data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    return x_train, x_test, y_train, y_test, class_names
