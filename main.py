# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-04-26
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_covtype

import preprocessing
import models
import evaluation


def run_model(model_type, dataset_name, samples, test_size):
    """
    Loading and processing the dataset and training the model based of the type of the model.

    :param model_type: Type of model.
    :param dataset_name: Name of dataset.
    :param samples: Number of samples.
    :param test_size: The amount of data that is used for testing.
    """
    # Load and process dataset.
    x_train, x_test, y_train, y_test, class_names = (
        preprocessing.process_dataset(dataset_name, test_size, samples, 17))
    model = None

    # Train model based on type of model.
    if model_type == 'DT':
        model = models.train_dt_model(x_train, y_train)
    elif model_type == 'SVM':
        model = models.train_svm_model(x_train, y_train)

    evaluation.predict_and_evaluate_model(model, model_type, dataset_name, class_names, x_test, y_test)


def main():
    # Train and evaluate decision tree model with Iris dataset.
    # run_model('DT', 'MNIST', 1000)
    run_model('SVM', 'Covertype', 10000, 3000)

    # Train and evaluate support vector machine model with Iris dataset.
    # run_model('SVM', 'MNIST', 0.3)

    # dataset_name = 'Covertype'
    # x, y, class_names = preprocessing.load_dataset(dataset_name, None, 42)

    # balancing.plot_distribution(x, y, class_names, dataset_name)

    # x_resampled, y_resampled = balancing.undersample_dataset(dataset_name)

    # evaluation.plot_distribution(x_resampled, y_resampled, class_names, dataset_name)


if __name__ == "__main__":
    main()
