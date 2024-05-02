# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-05-02

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
        preprocessing.process_dataset(dataset_name, test_size, samples, 123))

    # Train model.
    model = models.train_model(model_type, x_train, y_train)

    evaluation.predict_and_evaluate_model(model, model_type, dataset_name, class_names, x_test, y_test)


def main():
    # Train and evaluate decision tree model with Iris dataset.
    # run_model('DT', 'MNIST', 1000)
    run_model('DT', 'Bank Marketing', None, 0.3)
    run_model('SVM', 'Bank Marketing', None, 0.3)

    # Train and evaluate support vector machine model with MNIST dataset.
    # run_model('SVM', 'MNIST', 0.3)

    # x, y, class_names = preprocessing.load_dataset(dataset_name, None, 42)

    # evaluation.plot_distribution(x, y, class_names, dataset_name)

    # x_resampled, y_resampled = preprocessing.undersample_dataset()

    # evaluation.plot_distribution(x_resampled, y_resampled, class_names, dataset_name)


if __name__ == "__main__":
    main()
