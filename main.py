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

    # Test model.
    y_pred = model.predict(x_test)

    # Evaluate model.
    evaluation.evaluate_model(model_type, dataset_name, class_names, y_test, y_pred)



def main():
    # run_model('DT', 'Bank Marketing', None, 0.2)
    run_model('SVM', 'Bank Marketing', None, 0.2)
    # run_model('DT', 'MNIST', None, 10000)
    # run_model('SVM', 'MNIST', None, 10000)


if __name__ == "__main__":
    main()
