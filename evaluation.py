# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-05-09

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


def generate_confusion_matrix(model_type, dataset_name, class_names, y_test, y_pred):
    """
    Generates and display a confusion matrix.

    :param model_type: Type of model.
    :param dataset_name: Name of dataset.
    :param class_names: Names of the classes in the dataset.
    :param y_test: True labels of the test dataset.
    :param y_pred: Predicted labels by the model.
    """

    # Calculate confusion matrix.
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Greens')
    plt.title('Förvirringsmatris för ' + model_type + ' med ' + dataset_name + '-dataset')
    plt.ylabel('Faktiskt värde')
    plt.xlabel('Förutspått värde')
    plt.tight_layout()
    plt.show()


def evaluate_model(model_type, dataset_name, class_names, y_test, y_pred):
    """
    Predicts and evaluates model and plots the confusion matrix based on the results.

    :param model_type: Type of model.
    :param dataset_name: Name of dataset.
    :param class_names: Names of the classes in the dataset.
    :param y_test: True labels of the test dataset.
    :param y_pred: Predicted labels of the test dataset.
    """
    accuracy = accuracy_score(y_test, y_pred)
    print('Noggrannhet (accuracy): {:.4f}\n'.format(accuracy))

    clr = classification_report(y_test, y_pred)
    print(f'Resultat {model_type}:\n', clr)

    generate_confusion_matrix(model_type, dataset_name, class_names, y_test, y_pred)


def plot_distribution(y, dataset_name):
    """
    Plots the distribution of the classes in the dataset.

    :param y: Labels of the dataset
    :param dataset_name: Name of the dataset
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(10, 6))
    plt.bar(unique_classes, class_counts)
    plt.title(f"{dataset_name} dataset klassdistribution")
    plt.xlabel('Klassnamn')
    plt.ylabel('Frekvens')
    plt.xticks(unique_classes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
