# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-05-02

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
import pandas as pd


def predict_model(model, x_test):
    """
    Predict labels using a trained model.

    :param model: Trained model.
    :param x_test: Features of the test data.

    :return: Predicted labels for the test data.
    """
    return model.predict(x_test)


def evaluate_model(y_test, y_pred):
    """
    Evaluate the performance of a trained model.

    :param y_test: True labels of the test data.
    :param y_pred: Predicted labels for the test data.

    :return accuracy: Accuracy of the model.
    :return precision: Precision of the model.
    :return recall: Recall of the model.
    :return f1: F1-score of the model.

    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f1


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
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def predict_and_evaluate_model(model, model_type, dataset_name, class_names, x_test, y_test):
    """
    Predicts and evaluates model and plots the confusion matrix based on the results.

    :param model: Trained model.
    :param model_type: Type of model.
    :param dataset_name: Name of dataset.
    :param class_names: Names of the classes in the dataset.
    :param x_test: Features of the test data.
    :param y_test: True labels of the test dataset.
    """
    y_pred = predict_model(model, x_test)
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

    print(f'{model_type} = Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    # generate_confusion_matrix(model_type, dataset_name, class_names, y_test, y_pred)


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
