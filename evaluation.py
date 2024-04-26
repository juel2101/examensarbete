# Författare: Caroline Berglin
# Datum: 2024-04-09
# Kurs: DT099G, examensarbete


import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.datasets import fetch_openml, fetch_covtype
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_dataset(dataset_name):
    """
    Function to load a specified dataset. Return the data, labels and class name.
    :param dataset_name: Name of the dataset to load (iris or mnist)
    :return: x: feature data of the dataset, y: target or label data corresponding to x and tha name of the classes in the dataset.
    """
    if dataset_name == 'coverType':
        data = fetch_covtype()
        x = data.data
        y = data.target
        # Defining class names based on the dataset documentation
        class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir',
                       'Krummholz']

    elif dataset_name == 'mnist':
        mnist = fetch_openml('mnist_784', version=1)
        x = mnist.data / 255.0
        y = mnist.target.astype(int)
        class_names = [str(i) for i in range(10)]
    else:
        raise ValueError("Dataset not recognized. Please use 'iris' or 'mnist'.")
    return x, y, class_names


def evaluate_model(y_test, y_pred):
    """
    Function to evaluate the model. Returns accuracy, precision, recall, and F1 score.
    :param y_test: True labels of the test dataset
    :param y_pred: Predicted labels by the model
    :return: The values accuracy, precision, recall and F1
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f1


def generate_confusion_matrix(class_names, y_test, y_pred):
    """
    Function to generate and display a confusion matrix
    :param class_names: Names of the classes in the dataset
    :param y_test: True labels of the test dataset
    :param y_pred: Predicted labels by the model
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap='Greens')
    plt.title('Förvirrings matris')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Faktiskt värde')
    plt.xlabel('Förutspått värde')
    plt.show()


def predict_model(clf, x_test):
    """
    Function to make predictions using a trained model classifier
    :param clf: Trained decision tree classifier
    :param x_test: Test data features
    :return: Array of predicted labels for the test data
    """
    return clf.predict(x_test)
