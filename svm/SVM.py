# Författare: Julia Ellström
# Kurs: DT099, Examensarbete
# Datum: 2024-04-09


import numpy as np
from sklearn import svm, datasets
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import itertools


def load_dataset(name):
    if name == 'iris':
        dataset = datasets.load_iris()
        X = dataset.data
        y = dataset.target
        target_names = dataset.target_names
    elif name == 'mnist':
        dataset = fetch_openml('mnist_784', version=1, as_frame=False)
        X = dataset.data / 255.0
        y = dataset.target.astype(int)
        target_names = [str(i) for i in range(10)]
    else:
        raise ValueError("Datasetet är inte tillgängligt")
    return X, y, target_names


def split_dataset(X, y, test_size, random_state):
    """
    Split dataset into training and test data.

    :param X: Features or input data.
    :param y: Target variable or labels.
    :param test_size: Proportion of dataset to include in the test split.
    :param random_state: Controls shuffling of the dataset before splitting.

    :return X_train: Training data.
    :return X_test: Test data.
    :return y_train: Training labels.
    :return y_test: Test labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_svm_model(X_train, y_train):
    """
    Training of Support Vector Machine (SVM) model.

    :param X_train: Training data.
    :param y_train: Training labels.

    :return: Trained SVM model.
    """
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model


def predict_svm_model(model, X_test):
    """
    Testing of Support Vector Machine (SVM) model.

    :param model: Trained SVM model.
    :param X_test: Test data.

    :return: Predicted labels.
    """
    # Use the trained SVM model to predict labels for the test data.
    y_pred = model.predict(X_test)

    return y_pred


def evaluate_model(y_test, y_pred):
    """
    Evaluate the performance of a classification model.

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


def plot_confusion_matrix(y_test, y_pred, class_names):
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plotting the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap="Greens")
    plt.title('Förvirringsmatris')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Faktiskt värde')
    plt.xlabel('Förutspått värde')