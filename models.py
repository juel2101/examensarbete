# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-04-26

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


def train_dt_model(x_train, y_train):
    """
    Training of decision tree model.

    :param x_train: Training data features
    :param y_train: Training data labels

    :return: The trained decision tree classifier
    """
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model


def train_svm_model(x_train, y_train):
    """
    Training of Support Vector Machine (SVM) model.

    :param x_train: Training data.
    :param y_train: Training labels.

    :return: Trained SVM model.
    """
    # model = svm.SVC(kernel='linear')
    model = svm.SVC(kernel='rbf')
    model.fit(x_train, y_train)
    return model
