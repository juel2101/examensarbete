# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-05-02

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


def train_model(model_type, x_train, y_train):
    """
    Training of model.

    :param model_type: Type of model.
    :param x_train: Training data features.
    :param y_train: Training data labels.

    :return: The trained model.
    """
    if model_type == 'DT':
        model = DecisionTreeClassifier()
    elif model_type == 'SVM':
        model = svm.SVC(kernel='rbf')
    else:
        raise ValueError("Modellens typ kändes inte igen.")

    model.fit(x_train, y_train)
    return model
