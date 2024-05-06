# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-05-02
import numpy as np
from sklearn.model_selection import GridSearchCV
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
        model.fit(x_train, y_train)
        while True:
            weakest_link = np.argmin(model.tree_.impurity)
            if weakest_link == 0:
                break
            model.tree_.children_left[weakest_link] = model.tree_.children_right[weakest_link] = -1
    elif model_type == 'SVM':
        model = svm.SVC(kernel='rbf')
    else:
        raise ValueError("Modellens typ kändes inte igen.")

    model.fit(x_train, y_train)
    return model
