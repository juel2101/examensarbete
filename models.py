# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-05-02

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
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier())
        ])
    elif model_type == 'SVM':
        grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf']
        }
        model = GridSearchCV(estimator=svm.SVC(), param_grid=grid, cv=5)
        print(model.best_params_)
    else:
        raise ValueError("Modellens typ kändes inte igen.")

    model.fit(x_train, y_train)
    return model
