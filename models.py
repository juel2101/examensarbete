# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-05-09

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
        model = svm.SVC()
    else:
        raise ValueError("Modellens typ kändes inte igen.")

    model.fit(x_train, y_train)
    return model


def find_optimal_hyperparameters(model_type, x_train, y_train, dataset_name):
    """
    Optimizes model by finding the best hyperparameter values using grid search.

    :param model_type: Type of model.
    :param x_train: Training data features.
    :param y_train: Training data labels.
    :param dataset_name: Name of dataset.
    """
    if model_type == 'DT':
        model = DecisionTreeClassifier()
        grid = {
            'max_depth': [None, 1, 5, 10, 15, 20],
            'min_samples_split': [2, 4, 8, 16, 32],
            'min_samples_leaf': [1, 5, 10, 15, 20],
            'criterion': ['gini', 'entropy', 'log_loss']
        }
    elif model_type == 'SVM' and dataset_name == 'Bank Marketing':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVC())
        ])
        grid = {
            'classifier__C': [1, 10, 100, 1000],
            'classifier__gamma': [0.0001, 0.001, 0.01, 0.1]
        }
    elif model_type == 'SVM' and dataset_name == 'MNIST':
        model = svm.SVC()
        grid = {
            'C': [1, 10, 100, 1000],
            'gamma': [0.0001, 0.001, 0.01, 0.1]
        }
    else:
        raise ValueError("Modellens typ kändes inte igen.")

    model = GridSearchCV(estimator=model, param_grid=grid, cv=3, verbose=3)
    model.fit(x_train, y_train)
    print(f"Bästa parametrarna: {model.best_params_} med noggrannheten: {model.best_score_}")


def optimized_train_model(model_type, x_train, y_train, dataset_name):
    """
    Optimized training of model.

    :param model_type: Type of model.
    :param x_train: Training data features.
    :param y_train: Training data labels.
    :param dataset_name: Name of dataset.

    :return: Optimized model.
    """
    if model_type == 'DT' and dataset_name == 'Bank Marketing':
        model = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=15, min_samples_split=8)
    elif model_type == 'DT' and dataset_name == 'MNIST':
        model = DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=1, min_samples_split=2)
    elif model_type == 'SVM' and dataset_name == 'Bank Marketing':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVC(C=1, gamma=0.1))
        ])
    elif model_type == 'SVM' and dataset_name == 'MNIST':
        model = svm.SVC(C=100, gamma=0.01)
    else:
        raise ValueError("Modellens typ kändes inte igen.")

    model.fit(x_train, y_train)
    return model
