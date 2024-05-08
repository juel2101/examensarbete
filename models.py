# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-05-02
import optuna
from sklearn.model_selection import GridSearchCV, cross_val_score
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

    Bankmarketing = criterion='gini', max_depth=10, min_samples_leaf=20, min_samples_split=2
    """
    if model_type == 'DT':
        model = DecisionTreeClassifier()
    elif model_type == 'SVM':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVC(kernel='rbf'))
        ])
    else:
        raise ValueError("Modellens typ kändes inte igen.")

    model.fit(x_train, y_train)
    return model


def optimize_model(model_type, x_train, y_train):
    if model_type == 'DT':
        model = DecisionTreeClassifier()
        grid = {
            'max_depth': [None, 1, 5, 10, 15, 20],
            'min_samples_split': [2, 4, 8, 16, 32],
            'min_samples_leaf': [1, 5, 10, 15, 20],
            'criterion': ['gini', 'entropy', 'log_loss']
        }
    elif model_type == 'SVM':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVC(kernel='rbf'))
        ])
        grid = {
            'classifier__C': [0.1, 1, 10, 100, 1000],
            'classifier__gamma': [0.0001, 0.001, 0.01, 0.1, 1],
        }
    else:
        raise ValueError("Modellens typ kändes inte igen.")

    model = GridSearchCV(estimator=model, param_grid=grid, cv=3, verbose=3)
    model.fit(x_train, y_train)
    print(f"Bästa parametrarna: {model.best_params_} med noggrannheten: {model.best_score_}")
