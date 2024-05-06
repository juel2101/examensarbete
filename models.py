# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-05-02

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
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier())
        ])
    elif model_type == 'SVM':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVC(kernel='rbf'))
        ])
    else:
        raise ValueError("Modellens typ kändes inte igen.")

    pipeline.fit(x_train, y_train)
    return pipeline


def optimize_model(model_type, x_train, y_train):
    if model_type == 'DT':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier())
        ])
        grid = {
            'classifier__max_depth': list(range(1, 40)),
            'classifier__min_samples_split': list(range(1, 40)),
            'classifier__min_samples_leaf': list(range(1, 20)),
            'classifier__criterion': ['gini', 'entropy', 'log_loss']
        }
    elif model_type == 'SVM':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVC(kernel='rbf'))
        ])
        grid = {
            'classifier__C': [0.1, 1, 10, 100, 1000],
            'classifier__gamma': [0.0001, 0.001, 0.01, 0.1, 1],
        }
    else:
        raise ValueError("Modellens typ kändes inte igen.")

    model = GridSearchCV(estimator=pipeline, param_grid=grid, cv=5)
    model.fit(x_train, y_train)
    print(f"Bästa parametrarna: {model.best_params_} med noggrannheten: {model.best_score_}")
