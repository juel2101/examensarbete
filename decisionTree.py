# Författare: Caroline Berglin
# Datum: 2024-
# Kurs: DT099G, examensarbete

from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(x_train, y_train):
    """
    Function to train a decision tree classifier
    :param x_train: Training data features
    :param y_train: Training data labels
    :return: The trained decision tree classifier
    """
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return clf
