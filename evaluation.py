# Författare: Caroline Berglin
# Datum: 2024-04-09
# Kurs: DT099G, examensarbete


from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay


def predict_model(model, x_test):
    """
    Predict labels using a trained model.

    :param model: Trained model.
    :param x_test: Features of the test data.

    :return: Predicted labels for the test data.
    """
    return model.predict(x_test)


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

    # Plot confusion matrix.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Greens')
    plt.title('Förvirringsmatris för ' + model_type + ' med ' + dataset_name + '-dataset')
    plt.ylabel('Faktiskt värde')
    plt.xlabel('Förutspått värde')
    plt.tight_layout()
    plt.show()


def predict_and_evaluate_model(model, model_type, dataset_name, class_names, x_test, y_test):
    """
    Predicts and evaluates model and plots the confusion matrix based on the results.

    :param model: Trained model.
    :param model_type: Type of model.
    :param dataset_name: Name of dataset.
    :param class_names: Names of the classes in the dataset.
    :param x_test: Features of the test data.
    :param y_test: True labels of the test dataset.
    """
    y_pred = predict_model(model, x_test)
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

    print(f'{model_type} = Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    generate_confusion_matrix(model_type, dataset_name, class_names, y_test, y_pred)
