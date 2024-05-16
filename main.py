# Författare: Caroline Berglin
# Datum: 2024-04-09
# Kurs: DT099G, examensarbete

from sklearn.model_selection import train_test_split
import decisionTree
import evaluation


def process_dataset(dataset_name, model, test_size):
    """
    Function that processes the given dataset using a specified model.
    :param dataset_name: The name of the dataset (ex. iria or mnist)
    :param model: Modul that contains functions to train and test the model
    :param test_size: The amount of data that is used for testing
    """
    x, y, dataset_name = evaluation.load_dataset(dataset_name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=17)
    dt_model = model.train_decision_tree(x_train, y_train)
    y_pred = evaluation.predict_model(dt_model, x_test)
    accuracy, precision, recall, f1 = evaluation.evaluate_model(y_test, y_pred)
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    evaluation.generate_confusion_matrix(dataset_name, y_test, y_pred)


def main():
    process_dataset('coverType', decisionTree, 0.3)
    process_dataset('mnist', decisionTree, 10000)


if __name__ == '__main__':
    main()
