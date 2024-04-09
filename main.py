# Författare: Julia Ellström
# Kurs: DT099, Examensarbete
# Datum: 2024-04-08

from matplotlib import pyplot as plt

import SVM


def main():

    X, y, class_names = SVM.load_dataset('iris')

    # Split the dataset.
    X_train, X_test, y_train, y_test = SVM.split_dataset(X, y, 0.3, 17)

    # Train the SVM model.
    svm_model = SVM.train_svm_model(X_train, y_train)

    # Make predictions.
    y_pred = SVM.predict_svm_model(svm_model, X_test)

    # Evaluate the model.
    accuracy, precision, recall, f1 = SVM.evaluate_model(y_test, y_pred)

    # Print the results.
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

    # Plot the confusion matrix.
    SVM.plot_confusion_matrix(y_test, y_pred, class_names=class_names)
    plt.show()


if __name__ == "__main__":
    main()
