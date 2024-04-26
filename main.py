# Authors: Caroline Berglin and Julia Ellström
# Course: DT099G, Examensarbete
# Date: 2024-04-26

import preprocessing
import models
import evaluation


def main():
    # Train and evaluate decision tree model with Iris dataset.
    run_model('DT', 'Iris', 0.3)

    # Train and evaluate support vector machine model with Iris dataset.
    run_model('SVM', 'Iris', 0.3)


if __name__ == "__main__":
    main()
