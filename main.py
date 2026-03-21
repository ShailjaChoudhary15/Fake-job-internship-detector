from preprocessing.data_preparation import prepare_data

from models.random_forest import train_rf
from models.naive_bayes_model import train_nb
from models.sgd_model import train_sgd
from models.svm_model import train_svm

from analysis.comparison import compare_models

def main():
    print("Starting Fake Job Detection Project...")

    X_train, X_test, y_train, y_test = prepare_data("data/your_dataset.csv")
    rf = train_rf(X_train, y_train)
    nb = train_nb(X_train, y_train)
    sgd = train_sgd(X_train, y_train)
    svm = train_svm(X_train, y_train)

    compare_models(
        [rf, nb, sgd, svm],
        X_test,
        y_test
    )

if __name__ == "__main__":
    main()
