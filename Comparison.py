import matplotlib.pyplot as plt
from svm_model import run_svm
from sgd_model import run_sgd
from naive_bayes import run_nb
from random_forest import run_rf


def run_comparison():

    svm_acc = run_svm()
    sgd_acc = run_sgd()
    nb_acc = run_nb()
    rf_acc = run_rf()

    models = ["SVM", "SGD", "Naive Bayes", "Random Forest"]

    scores = [svm_acc, sgd_acc, nb_acc, rf_acc]

    plt.bar(models, scores)

    plt.title("Model Accuracy Comparison")

    plt.ylabel("Accuracy")

    plt.show()