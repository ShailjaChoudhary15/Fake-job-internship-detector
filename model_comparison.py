import matplotlib.pyplot as plt

def show_comparison(sgd_acc, svm_acc, nb_acc, rf_acc):
    models = ["SGD", "SVM", "Naive Bayes", "Random Forest"]
    scores = [sgd_acc, svm_acc, nb_acc, rf_acc]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, scores, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    plt.title("Model Accuracy Comparison", fontsize=14)
    plt.ylabel("Accuracy")
    plt.ylim(0.85, 1.0)

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()
    plt.show()
