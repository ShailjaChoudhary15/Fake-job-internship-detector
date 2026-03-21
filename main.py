import matplotlib.pyplot as plt
from sgd_model import run_sgd
from svm_model import run_svm
from naive_bayes import run_nb
from random_forest import run_rf

def main():
    print("=" * 50)
    print("  Fake Job / Internship Detector")
    print("=" * 50)

    sgd_acc = run_sgd()
    svm_acc = run_svm()
    nb_acc  = run_nb()
    rf_acc  = run_rf()

    print("\n" + "=" * 50)
    print("  Final Accuracy Comparison")
    print("=" * 50)
    print(f"  SGD          : {sgd_acc:.4f}")
    print(f"  SVM          : {svm_acc:.4f}")
    print(f"  Naive Bayes  : {nb_acc:.4f}")
    print(f"  Random Forest: {rf_acc:.4f}")
    print("=" * 50)

    models = ["SGD", "SVM", "Naive Bayes", "Random Forest"]
    scores = [sgd_acc, svm_acc, nb_acc, rf_acc]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, scores, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    plt.title("Model Accuracy Comparison", fontsize=14)
    plt.ylabel("Accuracy")
    plt.ylim(0.85, 1.0)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002,
                 f"{score:.4f}",
                 ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
