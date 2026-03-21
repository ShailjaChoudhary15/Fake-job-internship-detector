import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def run_nb():
    print("\n" + "-" * 40)
    print("  Running Naive Bayes Model...")
    print("-" * 40)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data", "fake_job_postings.csv")

    df = pd.read_csv(file_path)
    df = df.fillna('')
    df['text'] = df['title'] + " " + df['description']

    X = df['text']
    y = df['fraudulent']

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision_score(y_test, pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, pred):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=["Real Job", "Fake Job"]))

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title("Naive Bayes - Confusion Matrix", fontsize=14)
    plt.colorbar()
    plt.xticks([0, 1], ["Real Job", "Fake Job"])
    plt.yticks([0, 1], ["Real Job", "Fake Job"])
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black",
                     fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return acc
