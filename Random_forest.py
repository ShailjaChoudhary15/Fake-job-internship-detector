import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def run_rf():
    print("\nRunning Random Forest Model...\n")

    df = pd.read_csv("data/cleaned_jobs.csv")

    X = df["text"]
    y = df["fraudulent"]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("Accuracy :", acc)
    print("Precision :", precision)
    print("Recall :", recall)
    print("F1 Score :", f1)

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)
    plt.imshow(cm)
    plt.title("Random Forest Confusion Matrix")
    plt.colorbar()
    plt.show()

    return acc
