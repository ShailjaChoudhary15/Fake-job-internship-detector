import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def run_nb():
    df = pd.read_csv("data/cleaned_jobs.csv")
    X = df["text"]
    y = df["fraudulent"]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)
    plt.imshow(cm)
    plt.title("Naive Bayes Confusion Matrix")
    plt.colorbar()
    plt.show()

    return acc
