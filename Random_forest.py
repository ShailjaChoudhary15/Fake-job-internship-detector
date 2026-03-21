{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
import matplotlib.pyplot as plt\
\
from sklearn.model_selection import train_test_split\
from sklearn.feature_extraction.text import TfidfVectorizer\
from sklearn.ensemble import RandomForestClassifier\
\
from sklearn.metrics import (\
    accuracy_score,\
    precision_score,\
    recall_score,\
    f1_score,\
    classification_report,\
    confusion_matrix\
)\
\
\
def run_rf():\
\
    print("\\nRunning Random Forest Model...\\n")\
\
    # Load dataset\
    df = pd.read_csv("data/cleaned_jobs.csv")\
\
    # Features and labels\
    X = df["text"]\
    y = df["fraudulent"]\
\
    # Convert text to numerical vectors\
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)\
\
    X = vectorizer.fit_transform(X)\
\
    # Split dataset\
    X_train, X_test, y_train, y_test = train_test_split(\
        X,\
        y,\
        test_size=0.2,\
        random_state=42\
    )\
\
    # Create model\
    model = RandomForestClassifier(\
        n_estimators=100,\
        random_state=42\
    )\
\
    # Train model\
    model.fit(X_train, y_train)\
\
    # Predictions\
    predictions = model.predict(X_test)\
\
    # Evaluation metrics\
    acc = accuracy_score(y_test, predictions)\
    precision = precision_score(y_test, predictions)\
    recall = recall_score(y_test, predictions)\
    f1 = f1_score(y_test, predictions)\
\
    print("Accuracy :", acc)\
    print("Precision :", precision)\
    print("Recall :", recall)\
    print("F1 Score :", f1)\
\
    print("\\nClassification Report:\\n")\
    print(classification_report(y_test, predictions))\
\
    # Confusion matrix\
    cm = confusion_matrix(y_test, predictions)\
\
    print("\\nConfusion Matrix:\\n")\
    print(cm)\
\
    # Plot confusion matrix\
    plt.imshow(cm)\
\
    plt.title("Random Forest Confusion Matrix")\
\
    plt.xlabel("Predicted")\
\
    plt.ylabel("Actual")\
\
    plt.colorbar()\
\
    plt.show()\
\
    # Return accuracy for comparison module\
    return acc\
\
\
# Run independently (for testing)\
if _name_ == "_main_":\
    run_rf()}