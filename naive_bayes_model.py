{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;\red249\green248\blue242;\red16\green16\blue16;}
{\*\expandedcolortbl;;\cssrgb\c98039\c97647\c96078;\cssrgb\c7843\c7843\c7451;}
\paperw11900\paperh16840\margl1440\margr1440\vieww29200\viewh18460\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs32 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 import pandas as pd\
from sklearn.model_selection import train_test_split\
from sklearn.feature_extraction.text import TfidfVectorizer\
from sklearn.naive_bayes import MultinomialNB\
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix\
import matplotlib.pyplot as plt\
def run_nb():\
    df = pd.read_csv("data/cleaned_jobs.csv")\
    X = df["text"]\
    y = df["fraudulent"]\
    vectorizer = TfidfVectorizer(stop_words="english")\
    X = vectorizer.fit_transform(X)\
    X_train, X_test, y_train, y_test = train_test_split(\
        X, y, test_size=0.2, random_state=42\
    )\
    model = MultinomialNB()\
    model.fit(X_train, y_train)\
    pred = model.predict(X_test)\
    acc = accuracy_score(y_test, pred)\
    precision = precision_score(y_test, pred)\
    recall = recall_score(y_test, pred)\
    f1 = f1_score(y_test, pred)\
    print("Accuracy:", acc)\
    print("Precision:", precision)\
    print("Recall:", recall)\
    print("F1 Score:", f1)\
    print(classification_report(y_test, pred))\
    cm = confusion_matrix(y_test, pred)\
    plt.imshow(cm)\
    plt.title("Naive Bayes Confusion Matrix")\
    plt.colorbar()\
    plt.show()\
    return acc}