import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "data", "fake_job_postings.csv")
df = pd.read_csv(file_path)

df = df.fillna('')
df['text'] = df['title'] + " " + df['description']

X = df['text']
y = df['fraudulent']  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = SVC(probability=True)
model.fit(X_train_vec, y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")