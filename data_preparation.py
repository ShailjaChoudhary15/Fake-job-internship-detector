import pandas as pd
import os

def load_and_preprocess():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data", "fake_job_postings.csv")

    if not os.path.exists(file_path):
        print("Dataset not found at:", file_path)
        exit()

    df = pd.read_csv(file_path)

    df = df.fillna('')
    df['text'] = df['title'] + " " + df['description']

    X = df['text']
    y = df['fraudulent']

    return X, y
