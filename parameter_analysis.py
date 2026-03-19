import pandas as pd
import matplotlib.pyplot as plt


def run_analysis():

    df = pd.read_csv("data/fake_job_postings.csv")

    print("Dataset Shape:", df.shape)

    print("\nFraudulent distribution")
    print(df["fraudulent"].value_counts())

    df["fraudulent"].value_counts().plot(kind="bar")

    plt.title("Fake vs Real Jobs")

    plt.xlabel("Class")

    plt.ylabel("Count")

    plt.show()