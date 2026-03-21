import pandas as pd
import matplotlib.pyplot as plt

def run_analysis():
    df = pd.read_csv("data/fake_job_postings.csv")

    print("Dataset Shape:", df.shape)
    print("\nFraudulent Distribution:")
    print(df["fraudulent"].value_counts())
    print(f"\n  Real Jobs : {df['fraudulent'].value_counts()[0]}")
    print(f"  Fake Jobs : {df['fraudulent'].value_counts()[1]}")
    print(f"  Fake %    : {df['fraudulent'].mean() * 100:.2f}%")

    df["fraudulent"].value_counts().plot(
        kind="bar",
        color=["#55A868", "#C44E52"],
        edgecolor="black"
    )
    plt.title("Fake vs Real Jobs", fontsize=14)
    plt.xlabel("Class (0 = Real, 1 = Fake)")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=["Real Job", "Fake Job"], rotation=0)
    plt.tight_layout()
    plt.show()


