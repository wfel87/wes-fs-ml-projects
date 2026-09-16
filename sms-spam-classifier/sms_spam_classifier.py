"""
SMS Spam Classifier (Scikit-learn)
Converted from sms_spam_classifier.ipynb for CodeBuild/CI compatibility.

NOTE: This is a scaffold reconstruction based on the standard TF-IDF +
classifier approach documented for this project. Verify text-cleaning
steps, vectorizer settings, and model choice against the original notebook.
"""
import argparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


def build_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def train_model(data_path: str = "spam.csv", output_path: str = "spam_model.joblib"):
    df = pd.read_csv(data_path, encoding="latin-1")
    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1 Score: {f1_score(y_test, preds):.4f}")

    joblib.dump(pipeline, output_path)
    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="spam.csv")
    parser.add_argument("--output", default="spam_model.joblib")
    args = parser.parse_args()
    train_model(args.data, args.output)


if __name__ == "__main__":
    main()
