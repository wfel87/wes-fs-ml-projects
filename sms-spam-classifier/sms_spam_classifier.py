"""
SMS Spam Classifier (Scikit-learn)

A text classification pipeline using TF-IDF, moving from tabular data
into unstructured text. Includes a baseline Naive Bayes model and a
tuned Logistic Regression model (GridSearchCV over C and class_weight)
targeting recall on the minority spam class.

Converted from sms_spam_classifier.ipynb for CodeBuild/CI compatibility.
"""
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_data(data_path: str):
    df = pd.read_csv(data_path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["text"] = df["text"].fillna("")
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(stop_words="english", max_features=3000)


def train_baseline(X_train_tfidf, y_train, X_test_tfidf, y_test):
    """Baseline: Naive Bayes with TF-IDF, no tuning."""
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"Baseline Accuracy: {acc:.4f}")
    print(f"Baseline F1-Score: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))
    return model, acc, f1


def train_tuned(X_train_tfidf, y_train, X_test_tfidf, y_test):
    """Tuned: Logistic Regression with class_weight balanced option,
    selected via GridSearchCV(cv=5) over C and class_weight."""
    param_grid = {
        "C": [0.1, 1, 10],
        "class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid_search.fit(X_train_tfidf, y_train)
    print(f"Best parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"Tuned Accuracy: {acc:.4f}")
    print(f"Tuned F1-Score: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))
    return best_model, acc, f1


def train_model(data_path: str = "spam.csv", output_path: str = "spam_model.joblib"):
    """CodeBuild/CI entrypoint. Runs the Naive Bayes baseline for comparison,
    then keeps the tuned Logistic Regression model as the deployable artifact."""
    df = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    train_baseline(X_train_tfidf, y_train, X_test_tfidf, y_test)
    best_model, acc, f1 = train_tuned(X_train_tfidf, y_train, X_test_tfidf, y_test)

    joblib.dump({"vectorizer": vectorizer, "model": best_model}, output_path)
    return best_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="spam.csv")
    parser.add_argument("--output", default="spam_model.joblib")
    args = parser.parse_args()
    train_model(args.data, args.output)


if __name__ == "__main__":
    main()
