"""
Titanic Survival Classifier
Converted from titanic_survival_classifier.ipynb for CodeBuild/CI compatibility.

NOTE: This is a scaffold reconstruction based on the project's documented
approach (Scikit-learn RandomForest/LogisticRegression, GridSearchCV with
cv=5, Title/FamilySize feature engineering). Verify field names, model
choice, and hyperparameter grid against the original notebook before
relying on this for production training.
"""
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


def build_pipeline(numeric_features, categorical_features) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])


def train_model(data_path: str = "train.csv", output_path: str = "titanic_model.joblib"):
    df = pd.read_csv(data_path)
    df = engineer_features(df)

    numeric_features = ["Age", "Fare", "FamilySize"]
    categorical_features = ["Sex", "Pclass", "Embarked", "Title"]
    target = "Survived"

    X = df[numeric_features + categorical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numeric_features, categorical_features)

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 5, 10],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1 Score: {f1_score(y_test, preds):.4f}")

    joblib.dump(best_model, output_path)
    return best_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train.csv")
    parser.add_argument("--output", default="titanic_model.joblib")
    args = parser.parse_args()
    train_model(args.data, args.output)


if __name__ == "__main__":
    main()
