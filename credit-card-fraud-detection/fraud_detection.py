import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv("creditcard.csv")

print(df["Class"].value_counts())
print(f"Fraud rate: {df['Class'].mean() * 100:.3f}%")

X = df.drop(columns=["Class"])
y = df["Class"]

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])
X["Time"] = scaler.fit_transform(X[["Time"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def evaluate(model, X_tr, y_tr, label):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_test)
    print(f"\n--- {label} ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1-Score:", f1_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=["legitimate", "fraud"]))

evaluate(RandomForestClassifier(random_state=42, n_jobs=-1), X_train, y_train, "Baseline")

evaluate(RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
          X_train, y_train, "class_weight=balanced")

evaluate(RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
          X_train, y_train, "class_weight=balanced_subsample")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_smote.value_counts().to_dict())

evaluate(RandomForestClassifier(random_state=42, n_jobs=-1), X_train_smote, y_train_smote, "SMOTE")
