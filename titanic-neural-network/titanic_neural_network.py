"""
Titanic Neural Network (PyTorch)

A hand-built feedforward neural network predicting Titanic passenger
survival, built in PyTorch to compare against the Scikit-learn Random
Forest and Gradient Boosting models in this same project.

Version 1 trains for a fixed 100 epochs and exhibits an overfitting bug
(model reused/evaluated without tracking held-out loss). Version 2 fixes
this with early stopping on test loss (patience=10), checkpointing the
best-performing weights and reloading them before final evaluation.

Converted from titanic_neural_network.ipynb for CodeBuild/CI compatibility.
"""
import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


class TitanicNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x


def prepare_data(data_path: str):
    df = pd.read_csv(data_path)
    df["Age"] = SimpleImputer(strategy="median").fit_transform(df[["Age"]])
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
    df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

    X = df[FEATURES].values
    y = df["Survived"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train_t, X_test_t, y_train_t, y_test_t


def train_v1_basic(X_train_t, y_train_t, X_test_t, y_test_t, epochs: int = 100):
    """Fixed-epoch training. Reproduces the diagnosed overfitting bug:
    no held-out loss tracking, so the model trains past its generalization
    point with no signal to stop early."""
    model = TitanicNet(input_size=X_train_t.shape[1])
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = loss_function(predictions, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"[V1] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_preds_binary = (test_preds >= 0.5).float()

    acc = accuracy_score(y_test_t, test_preds_binary)
    f1 = f1_score(y_test_t, test_preds_binary)
    print(f"V1 Accuracy: {acc:.4f}")
    print(f"V1 F1-Score: {f1:.4f}")
    return model, acc, f1


def train_v2_early_stopping(X_train_t, y_train_t, X_test_t, y_test_t,
                              max_epochs: int = 300, patience: int = 10,
                              checkpoint_path: str = "best_model.pt"):
    """Fixes the V1 overfitting bug by tracking held-out loss each epoch
    and stopping once it fails to improve for `patience` consecutive epochs,
    reloading the best checkpoint before final evaluation."""
    model = TitanicNet(input_size=X_train_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.BCELoss()

    best_test_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = loss_function(predictions, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_loss = loss_function(model(X_test_t), y_test_t).item()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[V2] Stopped at epoch {epoch+1}, best test loss: {best_test_loss:.4f}")
            break

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_preds_binary = (test_preds >= 0.5).float()

    acc = accuracy_score(y_test_t, test_preds_binary)
    f1 = f1_score(y_test_t, test_preds_binary)
    print(f"V2 (Early Stopped) Accuracy: {acc:.4f}")
    print(f"V2 (Early Stopped) F1-Score: {f1:.4f}")
    return model, acc, f1


def train_model(data_path: str = "train.csv", output_path: str = "titanic_nn.pt"):
    """CodeBuild/CI entrypoint. Runs both versions, keeps the early-stopped
    (V2) model as the deployable artifact since it fixes the overfitting bug."""
    X_train_t, X_test_t, y_train_t, y_test_t = prepare_data(data_path)

    train_v1_basic(X_train_t, y_train_t, X_test_t, y_test_t)
    model, acc, f1 = train_v2_early_stopping(
        X_train_t, y_train_t, X_test_t, y_test_t, checkpoint_path=output_path
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train.csv")
    parser.add_argument("--output", default="titanic_nn.pt")
    args = parser.parse_args()
    train_model(args.data, args.output)


if __name__ == "__main__":
    main()
