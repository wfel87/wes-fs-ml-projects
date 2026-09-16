"""
Titanic Neural Network (PyTorch)
Converted from titanic_neural_network.ipynb for CodeBuild/CI compatibility.

NOTE: This is a scaffold reconstruction reflecting the project's documented
approach (PyTorch feed-forward network with dropout/regularization added to
address an earlier overfitting diagnosis). Verify layer sizes, learning
rate, and epoch count against the original notebook before treating this
as authoritative.
"""
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score


class TitanicNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def prepare_data(data_path: str):
    df = pd.read_csv(data_path)
    features = ["Age", "Fare", "SibSp", "Parch", "Pclass"]

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df[features])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = df["Survived"].values.astype(np.float32)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_model(data_path: str = "train.csv", output_path: str = "titanic_nn.pt",
                 epochs: int = 50, lr: float = 0.001, weight_decay: float = 1e-4):
    X_train, X_test, y_train, y_test = prepare_data(data_path)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = TitanicNet(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32))
        test_labels = (test_preds.numpy() > 0.5).astype(int).flatten()

    print(f"Accuracy: {accuracy_score(y_test, test_labels):.4f}")
    print(f"F1 Score: {f1_score(y_test, test_labels):.4f}")

    torch.save(model.state_dict(), output_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train.csv")
    parser.add_argument("--output", default="titanic_nn.pt")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train_model(args.data, args.output, args.epochs)


if __name__ == "__main__":
    main()
