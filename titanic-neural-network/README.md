# Titanic Neural Network (PyTorch)

A hand-built feedforward neural network predicting Titanic passenger survival, built from raw PyTorch components rather than a pre-built Scikit-learn model. Compares directly against the Random Forest and Gradient Boosting results from the sibling `titanic-survival-classifier` project.

## What This Project Does

This is Step 3 of the portfolio pipeline, moving from Scikit-learn's pre-built classical models into a neural network architecture defined and trained from scratch in PyTorch: layers, forward pass, loss function, backpropagation, and optimizer, all configured manually.

## Approach

### Version 1: Basic Neural Network
A 3-layer feedforward network (7 input features to 16 neurons to 8 neurons to 1 output probability), trained for a fixed 100 epochs using `BCELoss` and the `Adam` optimizer. Training loss decreased steadily and looked healthy on its own, but test loss was never tracked, hiding a real problem.

### Version 2: Early-Stopped Neural Network
Tracking test loss alongside train loss for 300 epochs revealed classic overfitting, train loss kept falling while test loss climbed from 0.4685 to over 1.3 by epoch 300. Early stopping was added to halt training automatically once test loss stopped improving for 10 consecutive epochs, and the best-performing checkpoint was reloaded for final evaluation.

## Results: Three-Way Comparison

| Model | Accuracy | F1-Score | Notes |
|---|---|---|---|
| Random Forest (baseline) | 0.8212 | 0.7746 | From titanic-survival-classifier |
| Gradient Boosting (tuned) | 0.8324 | 0.7917 | Best result, feature engineering + GridSearchCV |
| Neural Network (early-stopped) | 0.8101 | 0.7571 | This project |

## What This Comparison Shows

Despite proper regularization and early stopping, the neural network underperformed both tree-based models on this small (891-row) tabular dataset, consistent with known sample-efficiency advantages of gradient boosting on structured data. Neural networks are expected to show their advantage on larger or unstructured datasets, which the next phase of this portfolio (NLP fine-tuning with Hugging Face) will test directly.

## A Diagnosed Bug Worth Documenting

An early attempt at early stopping produced a nonsensical result (best test loss of 1.3192, stopping after only 11 epochs) because the model object was reused from the prior, already-overfit 300-epoch run instead of being reinitialized. PyTorch models retain their weights across cells unless explicitly recreated, unlike Scikit-learn's `.fit()`, which retrains cleanly on the same object. Reinitializing both the model and optimizer before the corrected run fixed this and produced the legitimate result above.

## Tools Used

- Python, Pandas
- PyTorch (`nn.Module`, `nn.Linear`, `nn.ReLU`, `nn.Sigmoid`, `nn.BCELoss`, `torch.optim.Adam`, manual training loop, early stopping)
- Scikit-learn (`StandardScaler`, `SimpleImputer`, `LabelEncoder`, `accuracy_score`, `f1_score`) for preprocessing and evaluation

## Next in This Series

Step 4 moves into Hugging Face, fine-tuning a pretrained transformer model on a text classification task using PyTorch under the hood via the Hugging Face `Trainer` API.
