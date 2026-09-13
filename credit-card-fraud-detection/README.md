# Credit Card Fraud Detection

## Problem
Detect fraudulent credit card transactions in a highly imbalanced dataset: 284,807 total transactions, only 492 (0.173%) are fraud.

## Dataset
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) - 28 PCA-anonymized features (V1-V28) plus `Time` and `Amount`. Target column: `Class` (0 = legitimate, 1 = fraud).

## Why Accuracy Is Misleading Here
A model that predicts "legitimate" for every transaction would score 99.827% accuracy while catching zero fraud. Because of this, precision and recall on the fraud class are the metrics that actually matter, not overall accuracy.

- Recall: of all real fraud cases, how many did the model catch?
- Precision: of all transactions flagged as fraud, how many were actually fraud?

Given that missed fraud (false negative) costs real stolen money while a false alarm (false positive) only inconveniences a legitimate customer, recall is prioritized over precision for this use case.

## Approach & Results

Four Random Forest variants were tested, all trained on an 80/20 stratified train/test split, evaluated on the same held-out 20% (56,962 transactions, 98 of them fraud).

| Approach | Precision (fraud) | Recall (fraud) | F1 (fraud) |
|---|---|---|---|
| Baseline Random Forest | 0.94 | 0.82 | 0.874 |
| `class_weight="balanced"` | 0.96 | 0.74 | 0.839 |
| `class_weight="balanced_subsample"` | 0.96 | 0.73 | 0.832 |
| SMOTE oversampling + Random Forest | 0.85 | 0.84 | 0.841 |

### Key Finding
Standard class-weighting techniques (`balanced`, `balanced_subsample`) moved results in the *opposite* direction expected: recall dropped instead of rising. This happens because Random Forest trains each tree on a random bootstrap subsample of the data; with fraud this rare (0.173%), some subsamples contain too few (or zero) fraud examples for reweighting alone to help. Reweighting the same scarce examples doesn't solve a scarcity problem.

SMOTE (Synthetic Minority Oversampling Technique) solved this by generating ~227,000 synthetic fraud examples from the 394 real fraud cases in the training set, balancing the training data to 227,451 / 227,451. This gave the model genuinely more fraud-pattern data to learn from, rather than just more emphasis on the same small pile, and successfully raised recall from 0.82 to 0.84 (catching 2 additional real fraud cases out of 98), at the cost of precision dropping from 0.94 to 0.85.

### Business Tradeoff
There is no single "correct" model here; it depends on the cost tolerance of the business:
- Baseline is preferable if false alarms are expensive to review (e.g., a small manual fraud review team).
- SMOTE is preferable if missed fraud (real financial loss) is the dominant cost, which is the more common assumption in production fraud systems.

## Tech Stack
- Scikit-learn (RandomForestClassifier, train_test_split, StandardScaler, classification_report)
- imbalanced-learn (SMOTE)
- Pandas

## Files
- `fraud_detection.py` - full pipeline: load, scale, split, train all four variants, evaluate
