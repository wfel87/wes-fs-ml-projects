# SMS Spam Classifier

A text classification project detecting spam vs. legitimate (ham) SMS messages using Scikit-learn, iterated from a baseline model to a tuned model targeting a diagnosed weakness.

## What This Project Does

This is Step 2 of a portfolio pipeline, bridging from tabular data (Step 1: Titanic survival classifier) into unstructured text. It proves the same core ML workflow, train, score, diagnose, fix, retrain, applies to language data once text is converted into numeric features.

## Dataset

- Source: [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- 5,572 SMS messages labeled ham (legitimate) or spam
- Class distribution: approximately 87% ham, 13% spam (imbalanced)

## Approach

### Baseline Model
1. Converted raw text into numeric features using `TfidfVectorizer`, which weighs words by how distinctive they are to a message versus how common they are overall.
2. Trained a `MultinomialNB` (Naive Bayes) classifier, the standard forgiving baseline for text classification.

### Tuned Model (Iteration 2)
1. Diagnosed the baseline's specific weakness: spam recall (0.81) was well below spam precision (0.98), meaning the model was too cautious about flagging real spam.
2. Swapped to `LogisticRegression`, which handles sparse TF-IDF features well, and added `class_weight="balanced"` to directly penalize the model more for missing the minority class.
3. Tuned with `GridSearchCV` across `C` and `class_weight` using 5-fold cross-validation.

## Results: Before vs After

| Metric | Baseline (Naive Bayes) | Tuned (Logistic Regression) | Change |
|---|---|---|---|
| Accuracy | 0.9731 | 0.9821 | +0.90 pts |
| F1-Score | 0.8898 | 0.9310 | +4.12 pts |
| Spam Precision | 0.98 | 0.96 | -0.02 |
| Spam Recall | 0.81 | 0.91 | +0.10 |

Best hyperparameters found: `C=10`, `class_weight="balanced"`

## What Changed The Score

The baseline model had a wide gap between spam precision (0.98) and spam recall (0.81), meaning it was accurate when it did flag spam, but let nearly 1 in 5 real spam messages through undetected. Rather than switching algorithms blindly, the fix specifically targeted that diagnosed weakness: `class_weight="balanced"` tells the model to penalize missed spam more heavily during training, directly raising recall from 0.81 to 0.91 while precision only dipped 2 points in exchange.

This confirms the same lesson learned on the Titanic project: a targeted fix aimed at a specifically diagnosed weakness outperforms a general algorithm swap. The gap between accuracy and F1 shrank from 8.3 points to 5.1 points, both metrics improved together, a sign of the model getting fairer across classes rather than just better at the easy one.

## Tools Used

- Python, Pandas
- Scikit-learn (`TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `GridSearchCV`, `train_test_split`, `accuracy_score`, `f1_score`, `classification_report`)

## Next in This Series

Step 3 moves into PyTorch, building a neural network from scratch, ahead of a Hugging Face fine-tuning project in Step 4.
