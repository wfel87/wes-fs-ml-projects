# Titanic Survival Classifier

A machine learning project predicting passenger survival on the Titanic using Scikit-learn, iterated from a baseline model to a tuned, feature-engineered model.

## What This Project Does

This is Step 1 of a portfolio pipeline moving from Scikit-learn (data cleaning and classical ML) toward PyTorch and Hugging Face (deep learning and pretrained model fine-tuning) for future projects.

## Dataset

- Source: [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

## Approach

### Baseline Model
1. Filled missing `Age` (median) and `Embarked` (mode) values with `SimpleImputer` and `fillna`.
2. Encoded `Sex` and `Embarked` with `LabelEncoder`.
3. Trained a `RandomForestClassifier` on Pclass, Sex, Age, SibSp, Parch, Fare, and Embarked.

### Tuned Model (Iteration 2)
1. Engineered a `Title` feature extracted from passenger names (Mr, Mrs, Miss, Master, Rare), one-hot encoded instead of label-encoded to avoid implying a false rank order.
2. Engineered a `FamilySize` feature (SibSp + Parch + 1) to capture the non-linear survival penalty for both solo travelers and large families.
3. Swapped to `GradientBoostingClassifier`, tuned with `GridSearchCV` across `n_estimators`, `max_depth`, and `learning_rate` using 5-fold cross-validation.

## Results: Before vs After

| Metric | Baseline (Random Forest) | Tuned (Gradient Boosting) | Change |
|---|---|---|---|
| Accuracy | 0.8212 | 0.8324 | +1.12 pts |
| F1-Score | 0.7746 | 0.7917 | +1.71 pts |

Best hyperparameters found: `learning_rate=0.1`, `max_depth=4`, `n_estimators=100`

## What Changed The Score

An earlier attempt at this same feature engineering, using `LabelEncoder` on `Title` instead of one-hot encoding, and untuned Gradient Boosting defaults, actually performed *worse* than the baseline (Accuracy 0.8101, F1 0.7571). Two fixes reversed that:

- One-hot encoding Title removed a false numeric ranking that ordinal encoding had implied between titles.
- GridSearchCV tuning found a deeper tree structure (max_depth=4 vs. the untuned default of 3) that was needed for the model to actually pick up on the Title/FamilySize signal without overfitting.

This is the real lesson: feature engineering only pays off when paired with correct encoding and proper tuning. Swapping algorithms or adding features blindly can make results worse before it makes them better.

## Tools Used

- Python, Pandas
- Scikit-learn (`SimpleImputer`, `LabelEncoder`, `pd.get_dummies`, `train_test_split`, `RandomForestClassifier`, `GradientBoostingClassifier`, `GridSearchCV`, `accuracy_score`, `f1_score`)

## Next in This Series

Step 2 moves into text classification (spam/ham detection) as a bridge from tabular data into NLP, ahead of a Hugging Face + PyTorch fine-tuning project.
