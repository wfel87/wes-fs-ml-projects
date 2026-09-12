# Titanic Survival Classifier

A baseline machine learning project predicting passenger survival on the Titanic using Scikit-learn.

## What This Project Does

This is Step 1 of a portfolio pipeline moving from Scikit-learn (data cleaning and classical ML) toward PyTorch and Hugging Face (deep learning and pretrained model fine-tuning) for future projects.

## Dataset

- Source: [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- Features used: Passenger class, sex, age, siblings/spouses aboard, parents/children aboard, fare, port of embarkation

## Approach

1. **Handle missing values**: Filled missing `Age` values with the median and missing `Embarked` values with the mode using `SimpleImputer` and `fillna`.
2. **Encode categorical features**: Converted `Sex` and `Embarked` into numeric form with `LabelEncoder`.
3. **Split the data**: 80/20 train-test split using `train_test_split`.
4. **Train the model**: A `RandomForestClassifier` from Scikit-learn.
5. **Evaluate**: Measured performance with `accuracy_score` and `f1_score`.

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.8212 |
| F1-Score | 0.7746 |

**What this means:** Accuracy measures overall correct predictions. F1-Score balances precision and recall, which matters here because survival outcomes in the dataset are imbalanced (more passengers died than survived). The F1-Score being a bit lower than accuracy shows the model is slightly weaker at catching every actual survivor than the raw accuracy number suggests on its own.

## What I'd Do Differently Next

- Engineer new features such as title extracted from passenger name, and family size from `SibSp` + `Parch`.
- Try a `GradientBoostingClassifier` or tuned hyperparameters to close the gap between accuracy and F1.
- Add cross-validation instead of a single train-test split for a more stable performance estimate.

## Tools Used

- Python
- Pandas
- Scikit-learn (`SimpleImputer`, `LabelEncoder`, `train_test_split`, `RandomForestClassifier`, `accuracy_score`, `f1_score`)

## Next in This Series

Step 2 moves into text classification (spam/ham detection) as a bridge from tabular data into NLP, ahead of a Hugging Face + PyTorch fine-tuning project.
