"""
Smoke tests for the Titanic survival classifier script.
"""
import pandas as pd
from titanic_survival_classifier import engineer_features, build_pipeline


def test_engineer_features_adds_columns():
    df = pd.DataFrame({
        "Name": ["Smith, Mr. John"],
        "SibSp": [1],
        "Parch": [0],
    })
    result = engineer_features(df)
    assert "Title" in result.columns
    assert "FamilySize" in result.columns
    assert result["FamilySize"].iloc[0] == 2


def test_build_pipeline_returns_pipeline():
    pipeline = build_pipeline(["Age", "Fare"], ["Sex", "Pclass"])
    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "predict")
