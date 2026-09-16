"""
Smoke tests for the SMS spam classifier pipeline.
"""
from sms_spam_classifier import build_pipeline


def test_build_pipeline_returns_pipeline():
    pipeline = build_pipeline()
    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "predict")


def test_pipeline_trains_on_toy_data():
    pipeline = build_pipeline()
    texts = ["win a free prize now", "hey are we still on for lunch",
              "free free free click here", "see you at the meeting tomorrow"]
    labels = [1, 0, 1, 0]
    pipeline.fit(texts, labels)
    preds = pipeline.predict(texts)
    assert len(preds) == len(labels)
