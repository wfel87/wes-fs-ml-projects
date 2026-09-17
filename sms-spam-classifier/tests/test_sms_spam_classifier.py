"""
Smoke tests for the SMS spam classifier baseline and tuned pipelines.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sms_spam_classifier import build_vectorizer, train_baseline, train_tuned


def test_build_vectorizer_matches_spec():
    vectorizer = build_vectorizer()
    assert isinstance(vectorizer, TfidfVectorizer)
    assert vectorizer.max_features == 3000
    assert vectorizer.stop_words == "english"


def test_baseline_and_tuned_train_on_toy_data():
    texts_train = ["win a free prize now", "hey are we still on for lunch",
                   "free free free click here", "see you at the meeting tomorrow",
                   "claim your free cash prize", "let's grab coffee later"]
    labels_train = [1, 0, 1, 0, 1, 0]
    texts_test = ["free prize winner", "meeting at noon"]
    labels_test = [1, 0]

    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(texts_train)
    X_test_tfidf = vectorizer.transform(texts_test)

    _, baseline_acc, _ = train_baseline(X_train_tfidf, labels_train, X_test_tfidf, labels_test)
    assert 0.0 <= baseline_acc <= 1.0
