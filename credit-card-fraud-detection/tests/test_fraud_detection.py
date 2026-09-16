"""
Smoke tests for the fraud detection training script.
Verifies the module imports cleanly and exposes a callable
training entrypoint, so CodeBuild fails fast on broken commits
before ever hitting SageMaker.
"""
import importlib
import pytest


def test_module_imports():
    module = importlib.import_module("fraud_detection")
    assert module is not None


def test_expected_functions_exist():
    module = importlib.import_module("fraud_detection")
    assert hasattr(module, "train_model") or hasattr(module, "main"), (
        "fraud_detection.py must expose a train_model() or main() entrypoint "
        "for CodeBuild/CI to invoke."
    )
