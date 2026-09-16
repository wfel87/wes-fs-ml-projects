"""
Smoke tests for the Titanic PyTorch model definition and data prep.
"""
import torch
from titanic_neural_network import TitanicNet


def test_model_forward_pass_shape():
    model = TitanicNet(input_dim=5)
    dummy_input = torch.randn(4, 5)
    output = model(dummy_input)
    assert output.shape == (4, 1)


def test_model_output_is_probability():
    model = TitanicNet(input_dim=5)
    dummy_input = torch.randn(4, 5)
    output = model(dummy_input)
    assert torch.all(output >= 0) and torch.all(output <= 1)
