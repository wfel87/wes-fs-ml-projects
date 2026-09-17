"""
Smoke tests for the Titanic PyTorch model definition and training entrypoints.
"""
import torch
from titanic_neural_network import TitanicNet


def test_model_forward_pass_shape():
    model = TitanicNet(input_size=7)
    dummy_input = torch.randn(4, 7)
    output = model(dummy_input)
    assert output.shape == (4, 1)


def test_model_output_is_probability():
    model = TitanicNet(input_size=7)
    dummy_input = torch.randn(4, 7)
    output = model(dummy_input)
    assert torch.all(output >= 0) and torch.all(output <= 1)


def test_model_architecture_matches_spec():
    model = TitanicNet(input_size=7)
    assert model.layer1.out_features == 16
    assert model.layer2.out_features == 8
    assert model.output.out_features == 1
