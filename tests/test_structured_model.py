"""
Unit tests for the structured models.
"""

from typing import Type

import numpy as np
import pytest
import torch

from aoc import (
    FeedbackAOCModel,
    FeedforwardAOCModel,
    HopfieldAOCModel,
    MatrixConnectivityType,
    StructuredAOCMatrixModel,
    get_structured_model,
)


@pytest.mark.parametrize(
    "model_class",
    [FeedforwardAOCModel, FeedbackAOCModel, HopfieldAOCModel],
)
def test_multilayer_aoc(model_class: Type[StructuredAOCMatrixModel]):
    """
    Just a smoke test to see if the model can be built.
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.Linear(2, 2, bias=False),
    )

    aoc_model = model_class.from_torch_model(model)
    aoc_model.build_aoc_matrix()
    aoc_model.build_aoc_bias()

    x = torch.randn(5, 2)
    aoc_model(x)
    print("Tests passed.")


@pytest.mark.parametrize("model_class", [FeedforwardAOCModel, FeedbackAOCModel])
def test_model_output(model_class: Type[StructuredAOCMatrixModel]):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(2, 2, bias=True),
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.Tanh(),
    )

    aoc_model = model_class.from_torch_model(model)

    x = torch.randn(1, 2)
    model_output = model(x).detach().numpy()
    aoc_forward = aoc_model(x).detach().numpy()
    assert np.allclose(model_output, aoc_forward, atol=1e-5), "Outputs do not match."


@pytest.mark.parametrize(
    "model_class",
    [
        FeedforwardAOCModel,
        FeedbackAOCModel,
        HopfieldAOCModel,
    ],
)
def test_from_aoc_components(model_class: Type[StructuredAOCMatrixModel]):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(2, 2, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(2, 2, bias=True),
        torch.nn.Tanh(),
    )

    aoc_model = model_class.from_torch_model(model)

    aoc_model_recon = model_class.from_aoc_components(
        aoc_model.build_aoc_matrix(),
        aoc_model.build_aoc_bias(),
        aoc_model.apply_tanh_to_layer,
        aoc_model.layer_shapes,
    )

    x = torch.randn(1, 2)
    aoc_forward = aoc_model(x).detach().numpy()
    aoc_forward_recon = aoc_model_recon(x).detach().numpy()
    assert np.allclose(aoc_forward_recon, aoc_forward, atol=1e-5), (
        "Outputs do not match."
    )


@pytest.mark.parametrize(
    "connectivity",
    [
        MatrixConnectivityType.FEEDFORWARD,
        MatrixConnectivityType.FEEDBACK,
        MatrixConnectivityType.HOPFIELD,
    ],
)
def test_get_structured_model(connectivity: MatrixConnectivityType):
    """
    Test if the get_structured_model function can build a model.
    """
    module = torch.nn.Linear(2, 2, bias=True)
    model = get_structured_model(module, connectivity)
    _ = model.matrix_size


@pytest.mark.parametrize("model_class", [FeedforwardAOCModel, FeedbackAOCModel])
def test_single_layer(model_class: Type[StructuredAOCMatrixModel]):
    """
    Test if the model can be built from a single layer.
    """
    model = torch.nn.Linear(2, 2, bias=False)
    aoc_model = model_class.from_torch_model(model, apply_tanh_to_layer=[False])
    # test access to inherited property
    assert (2, 2) == aoc_model.matrix_size, "Matrix size does not match."
    x = torch.randn(5, 2)
    assert torch.allclose(model(x), aoc_model(x), atol=1e-5), "Outputs do not match."


@pytest.mark.parametrize(
    "model_class",
    [FeedforwardAOCModel, FeedbackAOCModel, HopfieldAOCModel],
)
def test_single_layer_from_aoc_components(model_class: Type[StructuredAOCMatrixModel]):
    model = torch.nn.Linear(2, 2, bias=True)
    aoc_model = model_class.from_torch_model(model, apply_tanh_to_layer=[False])
    aoc_model2 = model_class.from_aoc_components(
        aoc_model.build_aoc_matrix(),
        aoc_model.build_aoc_bias(),
        aoc_model.apply_tanh_to_layer,
        aoc_model.layer_shapes,
    )

    x = torch.randn(5, 2)
    assert torch.allclose(aoc_model(x), aoc_model2(x), atol=1e-5), (
        "Outputs do not match."
    )
    assert torch.allclose(aoc_model(x), aoc_model2(x), atol=1e-5), (
        "Outputs do not match."
    )


@pytest.mark.parametrize("model_class", [HopfieldAOCModel])
def test_treat_single_layer_as_multi(model_class: Type[StructuredAOCMatrixModel]):
    """
    Test if the treat_single_layer_as_multi_layer flag works.
    """
    model = torch.nn.Linear(2, 2, bias=False)
    aoc_model = model_class.from_torch_model(
        model, treat_single_layer_as_multi_layer=False
    )

    aoc_model_recon = model_class.from_aoc_components(
        aoc_model.build_aoc_matrix(),
        aoc_model.build_aoc_bias(),
        aoc_model.apply_tanh_to_layer,
        aoc_model.layer_shapes,
        aoc_model.treat_single_layer_as_multi_layer,
    )

    assert torch.allclose(
        aoc_model.build_aoc_matrix(), aoc_model_recon.build_aoc_matrix(), atol=1e-5
    ), "Outputs do not match."
