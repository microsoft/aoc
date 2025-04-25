"""
Unit tests for wrapper models.
"""

import torch

from aoc import (
    DEQInputOutputProjection,
    MatrixConnectivityType,
)


def test_deq_input_output_projection_single_layer():
    """
    Test the DEQInputOutputProjection class. This tests tests
    - input and return shapes
    - backwardability
    - default model creation without ensemble
    """

    model = DEQInputOutputProjection.create_default_aoc_model(
        d_in=1,
        d_hidden=[48, 48],
        d_out=1,
    )
    test_input = torch.randn(1, 1)
    test_output, _ = model(test_input)
    assert test_output.shape == (1, 1), (
        f"Expected output shape (1, 1), got {test_output.shape}"
    )
    test_output.backward()
    assert model.input_projection[0].weight.grad is not None, "Gradients are not None"


def test_deq_input_output_projection_multi_layer():
    """
    Test the DEQInputOutputProjection class. This tests tests
    - input and return shapes
    - backwardability
    - default model creation without ensemble
    """

    model = DEQInputOutputProjection.create_default_aoc_model(
        d_in=1,
        d_hidden=[24, 24, 24],
        d_out=1,
    )
    test_input = torch.randn(1, 1)
    test_output, _ = model(test_input)
    assert test_output.shape == (1, 1), (
        f"Expected output shape (1, 1), got {test_output.shape}"
    )
    test_output.backward()
    assert model.input_projection[0].weight.grad is not None, "Gradients are not None"


def test_deq_input_output_projection_ensemble():
    """
    Test the DEQInputOutputProjection class with an ensemble of AOCCells.
    This tests tests
    - input and return shapes
    - backwardability
    - default model creation with ensemble
    """

    model = DEQInputOutputProjection.create_default_aoc_model(
        d_in=1,
        d_hidden=[48, 48],
        d_out=1,
        n_ensemble=2,
    )
    test_input = torch.randn(1, 1)
    test_output, _ = model(test_input)
    assert test_output.shape == (1, 1), (
        f"Expected output shape (1, 1), got {test_output.shape}"
    )
    test_output.backward()
    assert model.input_projection[0].weight.grad is not None, "Gradients are not None"


def test_deq_input_output_projection_hopfield():
    """
    Test the DEQInputOutputProjection class with a Hopfield structure.
    This tests tests
    - input and return shapes
    - backwardability
    """

    model = DEQInputOutputProjection.create_default_aoc_model(
        d_in=1,
        d_hidden=[24, 24, 24],
        d_out=1,
        n_ensemble=1,
        connectivity=MatrixConnectivityType.HOPFIELD,
    )
    test_input = torch.randn(1, 1)
    test_output, _ = model(test_input)
    assert test_output.shape == (1, 1), (
        f"Expected output shape (1, 1), got {test_output.shape}"
    )
    test_output.backward()
    assert model.input_projection[0].weight.grad is not None, "Gradients are not None"
