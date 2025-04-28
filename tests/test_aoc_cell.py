"""
Unit tests for the aoc_cell module.
"""

import pytest
import torch
import torch.nn as nn

from aoc import AOCCell, MatrixConnectivityType, from_config_file, get_structured_model


def test_aoc_cell_outputs():
    """
    Test AOCCell outputs to alert to changes.
    """
    # test the forward pass with bias in matrix.
    torch.manual_seed(0)
    n_variables = 16
    x = torch.randn(1, n_variables, requires_grad=False)
    z_0 = torch.randn(1, n_variables)
    module = torch.nn.Linear(n_variables, n_variables)
    matrix_structure = get_structured_model(
        module,
        connectivity=MatrixConnectivityType.FEEDBACK,
    )
    hw_parameters = from_config_file()
    cell = AOCCell(matrix_structure, hw_parameters)
    out = cell(z_0, x)
    print(out)
    z_target = torch.tensor(
        [
            [
                -1.1951,
                -1.1453,
                -0.3982,
                -0.3322,
                0.9118,
                0.6872,
                -0.3485,
                -1.2874,
                0.4871,
                -1.2927,
                0.4424,
                0.2267,
                -0.0112,
                1.2847,
                0.9983,
                -0.1617,
            ]
        ]
    )
    assert out.shape == (
        1,
        n_variables,
    ), f"Expected shape {(1, n_variables)}, got {out.shape}"
    torch.testing.assert_close(out, z_target, atol=1e-3, rtol=1e-3)


def test_aoc_cell_beta_power_conservation():
    """
    Test beta-matrix-max reinjection in forward function of aoc cell.
    """
    torch.manual_seed(0)
    n_parameters = 16
    module = torch.nn.Linear(n_parameters, n_parameters)
    matrix_structure = get_structured_model(
        module,
        connectivity=MatrixConnectivityType.FEEDBACK,
    )
    hw_parameters = from_config_file()
    cell_unnorm = AOCCell(
        matrix_structure,
        hw_parameters,
        normalise_matrix=True,
    )
    cell_unnorm.scale.data[...] = 2.0
    cell_norm = AOCCell(
        matrix_structure,
        hw_parameters,
        normalise_matrix=True,
    )
    scale = cell_unnorm.scale.data.item()
    x = torch.randn(1, n_parameters)
    z = torch.randn(1, n_parameters)
    unnorm_result = cell_unnorm(z.clone(), x)
    norm_result = cell_norm(z, x, beta=scale)
    assert norm_result.shape == (1, n_parameters)
    assert unnorm_result.shape == (1, n_parameters)
    torch.testing.assert_close(unnorm_result, norm_result, atol=1e-6, rtol=1e-6)


def test_aoccell_alpha_vector():
    """
    This unit-test makes sure to test alpha as a vector and the tanh mask.
    """
    torch.manual_seed(0)
    n_parameters = 16
    matrix_structure = get_structured_model(
        nn.Linear(n_parameters, n_parameters),
        connectivity=MatrixConnectivityType.FEEDBACK,
    )
    hw_parameters = from_config_file()
    test_cell = AOCCell(
        matrix_structure,
        hw_parameters,
        alpha=torch.tensor([1.0] * n_parameters),
    )
    x = torch.randn(1, n_parameters)
    z = torch.randn(1, n_parameters)
    out = test_cell(z, x)
    assert out.shape == (1, n_parameters)
    # test backprobability
    out.sum().backward()


@pytest.mark.parametrize(
    "connectivity,expected_matrix_size",
    [
        (MatrixConnectivityType.FEEDBACK, 96),
        (MatrixConnectivityType.FEEDFORWARD, 112),
    ],
)
def test_aoccell_multi_layer(
    connectivity: MatrixConnectivityType,
    expected_matrix_size: int,
):
    """
    Test the multi-layer capabilities inside AOCCell.
    """
    torch.manual_seed(0)
    d_hidden = [16, 32, 48, 16]  # 3 layers
    cell = AOCCell.from_parameters(
        d_hidden,
        connectivity=connectivity,
    )
    assert cell.matrix_size[0] == expected_matrix_size, (
        f"Expected matrix size {expected_matrix_size}, got {cell.matrix_size}"
    )
    x = torch.randn(1, d_hidden[0])
    z = torch.randn(1, d_hidden[0])
    x_padded = torch.cat((x, torch.zeros(1, expected_matrix_size - d_hidden[0])), dim=1)
    z_padded = torch.cat((z, torch.zeros(1, expected_matrix_size - d_hidden[0])), dim=1)
    out = cell(z_padded, x_padded)
    assert out.shape == (1, expected_matrix_size)
    # test backprobability
    out.sum().backward()
