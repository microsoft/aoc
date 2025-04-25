"""
Contains functions to build block matrices from a list of torch.Tensors and 0s.
"""

from typing import List, Union

import torch


def blockmatrix(blocks: List[List[Union[torch.Tensor, int, str]]]) -> torch.Tensor:
    """
    Creates the block matrix from a list of lists of torch.Tensors and 0s to fill the empty spaces.
    The input has to have the form of a list of lists, where each element is either a torch.Tensor or in [0, 1, "I"].
    blocks = [
        [a, 0, 0],
        [0, b, c]
    ]
    Blocks cannot overlapp, if you encounter errors check that the matrix can be written unambiguously in
    blockform and if necessary insert torch.zeros of the required shapes to remove ambiguities.
    """

    # Check that every row has the same number of columns
    num_columns = len(blocks[0])
    for i, row in enumerate(blocks):
        assert len(row) == num_columns, (
            f"Every row must have the same number of columns. Expected {num_columns} columns but got {len(row)} columns in row {i}."
        )

    # Infer the shapes of the zeros
    row_lengths = [None] * len(blocks)
    column_lengths = [None] * len(blocks[0])
    devices = []
    for i, row in enumerate(blocks):
        for j, column in enumerate(row):
            if isinstance(column, torch.Tensor):
                if row_lengths[i] is not None:
                    assert row_lengths[i] == column.shape[0], (
                        f"Block matrizes in row {i} must have same number of rows. Entry {j} has {column.shape[0]} rows but previous matrices {row_lengths[i]} rows."
                    )
                row_lengths[i] = column.shape[0]
                if column_lengths[j] is not None:
                    assert column_lengths[j] == column.shape[1], (
                        f"Block matrizes in column {j} must have same number of columns. Row {i} has {column.shape[1]} columns but previous matrices {column_lengths[j]} columns."
                    )
                column_lengths[j] = column.shape[1]
                devices.append(column.is_cuda)

        if row_lengths[i] is None:
            raise ValueError(
                f"Row {i} is empty, cannot infer shape, insert torch.zeros of the desired shape."
            )

    for j, col in enumerate(column_lengths):
        if col is None:
            raise ValueError(
                f"Column {j} is empty, cannot infer shape, insert torch.zeros of the desired shape."
            )

    device = "cuda" if any(is_cuda for is_cuda in devices) else "cpu"
    # Build the matrix
    rows = []
    for i, row in enumerate(blocks):
        columns = []
        for j, column in enumerate(row):
            if isinstance(column, torch.Tensor):
                columns.append(column.to(device))
            else:
                if column == 0:
                    columns.append(
                        torch.zeros(row_lengths[i], column_lengths[j], device=device)
                    )
                elif column == "1":
                    columns.append(
                        torch.ones(row_lengths[i], column_lengths[j], device=device)
                    )
                elif column == "I":
                    assert row_lengths[i] == column_lengths[j], (
                        f"Non-square identity of shape {(row_lengths[i], column_lengths[j])} at blockindex {(i, j)} requested, identity must be square."
                    )
                    columns.append(torch.eye(row_lengths[i], device=device))
        rows.append(torch.cat(columns, dim=1))
    return torch.cat(rows, dim=0).to(device)


def build_lower_triangular_square_matrix(
    weights: List[torch.Tensor],
) -> torch.Tensor:
    """
    Builds lower triangular block matrix for sequential model of the form
    A = [[0, 0, 0],
        [M_0, 0, 0],
        [0, M_1, 0],]
    """

    num_blocks = len(weights)
    assert num_blocks > 0, (
        "At least one matrix has to be provided to build the blockmatrix."
    )

    # The shape of the first blockrow and the last blockcolumn need to be inferred from the symmetry of the final matrix.
    # The first blockrow has to have as many rows as M_0^T. The last blockcolumn as many columns as M_{-1}^T.
    rows_in_first_blockrow = weights[0].shape[1]
    cols_in_last_blockcol = weights[-1].shape[0]

    blocks = [
        [0] * num_blocks
        + [
            torch.zeros(
                rows_in_first_blockrow, cols_in_last_blockcol, device=weights[0].device
            )
        ]
    ]
    for i in range(num_blocks):
        blocks.append([0] * i + [weights[i]] + [0] * (num_blocks - i))

    lower_block = blockmatrix(blocks)
    return lower_block


def print_blockmatrix(blocks: List[List[Union[torch.Tensor, int, str]]]) -> None:
    for block in blocks:
        print_block = []
        for b in block:
            print_block.append("M" if isinstance(b, torch.Tensor) else 0)
        print(print_block)
