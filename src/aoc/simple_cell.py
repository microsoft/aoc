"""
SimpleCell
==========
A standard torch cell to test against the AOC cell.
"""

from typing import Any, Dict, List, Optional, Type

import torch.nn as nn
from torch import Tensor

from .utils import get_activation_function


class SimpleBlock(nn.Module):
    """
    class for constructing a block of SimpleCells.
    This enables stacking multiple SimpleCells to simulate a multilayer self-recurrent network.
    """

    def __init__(
        self,
        unit_cell: nn.Module,
        d_hidden: List[int],
        unit_cell_parameters: Dict[str, Any],
    ):
        """
        Args:
        unit_cell (nn.Module): unit cell of which the block is composed of.
        d_hidden (List[int]):  list of hidden dimension(s) in the block
        unit_cell_parameters (Dict[str, Any]): additional parameters of the unit cell.
        """
        super().__init__()

        self.d_hidden = d_hidden
        self.num_cells = len(self.d_hidden) - 1
        self.recur_cell = self._make_block(unit_cell, unit_cell_parameters)
        self.f_log = None

    def _make_block(
        self,
        unit_cell: Type[nn.Module],
        unit_cell_parameters: Dict[str, Any],
    ):
        """
        constructing the hidden block of feedforward with unit_cells
        """
        layers = []
        for i in range(self.num_cells):
            layers.append(
                unit_cell(
                    d_hidden=self.d_hidden[i],
                    d_out=self.d_hidden[i + 1],
                    **unit_cell_parameters,
                )
            )
        return nn.ModuleList(layers)

    def forward(
        self,
        z: Tensor,
        x: Optional[Tensor] = None,
    ):
        """
        Args:
            z (Tensor): Input to the block z is the latent DEQ vector.
            In feedforward z is normal intermediate values and x is None.
            x (Optional[Tensor]): optional
            input to the unit cell. Default 0.
            save_intermediates (Optional[bool]): whether to save intermediate activities?
        """

        # only first layer gets injection
        for i in range(0, self.num_cells):
            z = self.recur_cell[i](z, x=x if i == 0 else None)
        return z


class SimpleCell(nn.Module):
    """
    A simple unit cell example which consists of FCN and ReLU.
    """

    def __init__(
        self,
        d_hidden: int = 90,
        d_out: Optional[int] = None,
        act_func_first: bool = False,
        act_func: str = "tanh",
    ):
        """
        Args:
            d_hidden (int): input-output dimension of FCN. Defaults to 90.
            d_out (int): output dimension. Overrides d_hidden if not None.
            act_func_first (bool): whether to apply activation function before MatMul or not?
            act_func (str): activation function to use. Defaults to "relu".
        """
        super().__init__()
        self.act_func_first = act_func_first
        d_out = d_hidden if not d_out else d_out
        self.linear = nn.Linear(d_hidden, d_out)
        self.actv = get_activation_function(act_func)

    def forward(self, z: Tensor, x: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            z (Tensor): Starting point for iteration.
            x (Tensor, optional): Input to the unit cell.
        Returns:
            Tensor: Output of the unit cell (z_t+1).
        """

        if self.act_func_first:
            z = self.actv(z)
        cell_signal = self.linear(z)
        if x is not None:
            cell_signal = cell_signal + x

        if not self.act_func_first:
            cell_signal = self.actv(cell_signal)
        return cell_signal
