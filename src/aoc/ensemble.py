"""
Time-multipexing through ensembling
===================================
"""

from typing import Callable, List, cast

import torch
from torch import Tensor, nn

from .aoc_cell import AOCCell


class Ensemble(nn.Module):
    """
    Ensemble of models. Input and output shapes are multiplied by the number of
    models participating in the ensemble.

    Args:
        models (List[nn.Module]): list of models.
        fold_fn (Callable[[Tensor], Tensor]): function to fold the input tensor.
        unfold_fn (Callable[[Tensor], Tensor]): function to unfold the output tensor.
    """

    def __init__(
        self,
        models: List[nn.Module],
        fold_fn: Callable[[Tensor], Tensor],
        unfold_fn: Callable[[Tensor], Tensor],
    ):
        super().__init__()
        assert len(models) > 0, "Ensemble must have at least one model"
        self.models = nn.ModuleList(models)
        self.fold_fn = fold_fn
        self.unfold_fn = unfold_fn

    def forward(self, z: Tensor, x: Tensor) -> Tensor:
        """
        Forward pass. Input and output shapes are the same.
        """
        x = self.fold_fn(x)
        z = self.fold_fn(z)

        combined_result = torch.stack(
            [model(z[i], x[i]) for i, model in enumerate(self.models)], dim=0
        )

        return self.unfold_fn(combined_result)
