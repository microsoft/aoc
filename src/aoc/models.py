"""
Models to wrap AOCCell
======================
"""

from functools import partial
from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from torch import Tensor
from torchdeq import get_deq, reset_deq

from .aoc_cell import AOCCell, create_saturation
from .ensemble import Ensemble
from .enums import MatrixConnectivityType
from .simple_cell import SimpleBlock, SimpleCell
from .utils import get_activation_function


class DEQInputOutputProjection(nn.Module):
    """
    Default DEQ class which sandwiches the DEQ between an input and output projection.
    The DEQ can be turned into an Ensemble where several independent DEQs are run in parallel.

    Args:
        d_in (int): dimension of input.
        d_hidden (int): input is projected to a hidden space
        whose dimension is d_hidden.
        d_out (int): number of outputs (classes).
        unit_cell_name (str): name of the unit cell (SimpleCell or AOCCell).
        unit_cell_parameters (Dict[str, Any]): parameters of the unit cell.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: List[int],
        d_out: int,
        unit_cell_name: str,
        unit_cell_parameters: Dict[str, Any],
        deq_parameters: Dict[str, Any],
        n_ensemble: int = 1,
        train_proj2hidden: bool = True,
        input_act_func: str = "none",
        apply_input_layer_norm: bool = False,
        apply_input_norm: bool = False,
        apply_output_saturation: bool = False,
    ):
        super().__init__()
        self.hidden_size = sum(d_hidden)
        self.first_hidden_size = d_hidden[0]
        self.last_hidden_size = d_hidden[-1]

        self.deq = get_deq(deq_parameters)

        self.n_ensemble = n_ensemble
        if self.n_ensemble > 1:
            assert len(d_hidden) <= 2, (
                "Ensemble DEQ is only implemented for a single dense matrix. "
                f"Got {len(d_hidden)} hidden layers."
            )
            cells = []
            for _ in range(self.n_ensemble):
                cell = self._create_cell(
                    unit_cell_name,
                    d_hidden,
                    unit_cell_parameters,
                )
                cells.append(cell)

            cell = Ensemble(
                cells,
                fold_fn=lambda t: self.fold_ensemble_into_first_dim(t, self.n_ensemble),
                unfold_fn=self.unfold_back_to_ensemble,
            )

            d_deq_in = d_hidden[0] * self.n_ensemble
            d_deq_out = d_hidden[-1] * self.n_ensemble
            assert d_deq_in == d_deq_out
        else:
            cell = self._create_cell(
                unit_cell_name,
                d_hidden,
                unit_cell_parameters,
            )
            d_deq_in = d_hidden[0]
            d_deq_out = d_hidden[-1]

        self.cell = cell
        in_act_func = get_activation_function(input_act_func)
        input_layer_norm = (
            nn.LayerNorm(d_hidden[0]) if apply_input_layer_norm else nn.Identity()
        )
        self.input_projection = nn.Sequential(
            nn.Linear(d_in, d_deq_in),
            in_act_func,
            input_layer_norm,
        )
        self.apply_input_norm = apply_input_norm
        if not train_proj2hidden:
            for param in self.input_projection.parameters():
                param.requires_grad = False

        if apply_output_saturation:
            if isinstance(cell, AOCCell):
                self.output_act_func = create_saturation(
                    cell.hardware_parameters.saturation
                )
            elif isinstance(cell, Ensemble) and isinstance(cell.models[0], AOCCell):
                self.output_act_func = create_saturation(
                    cell.models[0].cell_parameters.saturation
                )
            else:
                raise NotImplementedError(
                    "Output saturation is only implemented for AOCCell or an ensemble of AOCCells."
                )
        else:
            # even it self.apply_output_saturation==False, we still need to declare this here
            # since this function is always applied in the emulator export process
            self.output_act_func = nn.Identity()

        self.output_projection = nn.Linear(d_deq_out, d_out)

    @classmethod
    def create_default_aoc_model(
        cls,  # type: ignore
        d_in: int,
        d_hidden: List[int],
        d_out: int,
        n_ensemble: int = 1,
        connectivity: MatrixConnectivityType = MatrixConnectivityType.FEEDBACK,
    ) -> "DEQInputOutputProjection":
        """
        Create a default AOCCell model with the given parameters.
        """
        unit_cell_name = "aoccell"
        unit_cell_parameters = {
            "apply_input_efficiencies": True,
            "apply_weight_distortion": True,
            "apply_output_efficiencies": True,
            "apply_pd_crosstalk": True,
            "apply_pbs_crosstalk": False,
            "apply_slm_darkness": True,
            "normalise_matrix": True,
            "init_scale": 18.6,
            "alpha": 0.5,
            "connectivity": connectivity,
        }

        default_config_path = files("aoc.deq_config") / "default.yaml"
        with open(default_config_path, "r", encoding="UTF-8") as f:
            deq_parameters = yaml.safe_load(f)

        return cls(
            d_in=d_in,
            d_hidden=d_hidden,
            d_out=d_out,
            unit_cell_name=unit_cell_name,
            unit_cell_parameters=unit_cell_parameters,
            deq_parameters=deq_parameters,
            n_ensemble=n_ensemble,
        )

    @classmethod
    def create_simple_cell_model(
        cls,  # type: ignore
        d_in: int,
        d_hidden: List[int],
        d_out: int,
        n_ensemble: int = 1,
    ) -> "DEQInputOutputProjection":
        """
        Create a default SimpleCell model with the given parameters.
        """
        unit_cell_name = "simplecell"
        unit_cell_parameters = {
            "act_func": "tanh",
            "act_func_first": False,
        }

        default_config_path = files("aoc.deq_config") / "default.yaml"
        with open(default_config_path, "r", encoding="UTF-8") as f:
            deq_parameters = yaml.safe_load(f)

        return cls(
            d_in=d_in,
            d_hidden=d_hidden,
            d_out=d_out,
            unit_cell_name=unit_cell_name,
            unit_cell_parameters=unit_cell_parameters,
            deq_parameters=deq_parameters,
            n_ensemble=n_ensemble,
        )

    def _create_cell(
        self,
        unit_cell_name: str,
        d_hidden: List[int],
        unit_cell_parameters: Dict[str, Any],
    ) -> nn.Module:
        """
        Create a cell for the DEQInputOutputProjection.
        """
        if unit_cell_name.lower() == "simplecell":
            cell = SimpleBlock(
                SimpleCell,
                d_hidden,
                unit_cell_parameters,
            )
        elif unit_cell_name.lower() == "aoccell":
            cell = AOCCell.from_parameters(
                layer_sizes=d_hidden,
                **unit_cell_parameters,
            )
        else:
            raise NotImplementedError(
                f"unit_cell_name {unit_cell_name} not implemented yet."
            )
        return cell

    @staticmethod
    def fold_ensemble_into_first_dim(x: Tensor, n_ensemble: int) -> Tensor:
        """
        Folding function explaining how an ensemble would organise the shape.
        """
        if x.ndim > 2:
            x = x.squeeze(1)
        return rearrange(x, "b (n h) -> n b h", n=n_ensemble)

    @staticmethod
    def unfold_back_to_ensemble(x: Tensor) -> Tensor:
        """
        Undo fold_ensemble_into_first_dim.
        """
        return rearrange(x, "n b h -> b (n h)")

    def is_ensemble(self) -> bool:
        """
        Boolean property to test if the model is an ensemble model,
        i.e. if it has multiple unit cells run in parallel.
        """
        return isinstance(self.cell, Ensemble)

    def forward(  # type: ignore
        self, x: Tensor, fixed_point_init: Optional[str] = "x_proj", **kwargs
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Forward pass of the model.
        """
        reset_deq(self.deq)

        if x.ndim > 2:
            x = torch.flatten(x, start_dim=1)
        x_proj = self.input_projection(x)

        if fixed_point_init is not None:
            if fixed_point_init == "x_proj":
                z_0 = x_proj.clone().detach()
            elif fixed_point_init == "zeros":
                z_0 = torch.zeros_like(x_proj)
            elif fixed_point_init == "random":
                z_0 = 2 * torch.rand_like(x_proj) - 1
            else:
                raise NotImplementedError(
                    "fixed point initialization "
                    f"{fixed_point_init} not implemented yet."
                )
        else:
            z_0 = None
        if self.apply_input_norm:
            x_proj = F.normalize(x_proj, p=2, dim=-1)
        if isinstance(self.cell, AOCCell):
            if z_0 is not None:
                z_0 = self.cell.build_aoc_vector_from_input(z_0)
                z_0 += self.cell.bias if self.cell.bias is not None else 0
                z_0 = torch.clip(
                    z_0,
                    self.cell.hardware_parameters.deq_input_min_max[0],
                    self.cell.hardware_parameters.deq_input_min_max[1],
                )
            x_proj = self.cell.build_aoc_vector_from_input(x_proj)

        iter_func = partial(self.cell, x=x_proj)
        deq_out, iter_info = self.deq(iter_func, z_0, **kwargs)
        z_star = deq_out[-1]
        if isinstance(self.cell, AOCCell):
            z_star = self.cell.get_output_from_aoc_vector(z_star)
        z_star = self.output_act_func(z_star)
        x_out = self.output_projection(z_star)
        return x_out, iter_info
