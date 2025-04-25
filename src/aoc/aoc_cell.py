"""
AOC Cell implementation.
"""

from logging import getLogger
from math import sqrt
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch import Tensor, nn

from .enums import MatrixConnectivityType
from .hardware_parameters import (
    CURRENT_CONFIG,
    HardwareParameters,
    Saturation16,
    Saturation48,
    from_config_file,
)
from .structured_model import StructuredAOCMatrixModel, get_structured_model

logger = getLogger()


def torch_polyval(x: Tensor, coeffs: Tensor) -> Tensor:
    """Evaluate a polynomial at specific values."""
    p = torch.zeros_like(x)
    for i, coeff in enumerate(coeffs.flip(0)):
        p += coeff * x**i
    return p


class SignalSaturation16(nn.Module):
    """
    Non-linearity for signal saturation.
    """

    def __init__(self, saturation_parameters: Saturation16):
        super().__init__()
        self.register_buffer(
            "piece0_start_stop", saturation_parameters.piece0_start_stop
        )
        self.register_buffer(
            "piece1_start_stop", saturation_parameters.piece1_start_stop
        )
        self.piece0_cosine_coeff = saturation_parameters.piece0_cosine_coeff
        self.piece1_sine_coeff = saturation_parameters.piece1_sine_coeff
        self.piece0_constant = saturation_parameters.piece0_constant
        self.piece1_multiplier = saturation_parameters.piece1_multiplier
        self.negative_saturation = saturation_parameters.negative_saturation
        self.positive_saturation = saturation_parameters.positive_saturation

    def forward(self, z: Tensor) -> Tensor:
        """
        Apply saturation to the output (input unit: V, output unit: V).

        Args:
            z (Tensor): Input to the unit cell in volt.

        Returns:
            Tensor: Output of the unit cell in volt (same shape).
        """
        piece0_start, piece0_stop = self.piece0_start_stop
        piece1_start, piece1_stop = self.piece1_start_stop

        piece0_indices = (z > piece0_start) & (z <= piece0_stop)
        piece1_indices = (z > piece1_start) & (z <= piece1_stop)

        # from -inf to piece0 start
        z = torch.where(z <= piece0_start, self.negative_saturation, z)

        # piece 0
        z = torch.where(
            piece0_indices,
            self.piece0_constant - torch.cos(self.piece0_cosine_coeff * z),
            z,
        )

        # piece 1
        z = torch.where(
            piece1_indices,
            z - self.piece1_multiplier * (torch.sin(self.piece1_sine_coeff * z) ** 5),
            z,
        )

        # from piece1 stop to +inf
        z = torch.where(z > piece1_stop, self.positive_saturation, z)

        return z


class SignalSaturation48(nn.Module):
    """
    Non-linearity for signal saturation applied at the end of the AOC cell.
    48-variable version.

        def saturation(self, v):
            for 48var split into 5 parts
            The central 3 parts have minimum fitting error,
            whereas piece 0 and piece 4 are quite rough.
            If this is too complicated. we can also remove the edge two parts,
            and use deq_clip to further concentrate more on the central parts.

        coeffs = self.all_coeffs["saturation"]
        neg_inf_indices = v <= coeffs["piece0_start_stop"][0]
        piece0_indices = (v > coeffs["piece0_start_stop"][0]) & (
            v <= coeffs["piece0_start_stop"][1]
        )
        piece1_indices = (v > coeffs["piece1_start_stop"][0]) & (
            v <= coeffs["piece1_start_stop"][1]
        )
        piece3_indices = (v > coeffs["piece3_start_stop"][0]) & (
            v <= coeffs["piece3_start_stop"][1]
        )
        piece4_indices = (v > coeffs["piece4_start_stop"][0]) & (
            v <= coeffs["piece4_start_stop"][1]
        )
        pos_inf_indices = v > coeffs["piece4_start_stop"][1]

        # from -inf to piece 0 start
        v[neg_inf_indices] = coeffs["negative_saturation"]

        # piece 0 - an inverse tick
        v[piece0_indices] = coeffs["piece0_constant"] - np.cos(
            1 / (coeffs["piece0_cosine_coeff"] * v[piece0_indices])
        )

        # piece 1 : 3 order polynomial
        piece1_poly = coeffs["piece1_polynomials"]
        v[piece1_indices] = np.polyval(piece1_poly, v[piece1_indices])

        # piece 2: y = x, no change

        # piece 3: 5 order polynomial
        piece3_poly = coeffs["piece3_polynomials"]
        v[piece3_indices] = np.polyval(piece3_poly, v[piece3_indices])

        # piece 4: cos
        v[piece4_indices] = (
            np.cos(v[piece4_indices] + coeffs["piece4_first_constant"])
            + coeffs["piece4_second_constant"]
        )
        # end of piece 4 to inf
        v[pos_inf_indices] = coeffs["positive_saturation"]

        return v
    """

    def __init__(self, saturation_parameters: Saturation48):
        super().__init__()
        self.register_buffer(
            "piece0_start_stop", saturation_parameters.piece0_start_stop
        )
        self.register_buffer(
            "piece1_start_stop", saturation_parameters.piece1_start_stop
        )
        self.register_buffer(
            "piece3_start_stop", saturation_parameters.piece3_start_stop
        )
        self.register_buffer(
            "piece4_start_stop", saturation_parameters.piece4_start_stop
        )
        self.piece0_constant = saturation_parameters.piece0_constant
        self.piece0_cosine_coeff = saturation_parameters.piece0_cosine_coeff
        self.piece1_polynomials = saturation_parameters.piece1_polynomials
        self.piece3_polynomials = saturation_parameters.piece3_polynomials
        self.piece4_first_constant = saturation_parameters.piece4_first_constant
        self.piece4_second_constant = saturation_parameters.piece4_second_constant
        self.negative_saturation = saturation_parameters.negative_saturation
        self.positive_saturation = saturation_parameters.positive_saturation

    def forward(self, z: Tensor) -> Tensor:
        """
        Apply saturation to the output (input unit: V, output unit: V).

        Args:
            z (Tensor): Input to the unit cell in volt.

        Returns:
            Tensor: Output of the unit cell in volt (same shape).
        """
        piece0_start, piece0_stop = self.piece0_start_stop
        piece1_start, piece1_stop = self.piece1_start_stop
        piece3_start, piece3_stop = self.piece3_start_stop
        piece4_start, piece4_stop = self.piece4_start_stop

        piece0_indices = (z > piece0_start) & (z <= piece0_stop)
        piece1_indices = (z > piece1_start) & (z <= piece1_stop)
        piece3_indices = (z > piece3_start) & (z <= piece3_stop)
        piece4_indices = (z > piece4_start) & (z <= piece4_stop)

        # from -inf to piece0 start
        z = torch.where(z <= piece0_start, self.negative_saturation, z)

        # piece 0
        z = torch.where(
            piece0_indices,
            self.piece0_constant - torch.cos(self.piece0_cosine_coeff * z),
            z,
        )

        # piece 1
        z = torch.where(
            piece1_indices,
            torch_polyval(z, self.piece1_polynomials),
            z,
        )

        # piece 3
        z = torch.where(
            piece3_indices,
            torch_polyval(z, self.piece3_polynomials),
            z,
        )

        # piece 4
        z = torch.where(
            piece4_indices,
            torch.cos(z + self.piece4_first_constant) + self.piece4_second_constant,
            z,
        )

        # from piece4 stop to +inf
        z = torch.where(z > piece4_stop, self.positive_saturation, z)

        return z


def create_saturation(parameters: Union[Saturation16, Saturation48]) -> nn.Module:
    """
    Create a saturation module from the parameters.
    """
    if isinstance(parameters, Saturation16):
        return SignalSaturation16(parameters)
    return SignalSaturation48(parameters)


def layer_sizes_to_module(layer_sizes: List[int]) -> nn.Sequential:
    """
    Convert a list of layer sizes to a nn.Sequential module.
    """
    assert len(layer_sizes) >= 2, "At least two layers are required."
    if len(layer_sizes) == 2:
        return nn.Linear(layer_sizes[0], layer_sizes[1])
    return nn.Sequential(
        *[
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ]
    )


class AOCCell(nn.Module):
    """
    This is a level-1 representation of the AOC cell.
    The inputs to the AOCCell need to be padded or made to the correct size beforehand,
    this is intentional to enable structured x-vectors. The size hint can be obtained
    from the matrix_size property.

    Args:
        matrix_structure (StructuredAOCMatrixModel): The matrix structure of the AOC cell.
        hardware_parameters (HardwareParameters): Hardware parameters for the AOC cell.
        alpha (float/Tensor, optional): Set a permanent alpha value, can be overwritten
        in the forward function. Alpha can be vector-valued if the vector has the same length as the state vector.
        beta (float, optional): Set a permanent beta value, can be overwritten
        in the forward function.
        gamma (float, optional): Set a permanent gamma value, can be overwritten
        in the forward function.
        apply_weight_distortion (bool, optional): Whether to apply weight distortion.
        normalise_matrix (bool, optional): Whether to normalise the matrix.
        apply_input_efficiencies (bool, optional): Whether to apply input efficiencies.
        apply_output_efficiencies (bool, optional): Whether to apply output efficiencies.
        apply_pd_crosstalk (bool, optional): Whether to apply PD crosstalk.
        apply_pbs_crosstalk (bool, optional): Whether to apply PBS crosstalk.
        apply_slm_darkness (bool, optional): Whether to apply SLM darkness.
        init_scale (float, optional): Initial scale for the weights.
    """

    def __init__(
        self,
        matrix_structure: StructuredAOCMatrixModel,
        # Non-idealities & Hardware parameters
        hardware_parameters: HardwareParameters,
        alpha: Optional[Union[float, Tensor]] = None,
        beta: Optional[Union[float, Tensor]] = None,
        gamma: Optional[float] = None,
        apply_weight_distortion: bool = True,
        normalise_matrix: bool = False,
        apply_input_efficiencies: bool = True,
        apply_output_efficiencies: bool = True,
        apply_pbs_crosstalk: bool = False,
        apply_pd_crosstalk: bool = True,
        apply_slm_darkness: bool = True,
        init_scale: float = 18.6,
    ):
        super().__init__()
        self.hardware_parameters = hardware_parameters
        self.matrix_structure = matrix_structure
        # we can override the matrix with an externally set matrix via the setter
        # in which case the externally set matrix is used
        self._externally_set_matrix: Optional[nn.Parameter] = None
        self._externally_set_bias: Optional[nn.Parameter] = None
        # train a scale parameter instead of scaling the weights
        self.scale = nn.Parameter(
            torch.tensor([init_scale]), requires_grad=normalise_matrix
        )
        # determine matrix size here so that bias etc can be initialized
        self.matrix_structure = matrix_structure
        # non-idealities
        self._normalise_matrix = normalise_matrix
        self._apply_weight_distortion = apply_weight_distortion
        self._apply_input_efficiencies = apply_input_efficiencies
        self._apply_output_efficiencies = apply_output_efficiencies
        self._apply_pd_crosstalk = apply_pd_crosstalk
        self._apply_pbs_crosstalk = apply_pbs_crosstalk
        self._apply_slm_darkness = apply_slm_darkness
        self.is_16vars = hardware_parameters.is_16vars
        # PBS crosstalk tensor
        if self.is_16vars:
            self.pbs_poly_coeffs_pos = nn.Parameter(
                hardware_parameters.pbs_crosstalk.pbs_poly_coeffs_pos,
                requires_grad=False,
            )
            self.pbs_poly_coeffs_neg = nn.Parameter(
                hardware_parameters.pbs_crosstalk.pbs_poly_coeffs_neg,
                requires_grad=False,
            )
            self.solution_to_adc_ratio = None
            self.dc_offset = None
            self.linear_offset = None
            self.scaling = None
        else:
            self.pbs_positive_to_negative = nn.Parameter(
                hardware_parameters.pbs_crosstalk.positive_to_negative,
                requires_grad=False,
            )
            self.pbs_negative_to_positive = nn.Parameter(
                hardware_parameters.pbs_crosstalk.negative_to_positive,
                requires_grad=False,
            )
            # Closed-loop correction
            self.solution_to_adc_ratio = (
                self.hardware_parameters.closed_loop_correction.solution_to_adc_ratio
            )
            self.dc_offset = nn.Parameter(
                self.hardware_parameters.closed_loop_correction.dc_offset,
                requires_grad=False,
            )
            self.linear_offset = nn.Parameter(
                self.hardware_parameters.closed_loop_correction.linear_offset,
                requires_grad=False,
            )
            self.scaling = nn.Parameter(
                self.hardware_parameters.closed_loop_correction.scaling,
                requires_grad=False,
            )

        self.z_prev: Optional[Tensor] = None

        self._register_gain(alpha, "alpha")
        self._register_gain(beta, "beta")
        self._register_gain(gamma, "gamma")

    @classmethod
    def from_module_default(
        cls,
        module: nn.Module,
        matrix_connectivity: MatrixConnectivityType = MatrixConnectivityType.FEEDBACK,
    ) -> "AOCCell":
        """
        Construct a default DEQ AOCCell from a module.
        """
        matrix_structure = get_structured_model(
            module,
            connectivity=matrix_connectivity,
        )
        is_16var = (
            matrix_structure.input_dim == 16 and matrix_structure.output_dim == 16
        )
        if is_16var:
            config_file = "models/unit_cells/aoc_cell/configs/June2024_saturation_pieces_clip2p6.yaml"
            hardware_parameters = from_config_file(config_file, is_16var=True)
        else:
            hardware_parameters = from_config_file()
        return cls(matrix_structure, hardware_parameters=hardware_parameters)

    @classmethod
    def from_parameters(
        cls,
        layer_sizes: List[int],
        connectivity: MatrixConnectivityType = MatrixConnectivityType.FEEDBACK,
        init_scale: float = 18.6,
        config_file: str = CURRENT_CONFIG,
        apply_weight_distortion: bool = True,
        apply_input_efficiencies: bool = True,
        apply_output_efficiencies: bool = True,
        apply_pd_crosstalk: bool = True,
        apply_pbs_crosstalk: bool = False,
        apply_slm_darkness: bool = True,
        normalise_matrix: bool = True,
        alpha: Optional[Union[float, Tensor]] = None,
        beta: Optional[Union[float, Tensor]] = None,
        gamma: Optional[float] = None,
    ):
        """
        Construct an AOCCell from parameters.
        This function initializes all nested necessary classes.
        """
        is_16var = layer_sizes[0] == 16 and len(layer_sizes) <= 2
        module = layer_sizes_to_module(layer_sizes)
        matrix_structure = get_structured_model(
            module,
            connectivity=connectivity,
        )
        hardware_parameters = from_config_file(config_file, is_16var=is_16var)
        return cls(
            matrix_structure,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            hardware_parameters=hardware_parameters,
            init_scale=init_scale,
            apply_weight_distortion=apply_weight_distortion,
            normalise_matrix=normalise_matrix,
            apply_input_efficiencies=apply_input_efficiencies,
            apply_output_efficiencies=apply_output_efficiencies,
            apply_pd_crosstalk=apply_pd_crosstalk,
            apply_pbs_crosstalk=apply_pbs_crosstalk,
            apply_slm_darkness=apply_slm_darkness,
        )

    def _register_gain(
        self,
        gain: Optional[Union[Tensor, np.ndarray, np.generic, int, float]],
        name: str,
    ) -> None:
        """
        Register buffer for beta.
        """
        if isinstance(gain, (float, int)):
            self.register_buffer(name, torch.tensor(self.matrix_size[0] * [gain]))
        elif isinstance(gain, (Tensor, np.ndarray, np.generic)):
            assert gain.ndim == 1, (
                "If a tensor is provided, gain must be a 1D tensor. Try providing a float."
            )
            assert gain.shape[0] == self.matrix_size[0], (
                f"{name} must have length == {self.matrix_size} (but is {gain.shape})"
            )
            self.register_buffer(name, gain)
        else:
            setattr(self, name, gain)

    def _dc_offset(self, z: Tensor) -> Tensor:
        """
        In the first iteration, we calculate the DC offset of the signal.
        """
        z_dc = torch.zeros_like(z)
        z_dc = self._aoc_tanh(z_dc)  # input = 0
        return self._uled_nonlinearity(z_dc)  # power at input=0

    @property
    def matrix(self) -> Tensor:
        """
        Get the weight matrix.
        """
        return self.matrix_structure.build_aoc_matrix()

    @matrix.setter
    def matrix(self, matrix: Tensor) -> None:
        """
        Set the weight matrix. This overrides the matrix building
        operation (e.g. for quantization).
        """
        self._externally_set_matrix = matrix

    @property
    def bias(self) -> Optional[Tensor]:
        """
        Get the bias.
        """
        return self.matrix_structure.build_aoc_bias()

    @bias.setter
    def bias(self, bias: Tensor) -> None:
        """
        Set the bias.
        """
        self._externally_set_bias = bias

    @property
    def matrix_size(self) -> Tuple[int, int]:
        """
        Get the matrix size.
        """
        return self.matrix_structure.matrix_size

    def build_aoc_vector_from_input(self, input_vector: Tensor) -> Tensor:
        """
        Pad input vector to the correct size in case the matrix is structured.
        """
        return self.matrix_structure.build_aoc_vector_from_input(input_vector)

    def get_output_from_aoc_vector(self, output_vector: Tensor) -> Tensor:
        """
        Depad output vector to the correct size in case the matrix is structured.
        """
        return self.matrix_structure.get_output_from_aoc_vector(output_vector)

    def get_distorted_weight_matrix(self, split_by_sign: bool = False) -> Tensor:
        """
        Get the distorted weight matrix, normalized to 1. if split_by_sign is True,
        the weights are split into positive and negative parts and stacked along the
        first dimension, otherwise, the weights are returned as a single tensor.
        """
        weights_pos, weights_neg = self.split_slm(self.matrix)

        common_abs_max = self.matrix.abs().max()
        # Step 4 - weight distortion
        if self._apply_weight_distortion:
            # the weights are now normalized
            weights_pos_norm, weights_neg_norm = self.weight_distortion(
                weights_pos,
                weights_neg,
                common_abs_max=common_abs_max,
            )
        else:
            weights_pos_norm = weights_pos / common_abs_max
            weights_neg_norm = weights_neg / common_abs_max

        if split_by_sign:
            weights = torch.stack([weights_pos_norm, weights_neg_norm], dim=0)
        else:
            weights = weights_pos_norm - weights_neg_norm
        return weights

    def _uled_nonlinearity(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Zero-biased LED nonlinearity.

        Args:
            z (Tensor): Input to the nonlinearity in V.
        Returns:
            Tensor: Output of the led nonlinearity in V (same shape).
        """

        def _func(z: Tensor, sign: bool) -> Tensor:
            if sign:
                a, b, c, d = self.hardware_parameters.micro_led_coefficients_pos
            else:
                a, b, c, d = self.hardware_parameters.micro_led_coefficients_neg
            return a * torch.pow(z, 3) + b * torch.pow(z, 2) + c * z + d

        z_pos = _func(z, True)
        z_neg = _func(z, False)
        return z_pos, z_neg

    def _aoc_tanh(self, z: Tensor) -> Tensor:
        """
        Specialised AOC-version of tanh.
        """
        if len(self.hardware_parameters.tanh_coefficients) == 6:
            (a, b, c, d, f, g) = self.hardware_parameters.tanh_coefficients
        elif len(self.hardware_parameters.tanh_coefficients) == 5:
            (a, b, c, d, f) = self.hardware_parameters.tanh_coefficients
            g = 0.0
        else:
            raise ValueError(
                "tanh_coefficients must be of length 5 or 6, but got "
                f"{len(self.hardware_parameters.tanh_coefficients)}"
            )
        linear_component = g * z
        z = a * z + b
        z = d * z / (1 + (torch.abs(z)) ** c) ** (1 / c) + f + linear_component
        return z

    def power_conservation(self, z: Tensor) -> Tensor:
        """
        Power conservation, apply after PD crosstalk.
        Assumes channels are in last dimension.
        """
        return z / z.shape[-1]

    def fan_in_profile(self, z_pos: Tensor, z_neg: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Optical fan-in. Takes care of optical efficiencies.

        Args:
            z (Tensor): Input to the fan-in. Shape [batch, n_vars].
            hw_parameters (HardwareParameters): Hardware parameters.

        Returns:
            Tensor: Tuple of tensors shaped [batch, n_vars].
        """
        if (
            z_pos.shape[-1] == self.hardware_parameters.n_variables
            and z_neg.shape[-1] == self.hardware_parameters.n_variables
        ):
            return (
                self.hardware_parameters.input_efficiency[0].to(z_pos.device) * z_pos,
                self.hardware_parameters.input_efficiency[1].to(z_neg.device) * z_neg,
            )
        return (
            z_pos * float(self.hardware_parameters.input_efficiency[0].mean()),
            z_neg * float(self.hardware_parameters.input_efficiency[1].mean()),
        )

    def tia_efficiencies(
        self,
        z_pos: Tensor,
        z_neg: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply output efficiencies to the positive and negative paths.
        The efficiencies are applied per variable if the shape of the
        tensors matches the number of variables, otherwise the mean
        efficiency is applied.

        Args:
            z_pos (Tensor): Positive path.
            z_neg (Tensor): Negative path.

        Returns:
            Tuple[Tensor, Tensor]: Positive and negative paths with
            efficiencies applied.
        """
        # check if shapes match otherwise use mean
        if (
            z_pos.shape[-1] == self.hardware_parameters.n_variables
            and z_neg.shape[-1] == self.hardware_parameters.n_variables
        ):
            return (
                z_pos * self.hardware_parameters.output_efficiency_pos.to(z_pos.device),
                z_neg * self.hardware_parameters.output_efficiency_neg.to(z_neg.device),
            )
        return (
            z_pos
            * self.hardware_parameters.output_efficiency_pos.mean().to(z_pos.device),
            z_neg
            * self.hardware_parameters.output_efficiency_neg.mean().to(z_neg.device),
        )

    def split_slm(self, slm: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Split the SLM into positive and negative parts.
        """
        return torch.relu(slm), torch.relu(-slm)

    def weight_distortion(
        self,
        pos_weights: Tensor,
        neg_weights: Tensor,
        common_abs_max: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Distort weights on SLM (apply contrast ratio).
        This effect affects positive and negative weights differently.
        This is a non-linear effects on the weights, so the weights cannot come
        in re-scaled or therelike.
        """

        def _weight_distortion_per_sign(
            weights: Tensor,
            params: Tensor,
            abs_max: Optional[Tensor] = None,
        ) -> Tensor:
            """
            Args:
                weights (Tensor): Weights to be distorted.
                params (Tensor): Distortion parameters, shaped [n_vars, 3].
                abs_max (Tensor, optional): Absolute maximum of the weights.
            """
            if abs_max is None:
                abs_max = torch.tensor([1.0], device=weights.device)
            weights = weights / abs_max
            a, b, c = params.t().to(weights.device)
            if weights.shape[-1] == self.hardware_parameters.n_variables:
                # apply polynomial distortion to weights
                weights = a * weights**2 + b * weights + c
            else:
                # average a, b, c and apply to all weights
                weights = (
                    float(a.mean()) * torch.pow(weights, 2)
                    + float(b.mean()) * weights
                    + float(c.mean())
                )
            return weights

        pos_weights_ = _weight_distortion_per_sign(
            pos_weights, self.hardware_parameters.weight_distortion_pos, common_abs_max
        )
        neg_weights_ = _weight_distortion_per_sign(
            neg_weights, self.hardware_parameters.weight_distortion_neg, common_abs_max
        )
        return pos_weights_, neg_weights_

    def pd_crosstalk(
        self,
        z_pos: Tensor,
        z_neg: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Jiaqi's version of photodiode crosstalk.
        """
        assert z_pos.shape[-1] == z_neg.shape[-1], (
            "z_pos and z_neg must have same shape."
        )
        n_variables_actual = z_pos.shape[-1]
        # unload parameters
        crosstalk_to_lower_pos, crosstalk_to_upper_pos = (
            self.hardware_parameters.pd_cross_talk_pos[:, 1].to(z_pos.device),
            self.hardware_parameters.pd_cross_talk_pos[:, 0].to(z_pos.device),
        )
        crosstalk_to_lower_neg, crosstalk_upper_neg = (
            self.hardware_parameters.pd_cross_talk_neg[:, 1].to(z_neg.device),
            self.hardware_parameters.pd_cross_talk_neg[:, 0].to(z_neg.device),
        )
        if self._apply_slm_darkness:
            darkness_pos = self.hardware_parameters.darkness_pos.to(z_pos.device)
            darkness_neg = self.hardware_parameters.darkness_neg.to(z_neg.device)
            if z_pos.shape[-1] != self.hardware_parameters.n_variables:
                darkness_pos = darkness_pos.mean()
                darkness_neg = darkness_neg.mean()
            z_pos_tmp = z_pos - darkness_pos
            z_neg_tmp = z_neg - darkness_neg
        else:
            z_pos_tmp = z_pos
            z_neg_tmp = z_neg

        if n_variables_actual == self.hardware_parameters.n_variables:
            crosstalk_to_lower_pos = crosstalk_to_lower_pos * z_pos_tmp
            crosstalk_to_lower_neg = crosstalk_to_lower_neg * z_neg_tmp
            crosstalk_to_upper_pos = crosstalk_to_upper_pos * z_pos_tmp
            crosstalk_to_upper_neg = crosstalk_upper_neg * z_neg_tmp
        else:
            crosstalk_to_lower_pos = crosstalk_to_lower_pos.mean() * z_pos
            crosstalk_to_lower_neg = crosstalk_to_lower_neg.mean() * z_neg
            crosstalk_to_upper_pos = crosstalk_to_upper_pos.mean() * z_pos
            crosstalk_to_upper_neg = crosstalk_upper_neg.mean() * z_neg

        crosstalk_from_higher_pos = torch.roll(crosstalk_to_lower_pos, -1, dims=-1)
        crosstalk_from_higher_pos[:, -1] = 0
        crosstalk_from_higher_neg = torch.roll(crosstalk_to_lower_neg, -1, dims=-1)
        crosstalk_from_higher_neg[:, -1] = 0
        crosstalk_from_lower_pos = torch.roll(crosstalk_to_upper_pos, 1, dims=-1)
        crosstalk_from_lower_pos[:, 0] = 0
        crosstalk_from_lower_neg = torch.roll(crosstalk_to_upper_neg, 1, dims=-1)
        crosstalk_from_lower_neg[:, 0] = 0

        total_pos = z_pos + crosstalk_from_higher_pos + crosstalk_from_lower_pos
        total_neg = z_neg + crosstalk_from_higher_neg + crosstalk_from_lower_neg

        return total_pos, total_neg

    def pbs_crosstalk(
        self,
        z_pos: Tensor,
        z_neg: Tensor,
        after_fan_in_pos: Tensor,
        after_fan_in_neg: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        PBS crosstalk for 48 variables (and beyond).
        The first order of PBS crosstalk is a leakage from the light source.
        i.e. it uses normalised output from LED instead modulated output from SLM
        as the input.
        The normalisation is applied using multiplication of output correction instead
        of division.

        norm_led_pos, norm_led_neg = self.tia_efficiency(vars_pos, vars_neg)
        crosstalk_generated_by_pos_led = norm_led_pos * self.pbs_neg_to_pos
        crosstalk_generated_by_neg_led = norm_led_neg * self.pbs_pos_to_neg
        #Ccombine crosstalk term and signal term
        v_norm_pos = v_norm_pos + crosstalk_generated_by_pos_led
        v_norm_neg = v_norm_neg + crosstalk_generated_by_neg_led
        """
        norm_led_pos, norm_led_neg = self.tia_efficiencies(
            after_fan_in_pos, after_fan_in_neg
        )
        if z_pos.shape[-1] != self.hardware_parameters.n_variables:
            pbs_negative_to_positive = self.pbs_negative_to_positive.mean()
            pbs_positive_to_negative = self.pbs_positive_to_negative.mean()
        else:
            pbs_negative_to_positive = self.pbs_negative_to_positive
            pbs_positive_to_negative = self.pbs_positive_to_negative
        crosstalk_generated_by_pos_led = norm_led_pos * pbs_negative_to_positive
        crosstalk_generated_by_neg_led = norm_led_neg * pbs_positive_to_negative
        z_pos = z_pos + crosstalk_generated_by_pos_led
        z_neg = z_neg + crosstalk_generated_by_neg_led
        return z_pos, z_neg

    def pbs_crosstalk_16(
        self,
        z_pos: Tensor,
        z_neg: Tensor,
        z_pos_after_fan_in: Tensor,
        z_neg_after_fan_in: Tensor,
        weights_pos_norm: Tensor,
        weights_neg_norm: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        The PBS leaks power from the positive path to the negative path.

        Args:
            z_pos (Tensor): Positive path.
            z_neg (Tensor): Negative path.
            weights_pos_norm (Tensor): Normalized positive weights.
            weights_neg_norm (Tensor): Normalized negative weights.
            pbs_poly_coeffs_pos (Tensor): Polynomial coefficients for positive path.
            pbs_poly_coeffs_neg (Tensor): Polynomial coefficients for negative path.
        """

        assert z_pos.shape[-1] == z_neg.shape[-1], (
            "z_pos and z_neg must have same shape."
        )
        n_variables_actual = z_pos.shape[-1]

        pbs_power_norm_factor_pos = (
            self.hardware_parameters.pbs_crosstalk.pbs_power_norm_factor_pos
        )
        pbs_power_norm_factor_neg = (
            self.hardware_parameters.pbs_crosstalk.pbs_power_norm_factor_neg
        )
        n_variables_exp = self.hardware_parameters.n_variables

        pbs_poly_coeffs_pos = self.pbs_poly_coeffs_pos.to(z_pos.device)
        pbs_poly_coeffs_neg = self.pbs_poly_coeffs_neg.to(z_neg.device)
        if n_variables_actual != n_variables_exp:
            # use broadcasting to extend mean of coefficients across a dimension of any length
            pbs_poly_coeffs_pos = pbs_poly_coeffs_pos.mean(dim=0).unsqueeze(0)
            pbs_poly_coeffs_neg = pbs_poly_coeffs_neg.mean(dim=0).unsqueeze(0)
            n_variables_exp = n_variables_actual

        if (
            weights_pos_norm.shape[-1] != n_variables_exp
            or weights_neg_norm.shape[-1] != n_variables_exp
        ):
            logger.warning(
                "Weights are not designed for the current number of variables. "
                "Turn off pbs crosstalk for convolutions."
            )
            return z_pos, z_neg

        def positive_to_negative_leakage(
            z_pos: Tensor, weights_pos_norm: Tensor
        ) -> Tensor:
            """
            z_pos: Shape [B, n]
            weights_pos_norm: Shape [n, n]
            """
            crosstalk_pattern = torch.zeros(
                (n_variables_exp, n_variables_exp),
                device=z_pos.device,
                dtype=z_pos.dtype,
            )
            mod_rms = (
                pbs_poly_coeffs_neg[:, 0] * weights_pos_norm[:, 1:] ** 2
                + pbs_poly_coeffs_neg[:, 1] * weights_pos_norm[:, 1:]
                + pbs_poly_coeffs_neg[:, 2]
            )
            crosstalk_pattern[:, 1:] = torch.fliplr(mod_rms * sqrt(2))
            return z_pos @ crosstalk_pattern / pbs_power_norm_factor_neg

        def negative_to_positive_leakage(
            z_neg: Tensor, weight_neg_norm: Tensor
        ) -> Tensor:
            """
            variables_neg: Shape [n]
            """
            crosstalk_pattern = torch.zeros(
                (n_variables_exp, n_variables_exp),
                device=z_pos.device,
                dtype=z_pos.dtype,
            )
            mod_rms = (
                pbs_poly_coeffs_pos[:, 0] * weight_neg_norm[:, 1:] ** 2
                + pbs_poly_coeffs_pos[:, 1] * weight_neg_norm[:, 1:]
                + pbs_poly_coeffs_pos[:, 2]
            )
            crosstalk_pattern[:, 1:] = torch.fliplr(mod_rms * sqrt(2))
            return z_neg @ crosstalk_pattern / pbs_power_norm_factor_pos

        crosstalk_pos_to_neg = positive_to_negative_leakage(
            z_pos_after_fan_in, weights_pos_norm
        )
        crosstalk_neg_to_pos = negative_to_positive_leakage(
            z_neg_after_fan_in, weights_neg_norm
        )
        z_pos -= crosstalk_neg_to_pos
        z_neg += crosstalk_pos_to_neg
        return z_pos, z_neg

    def matrix_operation(
        self, z_pos: Tensor, z_neg: Tensor, beta: Optional[Union[float, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        This function performs the matrix operation of the AOC cell, essentially
        the action of the SLM on the light signal, as well as the photodiodes,
        that absorb the light after the fan-in.
        """
        # Construct the matrix
        if self._externally_set_matrix is not None:
            weights = self._externally_set_matrix.to(z_pos.device)
        else:
            weights = self.matrix_structure.build_aoc_matrix().to(z_pos.device)
        # cache undistorted weights here
        weights_pos, weights_neg = self.split_slm(weights)
        common_abs_max = weights.abs().max()

        weights_pos_norm, weights_neg_norm = self.get_distorted_weight_matrix(
            split_by_sign=True
        )
        # cache signals here
        z_pos_after_fan_in = z_pos.clone()
        z_neg_after_fan_in = z_neg.clone()

        # Matrix operation: Note that AOC implements a left-side matrix multiplication
        z_pos = z_pos @ weights_pos_norm
        z_neg = z_neg @ weights_neg_norm

        # Step 7 - efficiencies
        if self._apply_output_efficiencies:
            z_pos, z_neg = self.tia_efficiencies(
                z_pos,
                z_neg,
            )
        if self.is_16vars:
            # Step 8 - PD crosstalk
            if self._apply_pd_crosstalk:
                z_pos, z_neg = self.pd_crosstalk(
                    z_pos,
                    z_neg,
                )
            # Step 9 - PBS crosstalk
            if self._apply_pbs_crosstalk:
                z_pos, z_neg = self.pbs_crosstalk_16(
                    z_pos,
                    z_neg,
                    z_pos_after_fan_in,
                    z_neg_after_fan_in,
                    # do not use the distorted weights
                    weights_pos / common_abs_max,
                    weights_neg / common_abs_max,
                )
        else:
            # The order is changed for 48 variables
            # Step 7 - PBS Crosstalk
            if self._apply_pbs_crosstalk:
                z_pos, z_neg = self.pbs_crosstalk(
                    z_pos,
                    z_neg,
                    z_pos_after_fan_in,
                    z_neg_after_fan_in,
                )
            # Step 8 - PD crosstalk
            if self._apply_pd_crosstalk:
                z_pos, z_neg = self.pd_crosstalk(
                    z_pos,
                    z_neg,
                )
        # Step 10 - Power conservation
        # we already matmul with a normalized matrix, so no division by common_abs_max
        if self._normalise_matrix:
            z_pos = self.power_conservation(z_pos)
            z_neg = self.power_conservation(z_neg)
            z_pos = z_pos * self.scale if beta is None else z_pos * beta
            z_neg = z_neg * self.scale if beta is None else z_neg * beta
        else:
            z_pos = z_pos * common_abs_max
            z_neg = z_neg * common_abs_max

        return z_pos, z_neg

    def closed_loop_correction(
        self, gradient_term: Tensor, beta: Optional[Union[float, Tensor]] = None
    ) -> Tensor:
        """
        Only for 48 variables.
        """
        if self.is_16vars:
            return gradient_term
        if beta is None:
            scale = self.scale
        else:
            scale = beta
        if not self._normalise_matrix:
            scale = self.matrix.abs().max()

        dc_offset = self.dc_offset
        linear_offset = self.linear_offset
        scaling = self.scaling
        if gradient_term.shape[-1] != self.hardware_parameters.n_variables:
            dc_offset = dc_offset.mean() if dc_offset is not None else 0.0
            linear_offset = linear_offset.mean() if linear_offset is not None else 0.0
            scaling = scaling.mean() if scaling is not None else 1.0
        offset = self.solution_to_adc_ratio * (dc_offset + scale * linear_offset)
        gradient_term = gradient_term + offset
        gradient_term = gradient_term * scaling
        return gradient_term

    def forward(
        self,
        z: Tensor,
        x: Optional[Tensor] = None,
        alpha: Optional[Union[float, Tensor]] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        momentum_vector: Optional[Tensor] = None,
        return_after_vmm: bool = False,
    ) -> Tensor:
        """
        Args:
            z (Tensor): Input to the unit cell in volt.
            Shape must be [batch, n_variables].
            x (Tensor, optional): Input to the model. Defaults to None.
            Shape must be [batch, n_variables].
            alpha (float/Tensor, optional): AOC alpha parameter. Providing alpha adds
            alpha*z to the output, not just alpha*z. Defaults to None.
            beta (Tensor, optional): AOC beta parameter,
            overrides the beta given to the class at initialization time.
            Defaults to None.
            gamma (Tensor, optional): AOC gamma parameter. Defaults to None.
            momentum_vector (Tensor, optional): Vector injected into momentum arm,
            e.g. for constant-noise diffusion. Defaults to None.
            return_after_vmm (bool, optional): Return after VMM for diagnostic purposes.
            Defaults to False.
        Returns:
            Tensor: Output of the unit cell (z_t+1).
        """
        if self.alpha is not None and alpha is None:
            alpha = self.alpha
        if self.beta is not None and beta is None:
            beta = self.beta
        if self.gamma is not None and gamma is None:
            gamma = self.gamma

        z_out = self._aoc_tanh(z)
        z_out_pos, z_out_neg = self._uled_nonlinearity(z_out)
        if self._apply_input_efficiencies:
            z_out_pos, z_out_neg = self.fan_in_profile(z_out_pos, z_out_neg)
        z_out_pos, z_out_neg = self.matrix_operation(z_out_pos, z_out_neg, beta=beta)
        if return_after_vmm:
            return torch.stack([z_out_pos, z_out_neg], dim=-1)
        z_out = z_out_pos - z_out_neg

        # for 48 variable systems, otherwise this is a no-op
        z_out = self.closed_loop_correction(z_out, beta=beta)

        if alpha is not None:
            z_out = z_out + alpha * z

        offset = torch.zeros_like(z_out)
        if x is not None:
            offset = offset + x
        if self.bias is not None:
            offset = offset + self.bias

        z_out = z_out + torch.clip(
            offset,
            self.hardware_parameters.deq_input_min_max[0],
            self.hardware_parameters.deq_input_min_max[1],
        )
        if gamma is not None:
            if self.z_prev is None:
                self.z_prev = torch.zeros_like(z).detach()
            if momentum_vector is not None:
                z_out = z_out + gamma * momentum_vector
            else:
                # standard momentum
                # z_t+1 = ... + gamma*(z_t - z_t-1)
                z_out = z_out + gamma * (z - self.z_prev)
            # just copy the values
            self.z_prev[...] = z[...]

        return z_out


def calculate_beta(
    unit_cell: AOCCell,
    quantization_scale_factor: float,
    quantization_bitwidth: int,
    delta_t: float = 1.0,
) -> float:
    """
    Calculate the AOC beta, based on the weight matrix and quantization.
    What is beta correcting for?
    The hardware will normalize the matrix to one (max element), so we need to
    scale the matrix to the correct value.

    Args:
        matrix_was_normalized (bool): Whether the matrix was normalised.
        scale_factor (float): Scale factor.
        n_vars (int): Number of variables.
        quantization_bitwidth (int): Quantization bitwidth.
        delta_t (float, optional): Time step. Defaults to 1.0.

    Returns:
        float: Beta - the scaling coefficient for the matrix.
    """
    if unit_cell._normalise_matrix:  # pylint: disable=protected-access
        beta_hardware = float(unit_cell.scale)
    else:
        n_vars = unit_cell.matrix_size[0]  # assuming square matrix
        # multiply scale with quant_scale to obtain x.abs().max() again
        quant_scale = 2 ** (quantization_bitwidth - 1)
        # in the absence of a zero-point, the scale factor is the maximum value
        matrix_abs_max = quantization_scale_factor * quant_scale
        beta_hardware = float(matrix_abs_max * n_vars / delta_t)
    # multiply with beta in model which might already be defined
    if unit_cell.beta is not None:
        beta_hardware *= float(unit_cell.beta.mean())
    return beta_hardware
