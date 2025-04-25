"""
Hardware parameters for the AOC system.
=======================================
This class contains the parameters characterising the hardware.
"""

from importlib.resources import files
from typing import Any, Dict, List, NamedTuple, Optional, Union

import torch
import yaml  # type: ignore
from torch import Tensor

CONFIG_BASE_PATH = files("aoc.hw_config")
CURRENT_CONFIG = CONFIG_BASE_PATH / "April2025_corrected_pd_3.yaml"
LAST_16_VARS_CONFIG = CONFIG_BASE_PATH / "June2024_saturation_pieces_clip2p6.yaml"


class Saturation16(NamedTuple):
    """
    Saturation parameters for 16 variables.
    """

    # Saturation Parameters
    negative_saturation: float
    positive_saturation: float

    piece0_constant: float
    piece0_cosine_coeff: float
    piece0_start_stop: Tensor

    piece1_multiplier: float
    piece1_sine_coeff: float
    piece1_start_stop: Tensor

    @classmethod
    def from_yaml(cls, saturation_parameters: Dict[str, Any]) -> "Saturation16":
        if (
            "piece0_start_stop" not in saturation_parameters
            or "piece0_cosine_coeff" not in saturation_parameters
        ):
            raise ValueError("Possibly not a 16-variable config.")
        return cls(
            negative_saturation=float(saturation_parameters["negative_saturation"]),
            positive_saturation=float(saturation_parameters["positive_saturation"]),
            piece0_constant=float(saturation_parameters["piece0_constant"]),
            piece0_cosine_coeff=torch.tensor(
                saturation_parameters["piece0_cosine_coeff"], dtype=torch.float32
            ),
            piece0_start_stop=torch.tensor(
                saturation_parameters["piece0_start_stop"], dtype=torch.float32
            ),
            piece1_start_stop=torch.tensor(
                saturation_parameters["piece1_start_stop"], dtype=torch.float32
            ),
            piece1_multiplier=torch.tensor(
                saturation_parameters["piece1_multiplier"], dtype=torch.float32
            ),
            piece1_sine_coeff=torch.tensor(
                saturation_parameters["piece1_sine_coeff"], dtype=torch.float32
            ),
        )


class Saturation48(NamedTuple):
    """
    Saturation parameters for 48 variables.
    """

    negative_saturation: float
    piece0_start_stop: Tensor
    piece0_constant: float
    piece0_cosine_coeff: float
    piece1_start_stop: Tensor
    piece1_polynomials: Tensor
    piece3_start_stop: Tensor
    piece3_polynomials: Tensor
    piece4_start_stop: Tensor
    piece4_first_constant: float
    piece4_second_constant: float
    positive_saturation: float

    @classmethod
    def from_yaml(cls, saturation_parameters: Dict[str, Any]) -> "Saturation48":
        return cls(
            negative_saturation=saturation_parameters["negative_saturation"],
            piece0_start_stop=torch.tensor(
                saturation_parameters["piece0_start_stop"], dtype=torch.float32
            ),
            piece0_constant=saturation_parameters["piece0_constant"],
            piece0_cosine_coeff=saturation_parameters["piece0_cosine_coeff"],
            piece1_start_stop=torch.tensor(
                saturation_parameters["piece1_start_stop"], dtype=torch.float32
            ),
            piece1_polynomials=torch.tensor(
                saturation_parameters["piece1_polynomials"], dtype=torch.float32
            ),
            piece3_start_stop=torch.tensor(
                saturation_parameters["piece3_start_stop"], dtype=torch.float32
            ),
            piece3_polynomials=torch.tensor(
                saturation_parameters["piece3_polynomials"], dtype=torch.float32
            ),
            piece4_start_stop=torch.tensor(
                saturation_parameters["piece4_start_stop"], dtype=torch.float32
            ),
            piece4_first_constant=saturation_parameters["piece4_first_constant"],
            piece4_second_constant=saturation_parameters["piece4_second_constant"],
            positive_saturation=saturation_parameters["positive_saturation"],
        )


class PBSCrosstalkParameters16(NamedTuple):
    """
    PBS crosstalk parameters for 16 variables.
    """

    pbs_poly_coeffs_pos: Tensor
    pbs_power_norm_factor_pos: float
    # Negative
    pbs_poly_coeffs_neg: Tensor
    pbs_power_norm_factor_neg: float

    @classmethod
    def from_yaml(
        cls, pbs_crosstalk_parameters: Dict[str, Any], load_row: int
    ) -> "PBSCrosstalkParameters16":
        """ """
        if f"poly_coeffs_positive_row{load_row}" not in pbs_crosstalk_parameters:
            raise ValueError(
                f"Possibly not a 16-variable config. Missing poly_coeffs_positive_row{load_row}"
            )
        return cls(
            pbs_poly_coeffs_neg=torch.tensor(
                pbs_crosstalk_parameters[f"poly_coeffs_negative_row{load_row}"],
                dtype=torch.float32,
            ),
            pbs_poly_coeffs_pos=torch.tensor(
                pbs_crosstalk_parameters[f"poly_coeffs_positive_row{load_row}"],
                dtype=torch.float32,
            ),
            pbs_power_norm_factor_pos=0.04,
            pbs_power_norm_factor_neg=0.06,
        )


class PBSCrosstalkParameters48(NamedTuple):
    """
    PBS crosstalk parameters for 48 variables.
    """

    negative_to_positive: Tensor
    positive_to_negative: Tensor

    @classmethod
    def from_yaml(
        cls, pbs_crosstalk_parameters: Dict[str, Any]
    ) -> "PBSCrosstalkParameters48":
        """ """
        assert all(
            [
                field in pbs_crosstalk_parameters
                for field in ["positive_to_negative", "negative_to_positive"]
            ]
        ), "Possibly not a 48-variable config."
        return cls(
            negative_to_positive=torch.tensor(
                pbs_crosstalk_parameters["negative_to_positive"], dtype=torch.float32
            ),
            positive_to_negative=torch.tensor(
                pbs_crosstalk_parameters["positive_to_negative"], dtype=torch.float32
            ),
        )


class ClosedLoopCorrection(NamedTuple):
    """
    Only for 48-variable systems.
    """

    dc_offset: torch.Tensor
    linear_offset: torch.Tensor
    scaling: torch.Tensor
    solution_to_adc_ratio: float

    @classmethod
    def from_dict(
        cls, close_loop_offset_and_scale: Dict[str, Any]
    ) -> "ClosedLoopCorrection":
        return cls(
            dc_offset=torch.tensor(
                close_loop_offset_and_scale["dc_offset"], dtype=torch.float32
            ),
            linear_offset=torch.tensor(
                close_loop_offset_and_scale["linear_offset"], dtype=torch.float32
            ),
            scaling=torch.tensor(
                close_loop_offset_and_scale["scaling"], dtype=torch.float32
            ),
            solution_to_adc_ratio=float(
                close_loop_offset_and_scale["solution_to_adc_ratio"]
            ),
        )


class HardwareParameters(NamedTuple):
    """
    Hardware parameters.
    """

    ## Step 1 - tanh
    tanh_coefficients: List[float]
    ## Step 2 - LED power before SLM and DC removal
    # polynomial coefficients for micro-LED nonlinearity
    micro_led_coefficients_pos: torch.Tensor
    micro_led_coefficients_neg: torch.Tensor

    ## Step 3 - Input efficiency
    input_efficiency: Tensor
    ## Step 4 - weight distortion polynomials for positive and negative weights
    # shape [n_vars, 3]
    weight_distortion_pos: Tensor
    weight_distortion_neg: Tensor
    ## Step 7 - Output efficiency per path
    # eta_pos, eta_neg
    output_efficiency_pos: Union[float, Tensor]
    output_efficiency_neg: Union[float, Tensor]

    ## Step 8 - PD crosstalk
    # shape [n_vars, 2]: Column 0 is the crosstalk to the left, column 1 to the right.
    pd_cross_talk_pos: Tensor
    pd_cross_talk_neg: Tensor
    darkness_pos: Tensor
    darkness_neg: Tensor
    pd_offset_pos: Tensor
    pd_offset_neg: Tensor
    # PBS Crosstalk and Saturation
    pbs_crosstalk: Union[PBSCrosstalkParameters16, PBSCrosstalkParameters48]
    saturation: Union[Saturation16, Saturation48]
    # DEQ Input Clips
    deq_input_min_max: List[float]
    # Closed Loop Correction
    closed_loop_correction: Optional[ClosedLoopCorrection] = None

    @property
    def is_16vars(self) -> bool:
        """
        Check if the hardware parameters are for 16 variables.
        """
        raise NotImplementedError(
            "is_16vars property should be implemented in subclasses."
        )


class HardwareParameters48(HardwareParameters):
    """
    Hardware parameters for 48 variables.
    """

    n_variables = 48
    pbs_crosstalk: PBSCrosstalkParameters48
    saturation: Saturation48

    @property
    def is_16vars(self) -> bool:
        return False


class HardwareParameters16(HardwareParameters):
    """
    Hardware parameters for 16 variables.
    """

    n_variables = 16
    pbs_crosstalk: PBSCrosstalkParameters16
    saturation: Saturation16

    @property
    def is_16vars(self) -> bool:
        return True


def from_config_file(
    file_path: str = CURRENT_CONFIG,
    load_row: int = 12,
    is_16var: bool = False,
) -> HardwareParameters:
    """
    Load hardware parameters from config files.
    """
    # open yaml
    with open(file_path, "r", encoding="UTF-8") as file:
        all_coefficients = yaml.safe_load(file)
    # Step 1: Load tanh coefficients
    tanh_coefficients = all_coefficients["tanh_sigmoid_coeffs"]
    # Step 2: Load micro-LED coefficients
    micro_led_coefficients_pos = all_coefficients["led_poly_coeffs_pos"]
    micro_led_coefficients_neg = all_coefficients["led_poly_coeffs_neg"]

    # Step 3: Load input efficiencies
    row_profiles = all_coefficients["row_profiles"]
    pos_row = row_profiles[f"positive_row{load_row}"]
    neg_row = row_profiles[f"negative_row{load_row}"]
    input_efficiency = torch.tensor([pos_row, neg_row], dtype=torch.float32)  # [16, 2]

    # Step 4 - Load weight distortion polynomials
    weights_distortion_coeffs = all_coefficients["column_wise_weights_distortion"]
    weight_distortion_pos = torch.tensor(
        weights_distortion_coeffs[f"positive_row{load_row}"], dtype=torch.float32
    )
    weight_distortion_neg = torch.tensor(
        weights_distortion_coeffs[f"negative_row{load_row}"], dtype=torch.float32
    )
    output_efficiency_pos = 0.65 if is_16var else 1.0
    output_efficiency_neg = 1.0 if is_16var else 1.0
    # make sure that in both cases efficiencies are tensors
    if is_16var:
        output_efficiency_pos = torch.tensor(output_efficiency_pos, dtype=torch.float32)
        output_efficiency_neg = torch.tensor(output_efficiency_neg, dtype=torch.float32)
        closed_loop_correction = None
    else:
        output_efficiency_pos *= torch.tensor(
            all_coefficients["output_correction"]["positive"], dtype=torch.float32
        )
        output_efficiency_neg *= torch.tensor(
            all_coefficients["output_correction"]["negative"], dtype=torch.float32
        )
        closed_loop_correction = ClosedLoopCorrection.from_dict(
            all_coefficients["close_loop_offset_and_scale"]
        )
    # Step 8 - Load PD crosstalk
    pd_crosstalk = all_coefficients["pd_crosstalk"]
    ratio_to_later_positive = torch.tensor(
        pd_crosstalk[f"ratio_to_later_positive_row{load_row}"], dtype=torch.float32
    )
    ratio_to_earlier_positive = torch.tensor(
        pd_crosstalk[f"ratio_to_earlier_positive_row{load_row}"], dtype=torch.float32
    )
    # combine to get pd_cross_talk_pos [n_vars, 2]
    pd_cross_talk_pos = torch.stack(
        [ratio_to_later_positive, ratio_to_earlier_positive], dim=1
    )
    darkness_pos = weight_distortion_pos[:, -1]
    darkness_neg = weight_distortion_neg[:, -1]

    pd_offset = all_coefficients["pd_offset"]
    pd_offset_pos = torch.tensor(
        pd_offset[f"pd_offset_positive_row{load_row}"], dtype=torch.float32
    )
    pd_offset_neg = torch.tensor(
        pd_offset[f"pd_offset_negative_row{load_row}"], dtype=torch.float32
    )

    # again for negative
    ratio_to_later_negative = torch.tensor(
        pd_crosstalk[f"ratio_to_later_negative_row{load_row}"], dtype=torch.float32
    )
    ratio_to_earlier_negative = torch.tensor(
        pd_crosstalk[f"ratio_to_earlier_negative_row{load_row}"], dtype=torch.float32
    )
    # combine to get pd_cross_talk_neg
    pd_cross_talk_neg = torch.stack(
        [ratio_to_later_negative, ratio_to_earlier_negative], dim=1
    )

    # Step 9 - Load PBS crosstalk
    pbs_crosstalk: Union[PBSCrosstalkParameters16, PBSCrosstalkParameters48]
    if is_16var:
        pbs_crosstalk = PBSCrosstalkParameters16.from_yaml(
            all_coefficients["pbs_crosstalk"], load_row
        )
    else:
        pbs_crosstalk = PBSCrosstalkParameters48.from_yaml(
            all_coefficients["pbs_crosstalk"]
        )
    # Step 10 - Load saturation
    saturation: Union[Saturation16, Saturation48]
    if is_16var:
        saturation = Saturation16.from_yaml(all_coefficients["saturation"])
    else:
        saturation = Saturation48.from_yaml(all_coefficients["saturation"])
    deq_input_min_max = all_coefficients["deq_input_min_max"]
    cls = HardwareParameters16 if is_16var else HardwareParameters48
    return cls(
        deq_input_min_max=deq_input_min_max,
        # Step 1 - Tanh coefficients
        tanh_coefficients=tanh_coefficients,
        # step 2 - LED Coefficients
        micro_led_coefficients_pos=micro_led_coefficients_pos,
        micro_led_coefficients_neg=micro_led_coefficients_neg,
        # Step 3 - Input efficiency
        input_efficiency=input_efficiency,
        # Step 4 - Weight Distortion
        weight_distortion_pos=weight_distortion_pos,
        weight_distortion_neg=weight_distortion_neg,
        # Step 7 - Output efficiency
        output_efficiency_pos=output_efficiency_pos,
        output_efficiency_neg=output_efficiency_neg,
        # Step 8 - PD Crosstalk
        pd_cross_talk_pos=pd_cross_talk_pos,
        pd_cross_talk_neg=pd_cross_talk_neg,
        darkness_pos=darkness_pos,
        darkness_neg=darkness_neg,
        pd_offset_pos=pd_offset_pos,
        pd_offset_neg=pd_offset_neg,
        # Step 9 - PBS Crosstalk
        pbs_crosstalk=pbs_crosstalk,
        # Step 10 - Saturation
        saturation=saturation,
        # Closed Loop Correction
        closed_loop_correction=closed_loop_correction,
    )
