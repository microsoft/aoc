"""
Structured Model Classes for AOC
===================================
Contains classes for building AOC models from sequential torch models.
The current AOC hardware can carry out a single vector-matrix multiplication per iteration.
Multi-layer models can be implemented by using a sub-diagonal blockmatrix structure.
This module allows defining sequential models and building the corresponding AOC matrix and bias.

Warning: AOC operates using a x^T W convention. Therefore, matrices need to be transposed once generated.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import (
    Linear,
    Module,
    Parameter,
    ParameterList,
    Sequential,
    Tanh,
)

from .block_matrix import (
    blockmatrix,
    build_lower_triangular_square_matrix,
)
from .enums import MatrixConnectivityType


def to_aoc_matrix_convention(matrix: Tensor) -> Tensor:
    """
    Transposes a matrix to the AOC convention.
    """
    return matrix.T


def from_aoc_matrix_convention(matrix: Tensor) -> Tensor:
    """
    Transposes a matrix to the torch convention.
    """
    return matrix.T


class StructuredAOCMatrixModel(Module, ABC):
    """
    Base class for building an AOC model (weight, bias, activations) from a sequential (multi-layer) torch model.
    """

    def __init__(
        self,
        weights: List[Tensor],
        biases: List[Tensor],
        apply_tanh_to_layer: List[bool],
    ):
        super().__init__()
        self._matrix_size = None
        self._init_weights_and_biases(weights, biases, apply_tanh_to_layer)

        self.input_dim = self.layer_shapes[0][1]
        self.output_dim = self.layer_shapes[-1][0]
        self.apply_tanh_to_indices = self.get_tanh_indices()

    @property
    def num_layers(self) -> int:
        """
        Return number of layers in the sequential model.
        """
        return len(self.layer_shapes)

    @property
    def layer_shapes(self) -> List[Tuple[int, int]]:
        """
        Return shapes of the layers in the sequential model.
        Layer shapes might change in `_init_weights_and_biases`, re-calculate it for safety.

        """
        return [w.shape for w in self.weights]

    @property
    def matrix_size(self) -> Tuple[int, int]:
        """
        Return size of AOC matrix.
        """
        if self._matrix_size is None:
            self._matrix_size = self.build_aoc_matrix().shape
        assert self._matrix_size is not None, "Matrix size not initialized."
        return self._matrix_size

    @classmethod
    def from_torch_model(
        cls,
        torch_model: torch.nn.Module,
        apply_tanh_to_layer: Optional[List[bool]] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ) -> "StructuredAOCMatrixModel":
        """
        Builds a StructuredAOCMatrixModel from any model inheriting from torch.nn.Module.
        If argument `apply_tanh_to_layer` is not provided, the non-linearities are taken from the torch model.
        """
        weights, biases, apply_tanh_torch_model = get_structure_from_torch_model(
            torch_model
        )

        if apply_tanh_to_layer is None:
            apply_tanh_to_layer = apply_tanh_torch_model
        else:
            assert not any(apply_tanh_to_layer), (
                "Got non-linearities from torch_model but also set them manually."
            )

        matrix_structure = cls(weights, biases, apply_tanh_to_layer)
        if input_dim is not None:
            matrix_structure.input_dim = input_dim
        if output_dim is not None:
            matrix_structure.output_dim = output_dim
        return matrix_structure

    @classmethod
    def from_aoc_components(
        cls,
        aoc_matrix: torch.Tensor,
        aoc_bias: torch.Tensor,
        apply_tanh_to_layer: List[bool],
        layer_shapes: List[Tuple[int, int]],
    ) -> "StructuredAOCMatrixModel":
        """
        Builds a StructuredAOCMatrixModel from AOC components.
        The matrix has to be in AOC matrix convention.
        """
        aoc_matrix = from_aoc_matrix_convention(aoc_matrix)
        weights, biases = cls._get_structure_from_aoc_components(
            aoc_matrix, aoc_bias, layer_shapes
        )
        return cls(weights, biases, apply_tanh_to_layer)

    @abstractmethod
    def build_aoc_matrix(self) -> Tensor:
        """
        Build the AOC matrix from the weights of the sequential model.
        Returns the matrix in AOC matrix convention.
        """

    @abstractmethod
    def build_aoc_bias(self) -> Tensor:
        """
        Builds the AOC bias vector from the biases of the sequential model.
        """

    @abstractmethod
    def get_tanh_indices(self) -> torch.BoolTensor:
        """
        Determines to which indices of the aoc vector non-linearity should be applied.
        """

    @abstractmethod
    def get_output_from_aoc_vector(self, x: Tensor) -> Tensor:
        """
        Extracts the components of the AOC vector containing the output of the multi-layer model.
        """

    @staticmethod
    @abstractmethod
    def _get_structure_from_aoc_components(
        aoc_matrix: torch.Tensor,
        aoc_bias: torch.Tensor,
        layer_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Extracts weights and biases from AOC matrix and bias vector.
        """

    def _init_weights_and_biases(
        self,
        weights: List[Tensor],
        biases: List[Tensor],
        apply_tanh_to_layer: List[bool],
    ) -> None:
        """
        Initializes weights and biases from a torch model.
        """
        self.weights = ParameterList(
            [Parameter(w, requires_grad=True) for w in weights]
        )

        self.biases = ParameterList(
            [
                Parameter(b, requires_grad=True if b is not None else False)
                for b in biases
            ]
        )

        self.apply_tanh_to_layer = apply_tanh_to_layer

    def _fill_bias_with_zeros(self, biases: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Fills biases that have not been set in `torch_model` with zeros.
        """
        for i, b in enumerate(biases):
            if len(b) == 0:
                biases[i] = torch.zeros(
                    self.layer_shapes[i][0],
                    requires_grad=False,
                    device=biases[0].device,
                )
        return biases

    def build_aoc_vector_from_input(self, x: Tensor) -> Tensor:
        """
        Pads the input vector to the dimension of the AOC matrix keeping the d-dimensional input in the first d components.
        """
        assert len(x.shape) == 2, "Input must be 2D (bs, dim)."
        if x.shape[1] < self.matrix_size[1]:
            x = torch.cat(
                [
                    x,
                    torch.zeros(x.shape[0], self.matrix_size[1] - x.shape[1]).to(
                        x.device
                    ),
                ],
                dim=1,
            )
        return x

    def aoc_iter(self, x: Tensor) -> Tensor:
        """
        A single AOC iteration. Used only for debugging.
        """
        x = torch.matmul(x, self.build_aoc_matrix())
        x = x + self.build_aoc_bias()
        if self.apply_tanh_to_indices is not None:
            x = torch.where(self.apply_tanh_to_indices, torch.tanh(x), x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        A full forward pass through the sequential model. Used only for debugging.
        """
        x = self.build_aoc_vector_from_input(x)
        for _ in range(self.num_layers):
            x = self.aoc_iter(x)
        return self.get_output_from_aoc_vector(x)

    def insert_at_layer(
        self,
        z: torch.Tensor,
        new_values: torch.Tensor,
        neuron_layer_index: int,
    ) -> torch.Tensor:
        """
        Write new values to a specific layer of the neural state vector z.
        Note that this mutates the input z!
        Args:
            model: The model
            z: The states (dim: batch_size x total_number_neurons in the hidden layers)
            neuron_layer_index: The index of the neuron layer to write to. This is different from
            the typical layer index which normally counts the weight matrices.
            new_values: The new values to write to the layer (dim: batch_size x number_neurons_in_layer)
        returns:
            z: The updated states
        """
        # Check that the block number is within the number of hidden layers

        assert neuron_layer_index >= 0
        num_neural_layers = (
            self.num_layers + 1
        )  # count the number of neurons, not weight matrices
        max_index = num_neural_layers
        assert neuron_layer_index < max_index
        # stich together the shapes of the layers [[7, 5], [13, 7]] -> [5, 7, 13]
        unrolled_layer_shapes = [shape[1] for shape in self.layer_shapes] + [
            self.layer_shapes[-1][0]
        ]

        def _location_in_hidden_state(start_idx: int, end_idx: int) -> int:
            return sum(unrolled_layer_shapes[start_idx:end_idx])

        # Extract the start and end indices for the layer
        if neuron_layer_index < num_neural_layers:
            start_index = _location_in_hidden_state(0, neuron_layer_index)
            end_index = _location_in_hidden_state(0, neuron_layer_index + 1)
        else:  # This branch is only entered if dyadic_neurons is True
            start_index = self.matrix_size[0] // 2 + _location_in_hidden_state(
                0, neuron_layer_index - num_neural_layers
            )
            end_index = self.matrix_size[0] // 2 + _location_in_hidden_state(
                0, neuron_layer_index - num_neural_layers + 1
            )

        # Check that the new values have the correct shape
        assert new_values.size(1) == (end_index - start_index), (
            f"New values shape {new_values.size(1)} does not match layer shape {end_index - start_index}."
        )

        # Write the new values to the layer
        z[:, start_index:end_index] = new_values
        return z


class FeedbackAOCModel(StructuredAOCMatrixModel):
    """
    Builds block matrix and corresponding bias/non-linearity vector for sequential model with feedback of form
    A = [[0, 0, M_2],
        [M_0, 0, 0],
        [0, M_1, 0],]
    If a single layer model is provided, builds a (dense) single layer model. In this case this is equivalent to a feedforward model.
    """

    def build_aoc_matrix(self) -> Tensor:
        """
        Builds the AOC matrix for the feedback model.
        Returns in AOC matrix convention.
        """
        assert self.input_dim == self.output_dim, (
            "Input and output dimensions must agree for feedback model."
        )

        if self.num_layers == 1:
            return to_aoc_matrix_convention(self.weights[0])

        weights = list(self.weights)
        num_blocks = len(weights)

        assert weights[0].shape[1] == weights[-1].shape[0], (
            "First layer's input dimension must agree with last layer's output dimensions."
        )

        # Place final layer
        blocks = [[0] * (num_blocks - 1) + [weights[-1]]]

        # Place lower triangular block matrix
        for i in range(num_blocks - 1):
            blocks.append([0] * i + [weights[i]] + [0] * (num_blocks - 1 - i))

        aoc_matrix = blockmatrix(blocks)
        return to_aoc_matrix_convention(aoc_matrix)

    def build_aoc_bias(self) -> Tensor:
        """
        Builds the AOC bias vector for the feedback model.
        """
        biases = list(self.biases)
        biases = self._fill_bias_with_zeros(biases)

        if self.num_layers == 1:
            return biases[0]

        # Last layer is located in the first blockrow so need to move corresponding bias to first position
        biases_shifted = biases[-1:] + biases[:-1]
        aoc_bias = torch.concat(biases_shifted, dim=0).reshape(1, -1)
        return aoc_bias

    def get_tanh_indices(self) -> torch.BoolTensor:
        """
        Extracts indices of AOC vector to apply tanh to.
        """
        if self.num_layers == 1:
            return torch.BoolTensor(
                self.apply_tanh_to_layer * self.matrix_size[0]
            ).reshape(1, -1)

        # Non-linearity is applied before matmul -> n-th non-linearity needs to act on (n+1)-st blockentries
        apply_tanh_to_layer_shifted = (
            self.apply_tanh_to_layer[-1:] + self.apply_tanh_to_layer[:-1]
        )

        # Get the lengths of the respective outputs to apply the tanh to
        output_lengths = [shape[0] for shape in self.layer_shapes]
        output_lengths_shifted = output_lengths[-1:] + output_lengths[:-1]
        apply_tanh_to_index = []
        for l, apply_tanh in zip(output_lengths_shifted, apply_tanh_to_layer_shifted):
            apply_tanh_to_index.extend(l * [apply_tanh])

        apply_tanh_to_index = torch.BoolTensor(apply_tanh_to_index).reshape(1, -1)
        return apply_tanh_to_index

    def get_output_from_aoc_vector(self, x: Tensor) -> Tensor:
        """
        Select components of AOC vector corresponding to the output of the multi-layer model.
        """
        return x[:, : self.input_dim]

    @staticmethod
    def _get_structure_from_aoc_components(
        aoc_matrix: Tensor,
        aoc_bias: Tensor,
        layer_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Extracts weights and biases from AOC components.
        """
        # Remove batch dimensiona from bias
        aoc_bias = aoc_bias.squeeze()

        if len(layer_shapes) == 1:
            return [aoc_matrix], [aoc_bias]

        weights = []
        biases = []

        input_shape = layer_shapes[0][1]
        output_shape = layer_shapes[-1][0]

        row = input_shape
        column = 0
        for shape in layer_shapes[:-1]:
            weights.append(aoc_matrix[row : row + shape[0], column : column + shape[1]])
            biases.append(aoc_bias[row : row + shape[0]])
            row = row + shape[0]
            column = column + shape[1]

        # Final layer is places in first blockrow
        weights.append(aoc_matrix[:output_shape, -output_shape:])
        biases.append(aoc_bias[:output_shape])
        return weights, biases


class FeedforwardAOCModel(StructuredAOCMatrixModel):
    """
    Builds block matrix and corresponding bias/non-linearity vector for sequential model with feedback of form
    A = [[0, 0, 0],
        [M_0, 0, 0],
        [0, M_1, 0]]
    Note when using this we need to inject the input (x,0,...,0) at each iteration, otherwise the result gets lost.
    If a non-linearity is applied to the last layer, need to add an additional row to obtain a matrix of the form
    A = [[0, 0, 0, 0],
        [M_0, 0, 0, 0],
        [0, M_1, 0, 0],
        [0, 0, I, 0]]
    If a single layer model is provided, builds a (dense) single layer model. In this case this is equivalent to a feedback model.
    """

    def _init_weights_and_biases(
        self,
        weights: List[Tensor],
        biases: List[Tensor],
        apply_tanh_to_layer: List[bool],
    ) -> None:
        """
        Before initializing weights and biases, add an identity row to the weights and corresponding zeros to the biases if the last layer has a non-linearity.
        """
        output_dim = weights[-1].shape[0]

        if len(weights) > 1 and apply_tanh_to_layer[-1]:
            weights.append(torch.eye(output_dim))
            biases.append(torch.zeros(output_dim))
            apply_tanh_to_layer.append(False)

        super()._init_weights_and_biases(weights, biases, apply_tanh_to_layer)

    def build_aoc_matrix(self) -> Tensor:
        """
        Builds the AOC matrix for the feedforward model.
        Returns the matrix in AOC matrix convention.
        """
        if self.num_layers == 1:
            return to_aoc_matrix_convention(self.weights[0])

        weights = list(self.weights)
        num_blocks = len(weights)

        # Construct off-diagonal lower triangular block matrix
        blocks = [[0] * num_blocks + [torch.zeros(self.input_dim, self.output_dim)]]
        for i in range(num_blocks):
            blocks.append([0] * i + [weights[i]] + [0] * (num_blocks - i))

        aoc_matrix = blockmatrix(blocks)
        return to_aoc_matrix_convention(aoc_matrix)

    def build_aoc_bias(self) -> Tensor:
        """
        Builds the AOC bias vector for the feedforward model.
        """
        device = self.biases[0].device
        biases = list(self.biases)
        biases = self._fill_bias_with_zeros(biases)

        if self.num_layers == 1:
            return biases[0]

        if self.apply_tanh_to_layer[-1]:
            biases.append(torch.zeros(self.output_dim, device=device))

        biases_shifted = [torch.zeros(self.input_dim, device=device)] + biases
        aoc_bias = torch.cat(biases_shifted, dim=0).reshape(1, -1)
        return aoc_bias

    def get_tanh_indices(self) -> torch.BoolTensor:
        """
        Extracts indices of AOC vector to apply tanh to.
        """

        if self.num_layers == 1:
            return torch.BoolTensor(
                self.apply_tanh_to_layer * self.matrix_size[0]
            ).reshape(1, -1)

        output_lengths_shifted = [self.input_dim] + [
            shape[0] for shape in self.layer_shapes
        ]

        # Non-linearity is applied before matmul -> n-th non-linearity needs to act on (n + 1)-st blockentries
        apply_tanh_to_layer_shifted = [False] + self.apply_tanh_to_layer
        apply_tanh_to_index = []
        for l, apply_tanh in zip(output_lengths_shifted, apply_tanh_to_layer_shifted):
            apply_tanh_to_index.extend(l * [apply_tanh])
        apply_tanh_to_index = torch.BoolTensor(apply_tanh_to_index).reshape(1, -1)

        return apply_tanh_to_index

    def get_output_from_aoc_vector(self, x: Tensor) -> Tensor:
        """
        Select components of AOC vector corresponding to the output of the multi-layer model.
        """
        assert len(x.shape) == 2, "Input must be 2D (bs, dim)."
        return x[:, -self.output_dim :]

    @staticmethod
    def _get_structure_from_aoc_components(
        aoc_matrix: Tensor,
        aoc_bias: Tensor,
        layer_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Extracts weights and biases from AOC components.
        """
        # Remove batch dimensiona from bias
        aoc_bias = aoc_bias.squeeze()

        if len(layer_shapes) == 1:
            return [aoc_matrix], [aoc_bias]

        weights = []
        biases = []

        input_shape = layer_shapes[0][1]

        row = input_shape
        column = 0
        for shape in layer_shapes:
            weights.append(aoc_matrix[row : row + shape[0], column : column + shape[1]])
            biases.append(aoc_bias[row : row + shape[0]])
            row = row + shape[0]
            column = column + shape[1]

        return weights, biases


class HopfieldAOCModel(StructuredAOCMatrixModel):
    """
    Builds blockmatrix of multi-layer Hopfield model of the form
    A = [[0, M_0^T, 0],
        [M_0, 0, M_1^T],
        [0, M_1, 0],]
    For Hopfield models we set a single bias for the whole AOC vector.
    If a single layer model is provided, builds a (dense) single layer model.
    """

    def __init__(
        self,
        weights: List[Tensor],
        biases: List[Tensor],
        apply_tanh_to_layer: List[bool],
        use_input_bias: bool = False,
        treat_single_layer_as_multi_layer: bool = False,
    ):
        self.use_input_bias = use_input_bias
        self.treat_single_layer_as_multi_layer = treat_single_layer_as_multi_layer
        super().__init__(weights, biases, apply_tanh_to_layer)

    @classmethod
    def from_torch_model(
        cls,
        torch_model: Module,
        apply_tanh_to_layer: Optional[List[bool]] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        treat_single_layer_as_multi_layer: bool = False,
    ) -> StructuredAOCMatrixModel:
        """
        Builds a HopfieldAOCModel from any model inheriting from torch.nn.Module.
        If argument `apply_tanh_to_layer` is not provided, the non-linearities are taken from the torch model.

        Args:
            torch_model: torch.nn.Module : Any model inheriting from torch.nn.Module.
            apply_tanh_to_layer: Optional[List[bool]] : Determines if non-linearities are applied to the layers. If not provided, non-linearities are taken from the torch model.
            treat_single_layer_as_multi_layer: bool: If True, treat a single layer model as a multi-layer model otherwise create a dense AOC matrix if only 1 layer is provided

        Returns:
            StructuredAOCMatrixModel : Instance of HopfieldAOCModel
        """
        weights, biases, apply_tanh_torch_model = get_structure_from_torch_model(
            torch_model
        )

        if apply_tanh_to_layer is None:
            apply_tanh_to_layer = apply_tanh_torch_model
        else:
            assert not any(apply_tanh_to_layer), (
                "Got non-linearities from torch_model but also set them manually."
            )

        return cls(
            weights,
            biases,
            apply_tanh_to_layer,
            treat_single_layer_as_multi_layer=treat_single_layer_as_multi_layer,
        )

    @classmethod
    def from_aoc_components(
        cls,
        aoc_matrix: torch.Tensor,
        aoc_bias: torch.Tensor,
        apply_tanh_to_layer: List[bool],
        layer_shapes: List[Tuple[int, int]],
        treat_single_layer_as_multi_layer: bool = False,
    ) -> "HopfieldAOCModel":
        """
        Builds a HopfieldAOCModel from AOC components.
        Overrides base class method to use the correct aoc_bias as bias. This is needed as usually
        for Hopfield we set a single bias for the whole AOC vector in the __init__ methods.

        Args:
            aoc_matrix: torch.Tensor : AOC matrix in AOC convention.
            aoc_bias: torch.Tensor : AOC bias
            apply_tanh_to_layer: List[bool] : Determines if non-linearities are applied to the layers.
            layer_shapes: List[Tuple[int, int]] : Shapes of the layers in the sequential model.
            treat_single_layer_as_multi_layer: bool : If True, treat a single layer model as a multi-layer model otherwise create a dense AOC matrix if only 1 layer is provided

        Returns:
            HopfieldAOCModel : Instance of HopfieldAOCModel
        """
        aoc_matrix = from_aoc_matrix_convention(aoc_matrix)
        weights, biases = cls._get_structure_from_aoc_components(
            aoc_matrix,
            aoc_bias,
            layer_shapes,
            treat_single_layer_as_multi_layer=treat_single_layer_as_multi_layer,
        )
        return cls(
            weights,
            biases,
            apply_tanh_to_layer,
            use_input_bias=True,
            treat_single_layer_as_multi_layer=treat_single_layer_as_multi_layer,
        )

    def _init_weights_and_biases(self, weights, biases, apply_tanh_to_layer):
        """
        Initializes weights and biases from a torch model.
        """
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(w, requires_grad=True) for w in weights]
        )

        if self.use_input_bias:
            biases = [
                torch.nn.Parameter(torch.cat(biases).reshape(1, -1), requires_grad=True)
            ]
        else:
            assert all(b is None for b in biases) or not any(
                b is None for b in biases
            ), "Models require a bias for each layer or none at all."
            # Check if bias should be trainable or just 0
            use_bias = biases[0] is not None

            biases = [
                torch.nn.Parameter(
                    torch.zeros(1, self.matrix_size[0]), requires_grad=use_bias
                )
            ]
        self.biases = nn.ParameterList(biases)
        self.apply_tanh_to_layer = apply_tanh_to_layer

    def build_aoc_matrix(self) -> Tensor:
        """
        Builds the AOC matrix for the Hopfield model.
        Returns in AOC matrix convention (although it is symmetric).
        """
        if self.num_layers == 1 and not self.treat_single_layer_as_multi_layer:
            return 1 / 2 * (self.weights[0] + self.weights[0].T)

        lower_block = build_lower_triangular_square_matrix(self.weights)
        aoc_matrix = lower_block + lower_block.T
        return to_aoc_matrix_convention(aoc_matrix)

    def build_aoc_bias(self) -> Tensor:
        """
        Builds the AOC bias vector for the Hopfield model.
        """
        return self.biases[0]

    def get_tanh_indices(self) -> torch.BoolTensor:
        """
        Apply non-linearity to all or none of the channels.
        """
        if self.num_layers == 1:
            return torch.BoolTensor(
                self.apply_tanh_to_layer * self.matrix_size[0]
            ).reshape(1, -1)

        assert all(self.apply_tanh_to_layer) or not any(self.apply_tanh_to_layer), (
            "All layers must have non-linearities or none at all for Hopfield."
        )

        return torch.BoolTensor(
            [self.apply_tanh_to_layer[0]] * self.matrix_size[0]
        ).reshape(1, -1)

    def get_output_from_aoc_vector(self, x: Tensor) -> Tensor:
        """
        Select components of AOC vector corresponding to the output of the multi-layer model.
        """
        return x[:, -self.output_dim :]

    @staticmethod
    def _get_structure_from_aoc_components(
        aoc_matrix: Tensor,
        aoc_bias: Tensor,
        layer_shapes: List[Tuple[int, int]],
        treat_single_layer_as_multi_layer: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Extracts weights and biases from AOC components.
        """
        # Remove batch dimensiona from bias
        aoc_bias = aoc_bias.squeeze()

        if len(layer_shapes) == 1 and not treat_single_layer_as_multi_layer:
            return [aoc_matrix], [aoc_bias]

        weights = []
        input_shape = layer_shapes[0][1]

        row = input_shape
        column = 0
        for shape in layer_shapes:
            weights.append(aoc_matrix[row : row + shape[0], column : column + shape[1]])
            row = row + shape[0]
            column = column + shape[1]

        biases = [aoc_bias]
        return weights, biases


def get_structure_from_torch_model(
    model: Union[Module, Sequential],
) -> Tuple[List[Tensor], List[Union[Tensor, None]], List[bool]]:
    """
    Extracts weights, biases and non-linearities from a torch model.

    Args:
        model: torch.nn.Module or torch.nn.Sequential

    Returns:
        weights: List of torch tensors
        biases: List of torch tensors or Nones
        apply_tanh_to_layer: List of bools
    """

    # Allow same syntax for single layer models
    if len(list(model.children())) == 0:
        model = Sequential(model)

    weights = []
    biases = []
    apply_tanh_to_layer = []

    layer_shapes = []
    layers = list(model.children())
    for i, layer in enumerate(layers):
        if isinstance(layer, Linear):
            layer_shapes.append(layer.weight.shape)
            weights.append(layer.weight.data)
            biases.append(layer.bias.data if layer.bias is not None else None)
            if i < len(layers) - 1 and isinstance(layers[i + 1], Tanh):
                apply_tanh_to_layer.append(True)
            else:
                apply_tanh_to_layer.append(False)
        elif isinstance(layer, Tanh):
            continue
        else:
            raise ValueError(
                f"Only torch.nn.Linear layers are supported, got {layer} instead."
            )
    return weights, biases, apply_tanh_to_layer


def get_structured_model(
    model: nn.Module,
    connectivity: MatrixConnectivityType,
) -> StructuredAOCMatrixModel:
    """
    Builds a structured model from a torch model.

    Args:
        model (nn.Module): The torch model that is to be translated to AOC.
        connectivity (MatrixConnectivityType): The type of connectivity.

    Returns:
        StructuredAOCMatrixModel : The structured model
    """
    if connectivity == MatrixConnectivityType.FEEDBACK:
        return FeedbackAOCModel.from_torch_model(model)
    elif connectivity == MatrixConnectivityType.FEEDFORWARD:
        return FeedforwardAOCModel.from_torch_model(model)
    elif connectivity == MatrixConnectivityType.HOPFIELD:
        is_single_layer = isinstance(model, nn.Linear)
        return HopfieldAOCModel.from_torch_model(
            model, treat_single_layer_as_multi_layer=is_single_layer
        )
    raise ValueError(f"Unsupported model type {connectivity}.")
