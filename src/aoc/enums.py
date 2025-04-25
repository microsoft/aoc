"""
Enums for multi-choice options.
"""

from enum import Enum, auto


class MatrixConnectivityType(Enum):
    """
    What sort of AOC matrix are we looking at?
    Hopfield means the matrix can be block-structured but needs to be overall
    symmetric. Feedforward means the matrix is lower-diagonal block-structured and not symmetric.
    Feedback means, that the lower-diagonal block structure is moved one up and the last layer
    becomes a feedback block (standard DEQ/Diffusion models).
    """

    HOPFIELD = auto()
    FEEDFORWARD = auto()
    FEEDBACK = auto()

    def __str__(self) -> str:
        return self.name
