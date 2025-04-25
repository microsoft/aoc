"""
Miscellaneous utility functions.
"""

from torch import nn


def get_activation_function(actfun: str) -> nn.Module:
    """
    Convert string to activation function.
    """
    lower_case = actfun.lower()
    if lower_case == "relu":
        return nn.ReLU()
    elif lower_case == "gelu":
        return nn.GELU()
    elif lower_case == "tanh":
        return nn.Tanh()
    elif lower_case == "sigmoid":
        return nn.Sigmoid()
    elif lower_case in ["leaky_relu", "leakyrelu"]:
        return nn.LeakyReLU()
    elif lower_case == "softplus":
        return nn.Softplus()
    elif lower_case in ["swish", "silu"]:
        return nn.SiLU()
    elif lower_case == "selu":
        return nn.SELU()
    elif lower_case == "none":
        return nn.Identity()
    else:
        raise NotImplementedError(f"Activation function {actfun} not implemented")
