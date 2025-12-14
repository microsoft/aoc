"""
Dataset builders for MNIST and regression tasks.
Ported from AnalogDEQ for reproducibility.
"""
from typing import Tuple, Dict, Any, Callable, List, Optional, Union
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
import numpy as np
from einops import repeat


# ============================================================================
# MNIST Datasets
# ============================================================================

def build_mnist_dataset(
    train_valid_split_ratio: float = 0.8,
    rescale_to_one_minus_one: bool = False,
    small: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Build MNIST dataset for classification.

    Args:
        train_valid_split_ratio: ratio of training set size to total training data
        rescale_to_one_minus_one: Rescale pixel values to [-1, 1] instead of [0, 1]
        small: If True, use only 64 samples (for debugging)

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """

    def _to_m_one_one(x: torch.Tensor) -> torch.Tensor:
        if not rescale_to_one_minus_one:
            return x
        return (x.float() - x.min()) / (x.max() - x.min()) * 2 - 1

    transforms_both = transforms.Compose([transforms.ToTensor(), _to_m_one_one])

    dataset_train = datasets.MNIST(
        "data", train=True, download=True, transform=transforms_both
    )
    dataset_test = datasets.MNIST(
        "data", train=False, download=True, transform=transforms_both
    )
    
    if small:
        dataset_train.data = dataset_train.data[:64]
        dataset_train.targets = dataset_train.targets[:64]
        dataset_test.data = dataset_test.data[:64]
        dataset_test.targets = dataset_test.targets[:64]
        
    train_set_size = int(len(dataset_train) * train_valid_split_ratio)
    valid_set_size = len(dataset_train) - train_set_size
    dataset_train, dataset_valid = random_split(
        dataset_train, [train_set_size, valid_set_size]
    )
    return dataset_train, dataset_valid, dataset_test


def build_fashion_mnist_dataset(
    train_valid_split_ratio: float = 0.8,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Build Fashion MNIST dataset for classification.

    Args:
        train_valid_split_ratio: ratio of training set size to total training data

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    transforms_both = transforms.Compose([transforms.ToTensor()])

    dataset_train = datasets.FashionMNIST(
        "data", train=True, download=True, transform=transforms_both
    )
    dataset_test = datasets.FashionMNIST(
        "data", train=False, download=True, transform=transforms_both
    )
    
    train_set_size = int(len(dataset_train) * train_valid_split_ratio)
    valid_set_size = len(dataset_train) - train_set_size
    dataset_train, dataset_valid = random_split(
        dataset_train, [train_set_size, valid_set_size]
    )
    return dataset_train, dataset_valid, dataset_test


# ============================================================================
# Regression Datasets
# ============================================================================

REGRESSION_DATASETS = ["sinusoidal", "sinusoidal2", "polynomial", "gaussian"]


def sinusoidal(x_data: torch.Tensor) -> torch.Tensor:
    """Sinusoidal function on [-1, 1] -> [-1, 1]."""
    return 1 / 0.6 * x_data * torch.cos(x_data)


def polynomial(x_data: torch.Tensor) -> torch.Tensor:
    """Polynomial of fifth order on [-1,1] -> [-1,1]."""
    return (
        1 / 0.45 * (
            (x_data - 0.3) * (x_data + 0.15) * (x_data - 0.7) 
            * (x_data + 0.5) * (x_data + 0.7) - 0.17
        )
    )


def sinusoidal2(x_data: torch.Tensor) -> torch.Tensor:
    """Sinusoidal function on [-1,1] -> [-1,1]."""
    return torch.sqrt(x_data.abs()) * torch.sin(3 * np.pi * x_data)


def gaussian(x_data: torch.Tensor, std: float = 0.25) -> torch.Tensor:
    """Gaussian function on [-1, 1] -> [-1, 1]."""
    return 2 * (torch.exp(-(x_data**2) / (2 * std**2)) - 0.5)


def get_regression_function(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get the regression function by name."""
    functions = {
        "sinusoidal": sinusoidal,
        "sinusoidal2": sinusoidal2,
        "polynomial": polynomial,
        "gaussian": gaussian,
    }
    if name not in functions:
        raise ValueError(f"Unknown regression dataset name: {name}. Choose from {list(functions.keys())}")
    return functions[name]


class RegressionDataset(Dataset):
    """
    General 1D-function-regression dataset.
    Returns (x, y) pairs of shape (1,) each -> (batch_size, 1).
    """

    def __init__(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        n_points: int = 1000,
        x_min: float = -1.0,
        x_max: float = 1.0,
        random_points: bool = False,
    ):
        """
        Args:
            func: Function to regress
            n_points: Number of data points
            x_min: Minimum x value
            x_max: Maximum x value  
            random_points: If True, sample random x values; else use linspace
        """
        self._func = func
        if random_points:
            self.x_data = torch.rand(n_points) * (x_max - x_min) + x_min
        else:
            self.x_data = torch.linspace(x_min, x_max, steps=n_points)
        self.y_data = self._func(self.x_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (repeat(self.x_data[idx], "-> 1"), repeat(self.y_data[idx], "-> 1"))


def get_regression_dataset(
    dataset_name: str,
    n_points: int = 1000,
    random_points: bool = False,
    x_min: float = -1.0,
    x_max: float = 1.0,
) -> RegressionDataset:
    """
    Get a regression dataset by name.
    
    Args:
        dataset_name: One of 'sinusoidal', 'sinusoidal2', 'polynomial', 'gaussian'
        n_points: Number of data points
        random_points: If True, sample random x values
        x_min: Minimum x value
        x_max: Maximum x value
        
    Returns:
        RegressionDataset instance
    """
    regression_function = get_regression_function(dataset_name)
    return RegressionDataset(
        func=regression_function,
        n_points=n_points,
        random_points=random_points,
        x_min=x_min,
        x_max=x_max,
    )
